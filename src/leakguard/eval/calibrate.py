"""Calibrate the agent's decision thresholds on the DEV split (never test).

1. Run the detectors and Laya once per dev example and cache the outputs (the expensive part).
2. Grid-search the policy intent thresholds and the residual-sensitive cut-off, re-evaluating the policy
   from the cache (fast).
3. Objective: maximise protect accuracy subject to false-block rate <= --max-fbr; ties -> action accuracy.

    python -m leakguard.eval.calibrate --n 600
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

from ..config import get_settings
from ..decision.laya_client import LayaModel
from ..detection.ensemble import Detector
from ..detection.textkind import bulk_request, text_kind
from ..policy.engine import EntityHit, PolicyInput, evaluate, load_policy
from .benchmark import ROOT, load_split, stratified

log = logging.getLogger("calibrate")


def build_cache(rows: list[dict], cache: Path) -> list[dict]:
    if cache.exists():
        return [json.loads(line) for line in cache.open()]
    from ..agent.tools import mask

    det, laya = Detector(mode="rules+gliner"), LayaModel.get()
    out = []
    with cache.open("w") as f:
        for i, ex in enumerate(rows):
            res = det.detect(ex["text"])
            fs = [x.to_dict() for x in res.findings]
            v = laya.decide(mask(res.text, fs)["text"])
            rec = {"id": ex["id"], "scenario": ex["scenario"], "gold_action": ex["gold_action"],
                   "gold_protect": ex["gold_protect"], "direction": ex["direction"], "destination": ex["destination"],
                   "role": ex["role"], "findings": [(x["entity"], x["category"], x["severity"]) for x in fs],
                   "laya": v.to_dict()}
            f.write(json.dumps(rec) + "\n")
            out.append(rec)
            if (i + 1) % 50 == 0:
                log.info("cached %d/%d", i + 1, len(rows))
    return out


def decide(rec: dict, policy, residual: float) -> str:
    origin = "new" if rec["direction"] == "response" else "input"
    pi = PolicyInput(rec["direction"], rec["destination"], rec["role"],
                     [EntityHit(e, c, s, origin) for e, c, s in rec["findings"]],
                     {"exfiltration": rec["laya"]["exfiltration"], "injection": rec["laya"]["injection"]},
                     signals=set(rec.get("signals", [])), text_kind=rec.get("text_kind", "prose"))
    d = evaluate(pi, policy)
    if not rec["findings"] and d.action == "allow" and rec["laya"]["residual_sensitive"] >= residual:
        return "escalate"
    return d.action


def score(cache: list[dict], policy, residual: float) -> dict[str, float]:
    preds = [decide(r, policy, residual) for r in cache]
    allow = [(r, p) for r, p in zip(cache, preds, strict=True) if r["gold_action"] == "allow"]
    four = [(r, p) for r, p in zip(cache, preds, strict=True) if r["gold_action"] in ("allow", "redact", "escalate", "block")]
    return {
        "fbr": sum(p in ("block", "escalate") for _, p in allow) / max(1, len(allow)),
        "protect": sum((p != "allow") == r["gold_protect"] for r, p in zip(cache, preds, strict=True)) / len(cache),
        "action": sum(p == r["gold_action"] for r, p in four) / max(1, len(four)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-fbr", type=float, default=0.03)
    ap.add_argument("--ignore-text-kind", action="store_true", help="ablation: apply intent rules to all text")
    ap.add_argument("--no-corroboration", action="store_true", help="ablation: exfiltration without the bulk cue")
    ap.add_argument("--cache", default=str(ROOT / "eval" / "results" / "calibration_cache_dev.jsonl"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rows = stratified(load_split(ROOT / "data" / "processed" / "dev.jsonl"), a.n, seed=11)
    cache = build_cache(rows, Path(a.cache))
    texts = {r["id"]: r["text"] for r in rows}
    for r in cache:
        t = texts.get(r["id"], "")
        r["text_kind"] = "prose" if a.ignore_text_kind else text_kind(t)
        r["signals"] = [] if a.no_corroboration else (["bulk_request"] if bulk_request(t) else [])
    policy = load_policy()
    if a.no_corroboration:  # ablation: drop the requirement from the INT rules
        for rule in policy.rules:
            rule.match.requires_signals = None
    base = dict(policy.settings)
    grid = {
        "injection_block": [0.85, 0.90, 0.95, 0.98, 0.995],
        "injection_escalate": [0.50, 0.60, 0.70, 0.80, 0.90, 0.97],
        "exfiltration_block": [0.80, 0.85, 0.90, 0.95, 0.98],
        "exfiltration_escalate": [0.40, 0.55, 0.70, 0.85, 0.95],
        "residual": [0.90, 0.95, 0.99, 1.01],
    }
    results = []
    for combo in itertools.product(*grid.values()):
        cfg = dict(zip(grid, combo, strict=True))
        if cfg["injection_escalate"] >= cfg["injection_block"] or cfg["exfiltration_escalate"] >= cfg["exfiltration_block"]:
            continue
        policy.settings = {**base, **{k: v for k, v in cfg.items() if k != "residual"}}
        results.append((cfg, score(cache, policy, cfg["residual"])))
    policy.settings = base
    current = score(cache, policy, get_settings().laya_residual_threshold)
    feasible = [r for r in results if r[1]["fbr"] <= a.max_fbr] or results
    best = max(feasible, key=lambda r: (r[1]["protect"], r[1]["action"], -r[1]["fbr"]))
    report = {"n": len(cache), "max_fbr": a.max_fbr, "current": {"settings": base, **current},
              "best": {"settings": best[0], **best[1]}}
    out = ROOT / "eval" / "results" / "calibration_dev.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
