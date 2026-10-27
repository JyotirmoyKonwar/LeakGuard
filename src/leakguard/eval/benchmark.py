"""Ablation benchmark: baseline -> improved -> full agent, on the held-out test split.

  rules            regex + validators + entropy + de-obfuscation, then the policy engine     (baseline)
  gliner_base      rules + GLiNER (urchade/gliner_multi_pii-v1, zero-shot), then policy
  agent            full LangGraph agent: detectors + policy + Laya + memory + guardrails + tools

Usage:
  python -m leakguard.eval.benchmark --split test --systems rules,gliner_base,agent --agent-limit 600
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from ..detection.ensemble import Detector
from ..policy.engine import EntityHit, PolicyInput, evaluate, load_policy
from .metrics import breakdown, entity_recall, leaked_values, overlap, span_prf, summarise

log = logging.getLogger("bench")
ROOT = Path(__file__).resolve().parents[3]


def load_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Keep every scenario represented when sub-sampling."""
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r)
    out: list[dict] = []
    for _, g in sorted(by.items()):
        rng.shuffle(g)
        out.extend(g[:max(3, round(n * len(g) / len(rows)))])
    rng.shuffle(out)
    return out[:n]


def _record(ex: dict, final: str, output: str | None, pred: list[dict], latency: float, **extra: Any) -> dict:
    gold = ex["spans"]
    tp, fp, fn = span_prf(gold, pred)
    leaked = leaked_values(ex["text"], gold, output)
    leaked_ids = {(g["start"], g["end"]) for g in leaked}
    hits = [(g["entity"], (g["start"], g["end"]) not in leaked_ids) for g in gold]
    return {"id": ex["id"], "source": ex["source"], "scenario": ex["scenario"], "lang": ex.get("lang", "en"),
            "direction": ex["direction"], "gold_action": ex["gold_action"], "gold_protect": ex["gold_protect"],
            "final": final, "output_changed": output is not None and output != ex["text"],
            "gold_spans_n": len(gold), "leaked_n": len(leaked), "tp": tp, "fp": fp, "fn": fn,
            "latency_ms": latency, "span_hits": hits,
            "missed": [{"entity": g["entity"], "value_len": g["end"] - g["start"]} for g in gold
                       if not any(overlap(g, p) for p in pred)],
            "false_pos": [{"entity": p["entity"], "source": p.get("source")} for p in pred
                          if not any(overlap(g, p) for g in gold)],
            **extra}


def run_detector_system(rows: list[dict], detector: Detector, policy) -> list[dict]:
    from ..agent.tools import mask

    recs = []
    for ex in rows:
        t0 = time.perf_counter()
        res = detector.detect(ex["text"])
        pred = [f.to_dict() for f in res.findings]
        dest = ex["destination"]
        if ex["direction"] == "tool_payload" and ex.get("tool_target"):
            dest = "external"
        pi = PolicyInput(direction=ex["direction"], destination=dest, role=ex["role"],
                         entities=[EntityHit(f["entity"], f["category"], f["severity"], "new" if ex["direction"] == "response" else "input")
                                   for f in pred])
        d = evaluate(pi, policy)
        out = ex["text"] if d.action == "allow" else (mask(res.text, pred)["text"] if d.action == "redact" else None)
        # spans refer to normalised text; for leak checks compare against the original too
        recs.append(_record(ex, d.action, out, pred, (time.perf_counter() - t0) * 1000, citations=d.rule_ids))
    return recs


def run_agent(rows: list[dict]) -> list[dict]:
    from ..agent.graph import run_guard

    recs = []
    for i, ex in enumerate(rows):
        inp = {k: ex[k] for k in ("text", "direction", "destination", "role", "tool_name", "tool_target")}
        t0 = time.perf_counter()
        st = run_guard(inp)
        lat = (time.perf_counter() - t0) * 1000
        d = st["decision"]
        tools_ = [t for t in st.get("trace", []) if t.get("type") == "tool"]
        recs.append(_record(ex, d["final_action"], st.get("output_text"), st.get("findings", []), lat,
                            citations=d["citations"], codes=d["codes"],
                            tools_ok=sum(t["status"] == "success" for t in tools_),
                            tools_err=sum(t["status"] != "success" for t in tools_),
                            laya=d.get("laya") and {k: d["laya"][k] for k in ("exfiltration", "injection", "residual_sensitive", "confidence")}))
        if (i + 1) % 50 == 0:
            log.info("agent: %d/%d", i + 1, len(rows))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", default="rules,gliner_base,agent")
    ap.add_argument("--limit", type=int, default=0, help="sub-sample (stratified) for detector systems; 0 = all")
    ap.add_argument("--agent-limit", type=int, default=600, help="stratified sub-sample for the (slow, CPU) agent")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--from-records", action="store_true",
                    help="recompute summaries/docs from saved records (no model runs)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = load_split(Path(a.data) / f"{a.split}.jsonl")
    if a.limit:
        rows = stratified(rows, a.limit, a.seed)
    policy = load_policy()
    valid = policy.rule_ids | {"DEF-002"}
    from ..agent.reasoner import AGENT_CODES

    valid |= set(AGENT_CODES)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"split": a.split, "n_total": len(rows), "systems": {}}

    if a.from_records:
        recompute_from_records(a.split, out_dir, valid, results)
        return

    for system in a.systems.split(","):
        t0 = time.time()
        if system == "rules":
            recs = run_detector_system(rows, Detector(mode="rules"), policy)
        elif system == "gliner_base":
            recs = run_detector_system(rows, Detector(mode="rules+gliner", gliner_model="urchade/gliner_multi_pii-v1"), policy)
        elif system == "agent":
            tmp = tempfile.mkdtemp()
            os.environ.setdefault("LG_DATABASE_URL", f"sqlite:///{tmp}/bench.db")
            from ..storage.db import init_db

            init_db()
            recs = run_agent(stratified(rows, a.agent_limit, a.seed) if a.agent_limit else rows)
        else:
            raise SystemExit(f"unknown system {system}")
        summary = summarise(recs, valid)
        summary["by_scenario"] = breakdown(recs, "scenario", valid)
        summary["by_lang"] = breakdown(recs, "lang", valid)
        summary["by_direction"] = breakdown(recs, "direction", valid)
        summary["entity_recall"] = entity_recall(recs)
        summary["wall_s"] = round(time.time() - t0, 1)
        results["systems"][system] = summary
        with (out_dir / f"records_{a.split}_{system}.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("%s: %s", system, {k: v for k, v in summary.items() if not isinstance(v, dict)})

    add_paired(results, out_dir, a.split, valid)
    (out_dir / f"summary_{a.split}.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, ROOT / "docs" / "results.md")
    update_readme(results, ROOT / "README.md")
    if a.mlflow:
        log_mlflow(results, out_dir)


def add_paired(results: dict, out_dir: Path, split: str, valid: set[str]) -> None:
    """The agent runs on a stratified sub-sample (it is ~1 s/item on CPU). For a like-for-like comparison,
    re-score every system on exactly the agent's items."""
    agent_path = out_dir / f"records_{split}_agent.jsonl"
    if not agent_path.exists():
        return
    ids = [json.loads(line)["id"] for line in agent_path.open()]
    paired = {}
    for system in results["systems"]:
        path = out_dir / f"records_{split}_{system}.jsonl"
        recs = {json.loads(line)["id"]: json.loads(line) for line in path.open()}
        subset = [recs[i] for i in ids if i in recs]
        if len(subset) == len(ids):
            m = summarise(subset, valid)
            m["by_scenario"] = breakdown(subset, "scenario", valid)
            paired[system] = {k: v for k, v in m.items() if k != "confusion"}
    results["paired"] = {"n": len(ids), "systems": paired}


def recompute_from_records(split: str, out_dir: Path, valid: set[str], results: dict) -> None:
    for system in ("rules", "gliner_base", "agent"):
        path = out_dir / f"records_{split}_{system}.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(line) for line in path.open()]
        summary = summarise(recs, valid)
        summary["by_scenario"] = breakdown(recs, "scenario", valid)
        summary["by_lang"] = breakdown(recs, "lang", valid)
        summary["by_direction"] = breakdown(recs, "direction", valid)
        summary["entity_recall"] = entity_recall(recs)
        old = out_dir / f"summary_{split}.json"
        if old.exists():
            summary["wall_s"] = json.loads(old.read_text())["systems"].get(system, {}).get("wall_s")
        results["systems"][system] = summary
    add_paired(results, out_dir, split, valid)
    (out_dir / f"summary_{split}.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, ROOT / "docs" / "results.md")
    update_readme(results, ROOT / "README.md")


def write_markdown(results: dict, path: Path) -> None:
    cols = [("n", "n"), ("leakage_recall", "Leakage recall"), ("false_block_rate", "False-block rate"),
            ("over_redaction_rate", "Over-redaction"), ("detection_precision", "Det. precision"),
            ("detection_recall", "Det. recall"), ("action_accuracy", "Action acc."), ("protect_accuracy", "Protect acc."),
            ("escalation_rate", "Escalation"), ("groundedness", "Grounded"), ("tool_success_rate", "Tool success"),
            ("latency_ms_p50", "p50 ms"), ("latency_ms_p95", "p95 ms")]
    lines = [f"# Benchmark results ({results['split']} split)", "",
             "Generated by `python -m leakguard.eval.benchmark`. Metric definitions: `src/leakguard/eval/metrics.py`.", "",
             "| System | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, s in results["systems"].items():
        lines.append(f"| {name} | " + " | ".join("–" if s.get(k) is None else str(s.get(k)) for k, _ in cols) + " |")
    if results.get("paired"):
        pr = results["paired"]
        lines += ["", f"## Like-for-like: every system on the agent's {pr['n']} items", "",
                  "| System | " + " | ".join(c[1] for c in cols[1:]) + " |", "|---" * len(cols) + "|"]
        for name, s in pr["systems"].items():
            lines.append(f"| {name} | " + " | ".join("–" if s.get(k) is None else str(s.get(k)) for k, _ in cols[1:]) + " |")
        lines += ["", "| Scenario | " + " | ".join(f"{n} protect" for n in pr["systems"]) + " |",
                  "|---" * (len(pr["systems"]) + 1) + "|"]
        scen = sorted({k for s in pr["systems"].values() for k in s["by_scenario"]})
        for sc in scen:
            lines.append(f"| {sc} | " + " | ".join(str(s["by_scenario"].get(sc, {}).get("protect_accuracy", "–"))
                                                   for s in pr["systems"].values()) + " |")
    for name, s in results["systems"].items():
        lines += ["", f"## {name}: by scenario", "", "| Scenario | n | Leakage recall | False-block | Action acc. | Protect acc. |",
                  "|---|---|---|---|---|---|"]
        for sc, v in s["by_scenario"].items():
            lines.append(f"| {sc} | {v['n']} | {v.get('leakage_recall', '–')} | {v.get('false_block_rate', '–')} | "
                         f"{v.get('action_accuracy', '–')} | {v.get('protect_accuracy', '–')} |")
        lines += ["", "| Language | n | Leakage recall | Det. recall |", "|---|---|---|---|"]
        for lg, v in s["by_lang"].items():
            lines.append(f"| {lg} | {v['n']} | {v.get('leakage_recall', '–')} | {v.get('detection_recall', '–')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


NAMES = {"rules": "Baseline: rules + validators", "gliner_base": "+ GLiNER (zero-shot)",
         "agent": "Full agent (+ policy, Laya, memory, guardrails)"}


def update_readme(results: dict, path: Path) -> None:
    """Replace the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> with the headline table."""
    if results["split"] != "test" or not path.exists():
        return
    rows = ["| System | Leakage recall ↑ | False-block rate ↓ | Injections + jailbreaks stopped ↑ | Action acc. ↑ | "
            "Protect acc. ↑ | p50 / p95 latency (CPU) |", "|---|---|---|---|---|---|---|"]

    def fmt(m: dict, k: str, pct: bool = True) -> str:
        v = m.get(k)
        if v is None:
            return "–"
        return f"{v:.1%}" if pct else (f"{v:.1f}" if v < 10 else f"{v:.0f}")

    def intent_caught(m: dict) -> str:
        sc = [m["by_scenario"][k] for k in ("injection", "policy_jailbreak") if k in m.get("by_scenario", {})]
        n = sum(x["n"] for x in sc)
        return f"{sum(x['stop_rate'] * x['n'] for x in sc) / n:.0%}" if n else "–"

    systems = results.get("paired", {}).get("systems") or results["systems"]
    for name, m in systems.items():
        def f(k: str, pct: bool = True, m: dict = m) -> str:
            return fmt(m, k, pct)
        rows.append(f"| {NAMES.get(name, name)} | {f('leakage_recall')} | {f('false_block_rate')} | {intent_caught(m)} | "
                    f"{f('action_accuracy')} | {f('protect_accuracy')} | {f('latency_ms_p50', False)} / {f('latency_ms_p95', False)} ms |")
    block = "\n".join(["<!-- RESULTS:START -->", *rows, "",
                        f"_Held-out test split built only from Hugging Face datasets; all systems scored on the same "
                        f"{next(iter(systems.values()))['n']} stratified items (full-split numbers for the detector systems are in "
                        "docs/results.md). Regenerate with `make bench`. "
                        "Full tables: [docs/results.md](docs/results.md) · failures: [docs/failure_analysis.md](docs/failure_analysis.md)._",
                        "<!-- RESULTS:END -->"])
    text = path.read_text()
    i, j = text.find("<!-- RESULTS:START -->"), text.find("<!-- RESULTS:END -->")
    if i >= 0 and j > i:
        path.write_text(text[:i] + block + text[j + len("<!-- RESULTS:END -->"):])


def log_mlflow(results: dict, out_dir: Path) -> None:
    import mlflow

    mlflow.set_experiment("leakguard-benchmark")
    for name, s in results["systems"].items():
        with mlflow.start_run(run_name=f"{name}-{results['split']}"):
            mlflow.log_params({"system": name, "split": results["split"], "n": s["n"]})
            mlflow.log_metrics({k: v for k, v in s.items() if isinstance(v, int | float) and v is not None})
            mlflow.log_artifact(str(out_dir / f"summary_{results['split']}.json"))


if __name__ == "__main__":
    main()
