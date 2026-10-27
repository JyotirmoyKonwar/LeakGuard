"""Metric definitions (also documented in docs/report.md).

leakage recall      share of gold sensitive values that did NOT reach the destination (blocked, held, or removed
                    from the released text). This is end-to-end: detection + decision + transformation.
detection P/R/F1    span-level, a prediction matches a gold span if they overlap (category-aware).
false-block rate    share of gold-'allow' items that were blocked or held for review (user-visible friction).
over-redaction      share of gold-'allow' items that were released but modified.
action accuracy     exact match of final action vs gold action (items with a 4-way gold label).
protect accuracy    binary: did the guard intervene (anything but allow) exactly when it should?
groundedness        share of decisions whose citations are all valid rule IDs / agent codes and include >=1 policy rule.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from ..detection.types import category_of

ACTIONS = ["allow", "redact", "escalate", "block"]


def overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def span_prf(gold: list[dict], pred: list[dict]) -> tuple[int, int, int]:
    """(tp, fp, fn) with category-aware overlap matching."""
    used: set[int] = set()
    tp = 0
    for g in gold:
        for j, p in enumerate(pred):
            if j in used:
                continue
            if overlap(g, p) and category_of(g["entity"]) == category_of(p["entity"]):
                used.add(j)
                tp += 1
                break
    return tp, len(pred) - len(used), len(gold) - tp


def leaked_values(text: str, gold: list[dict], output: str | None) -> list[dict]:
    if output is None:
        return []
    leaked = []
    for g in gold:
        value = text[g["start"]:g["end"]]
        if len(value.strip()) >= 3 and value in output:
            leaked.append(g)
    return leaked


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def summarise(records: list[dict[str, Any]], valid_citations: set[str] | None = None) -> dict[str, Any]:
    """records: one per example with keys gold_action, gold_protect, final, output_changed, gold_spans_n,
    leaked_n, tp, fp, fn, latency_ms, scenario, lang, citations, tools_ok, tools_err."""
    n = len(records)
    gold_spans = sum(r["gold_spans_n"] for r in records)
    leaked = sum(r["leaked_n"] for r in records)
    tp, fp, fn = (sum(r[k] for r in records) for k in ("tp", "fp", "fn"))
    allow_items = [r for r in records if r["gold_action"] == "allow"]
    four_way = [r for r in records if r["gold_action"] in ACTIONS]
    lat = [r["latency_ms"] for r in records]

    out: dict[str, Any] = {
        "n": n,
        "leakage_recall": round(1 - leaked / gold_spans, 4) if gold_spans else None,
        "items_with_leak": sum(1 for r in records if r["leaked_n"] > 0),
        "detection_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "detection_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_block_rate": round(sum(r["final"] in ("block", "escalate") for r in allow_items) / len(allow_items), 4)
        if allow_items else None,
        "over_redaction_rate": round(sum(r["final"] == "redact" for r in allow_items) / len(allow_items), 4)
        if allow_items else None,
        "action_accuracy": round(sum(r["final"] == r["gold_action"] for r in four_way) / len(four_way), 4)
        if four_way else None,
        "protect_accuracy": round(sum((r["final"] != "allow") == r["gold_protect"] for r in records) / n, 4) if n else None,
        "escalation_rate": round(sum(r["final"] == "escalate" for r in records) / n, 4) if n else None,
        "stop_rate": round(sum(r["final"] in ("block", "escalate") for r in records) / n, 4) if n else None,
        "latency_ms_p50": round(pct(lat, 0.5), 2), "latency_ms_p95": round(pct(lat, 0.95), 2),
        "latency_ms_p99": round(pct(lat, 0.99), 2), "latency_ms_mean": round(statistics.fmean(lat), 2) if lat else 0.0,
    }
    if out["detection_precision"] and out["detection_recall"]:
        p, r = out["detection_precision"], out["detection_recall"]
        out["detection_f1"] = round(2 * p * r / (p + r), 4)
    tools_ok = sum(r.get("tools_ok", 0) for r in records)
    tools_err = sum(r.get("tools_err", 0) for r in records)
    if tools_ok + tools_err:
        out["tool_success_rate"] = round(tools_ok / (tools_ok + tools_err), 4)
        out["tool_calls"] = tools_ok + tools_err
    if valid_citations is not None:
        grounded = [all(c in valid_citations for c in r["citations"]) and any(not c.startswith("AGENT-") for c in r["citations"])
                    for r in records if r.get("citations")]
        out["groundedness"] = round(sum(grounded) / len(grounded), 4) if grounded else None
    out["confusion"] = confusion(four_way)
    return out


def confusion(records: list[dict]) -> dict[str, dict[str, int]]:
    m: dict[str, dict[str, int]] = {g: dict.fromkeys(ACTIONS, 0) for g in ACTIONS}
    for r in records:
        m[r["gold_action"]][r["final"]] += 1
    return m


def breakdown(records: list[dict], key: str, valid: set[str] | None = None) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(key))].append(r)
    keep = ("n", "leakage_recall", "detection_recall", "false_block_rate", "action_accuracy", "protect_accuracy",
            "escalation_rate", "stop_rate", "latency_ms_p50")
    return {k: {m: v for m, v in summarise(g, valid).items() if m in keep} for k, g in sorted(groups.items())}


def entity_recall(records: list[dict]) -> dict[str, dict[str, Any]]:
    c: Counter[str] = Counter()
    hit: Counter[str] = Counter()
    for r in records:
        for e, ok in r.get("span_hits", []):
            c[e] += 1
            hit[e] += ok
    return {e: {"n": c[e], "recall": round(hit[e] / c[e], 4)} for e in sorted(c, key=lambda e: -c[e])}
