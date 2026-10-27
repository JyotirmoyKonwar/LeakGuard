"""The closed-loop guard agent, as an explicit LangGraph state machine.

observe -> perceive -> semantic_state -> recall (memory/RAG) -> reason (policy + Laya) -> act (tools)
        -> verify (guardrails) -> outcome (persist + audit) ; any failure -> fail_closed -> outcome
Feedback happens out of band: reviewer verdicts on HITL cases are written to memory (see api/routes/hitl.py)."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, cast

from langgraph.graph import END, StateGraph

from .. import observability as obs
from ..audit import chain
from ..config import get_settings
from ..decision.laya_client import get_decision_model
from ..detection.ensemble import get_detector
from ..detection.textkind import bulk_request, text_kind
from ..policy.engine import EntityHit, Policy, PolicyInput, load_policy, strictest
from ..storage.db import session_scope
from ..storage.models import DecisionRow, FindingRow, GuardRequest, ToolCallRow
from . import guardrails, tools
from .memory import PolicyIndex, precedents
from .reasoner import reason
from .state import GuardInput, GuardState

log = logging.getLogger(__name__)

GOAL = "Deliver the message with zero sensitive-data leakage while minimising false blocks; cite policy for every decision."


class _Ctx:
    """Process-wide resources (loaded once)."""

    policy: Policy | None = None
    index: PolicyIndex | None = None

    @classmethod
    def set_policy(cls, policy: Policy) -> None:
        cls.policy, cls.index = policy, PolicyIndex(policy)

    @classmethod
    def get(cls) -> tuple[Policy, PolicyIndex]:
        if cls.policy is None:
            cls.set_policy(load_policy())
        assert cls.policy is not None and cls.index is not None
        return cls.policy, cls.index


set_active_policy = _Ctx.set_policy


def get_active_policy() -> Policy:
    return _Ctx.get()[0]


def node(name: str) -> Callable:
    """Wrap a node: time it, append to trace, convert exceptions into a fail-closed route."""

    def deco(fn: Callable[[GuardState], dict[str, Any]]) -> Callable[[GuardState], dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(state: GuardState) -> dict[str, Any]:
            t0 = time.perf_counter()
            trace = list(state.get("trace", []))
            try:
                upd = fn(state) or {}
                summary = upd.pop("_summary", None)
                trace.extend(upd.pop("_tool_trace", []))
                status = "ok"
            except Exception as e:
                log.exception("node %s failed", name)
                upd, summary, status = {"error": f"{name}: {type(e).__name__}: {e}"[:400]}, None, "error"
            ms = (time.perf_counter() - t0) * 1000
            obs.STAGE_LATENCY.labels(name).observe(ms / 1000)
            trace.append({"type": "node", "node": name, "ms": round(ms, 3), "status": status, "summary": summary})
            upd["trace"] = trace
            return upd

        return wrapper

    return deco


# ------------------------------------------------------------------------------------------- nodes
@node("observe")
def observe(state: GuardState) -> dict[str, Any]:
    inp = state["inp"]
    s = get_settings()
    signals = guardrails.check_input(inp["text"], s.max_input_chars)
    rid = state.get("request_id") or uuid.uuid4().hex
    sha = hashlib.sha256(inp["text"].encode()).hexdigest()
    with session_scope() as db:  # register the request first so tool side effects can reference it
        db.add(GuardRequest(id=rid, conversation_id=inp.get("conversation_id"), direction=inp["direction"],
                            destination=inp["destination"], role=inp["role"], user_id=inp.get("user_id"),
                            input_sha256=sha, redacted_input="", status="processing", final_action="pending"))
        chain.append(db, "request_received", {"direction": inp["direction"], "destination": inp["destination"],
                                              "role": inp["role"], "input_sha256": sha, "chars": len(inp["text"]),
                                              "tool": inp.get("tool_name")}, rid)
    return {"request_id": rid, "input_sha256": sha, "signals": signals, "goal": GOAL,
            "_summary": f"{inp['direction']} -> {inp['destination']} ({len(inp['text'])} chars)"}


@node("perceive")
def perceive(state: GuardState) -> dict[str, Any]:
    inp = state["inp"]
    res = get_detector().detect(inp["text"])
    findings = []
    known = set(inp.get("known_values") or [])
    for f in res.findings:
        d = f.to_dict(include_text=False)
        if inp["direction"] == "response":
            d["origin"] = "input" if tools.sha(f.text) in known else "new"
        else:
            d["origin"] = "input"
        findings.append(d)
        obs.FINDINGS.labels(f.entity, f.source).inc()
    signals = list(state.get("signals", []))
    if inp["direction"] == "response" and guardrails.system_prompt_leak(res.text, inp.get("system_prompt")):
        signals.append("system_prompt_leak")
    if res.decoded_methods:
        signals.append("obfuscation:" + ",".join(res.decoded_methods))
    if bulk_request(res.text):
        signals.append("bulk_request")
    return {"text": res.text, "findings": findings, "detection_ms": res.latency_ms, "signals": signals,
            "decoded_methods": res.decoded_methods,
            "_summary": f"{len(findings)} findings via {'+'.join(d.split(':')[0] for d in res.detectors)}"
                        f" in {res.latency_ms:.1f} ms"}


@node("semantic_state")
def semantic_state(state: GuardState) -> dict[str, Any]:
    inp, findings = state["inp"], state["findings"]
    counts = Counter(f["entity"] for f in findings)
    masked = tools.mask(state["text"], findings)["text"]
    target = inp.get("tool_target") or ""
    destination = inp["destination"]
    if inp["direction"] == "tool_payload" and target:
        internal = any(target.endswith(d) or f"@{d}" in target for d in get_settings().internal_domains)
        destination = "internal" if internal else "external"
    sem: dict[str, Any] = {
        "direction": inp["direction"], "destination": destination, "role": inp["role"],
        "text_kind": text_kind(state["text"]),
        "tool": inp.get("tool_name"), "entity_counts": dict(counts),
        "max_severity": max((f["severity"] for f in findings), default=0),
        "categories": sorted({f["category"] for f in findings}),
        "new_in_response": sorted({f["entity"] for f in findings if f.get("origin") == "new"}),
        "signals": state.get("signals", []),
    }
    signature = (f"dir:{inp['direction']} dest:{destination} role:{inp['role']} "
                 f"ents:{','.join(sorted(counts)) or 'none'} sig:{','.join(sorted(sem['signals'])) or 'none'}")
    return {"semantic": sem, "redacted_preview": masked, "signature": signature,
            "_summary": f"entities={dict(counts)} max_sev={sem['max_severity']} dest={destination} kind={sem['text_kind']}"}


@node("recall")
def recall(state: GuardState) -> dict[str, Any]:
    _, index = _Ctx.get()
    sem = state["semantic"]
    query = f"{sem['direction']} {sem['destination']} {' '.join(sem['entity_counts'])} {' '.join(sem['categories'])}"
    rules = index.search(query, k=3)
    precs = precedents.search(state["signature"], state["redacted_preview"][:500], k=3)
    return {"retrieved_rules": rules, "precedents": precs,
            "_summary": f"rules={[r['id'] for r in rules]} precedents={len(precs)}"
                        f"{' (exact match)' if any(p['exact'] for p in precs) else ''}"}


@node("reason")
def reason_node(state: GuardState) -> dict[str, Any]:
    policy, _ = _Ctx.get()
    sem, findings = state["semantic"], state["findings"]
    pinput = PolicyInput(
        direction=sem["direction"], destination=sem["destination"], role=sem["role"],
        entities=[EntityHit(f["entity"], f["category"], f["severity"], f.get("origin", "input")) for f in findings],
        signals=set(sem["signals"]), text_kind=sem["text_kind"],
    )
    model = get_decision_model()
    d = reason(pinput, policy, model, state["redacted_preview"], state.get("precedents", []), state["signature"])
    plan = plan_for(d["final_action"], sem["direction"], bool(state["inp"].get("request_ref")))
    lv = d["laya"]
    laya_note = (f"laya inj={lv['injection']:.2f} exf={lv['exfiltration']:.2f} res={lv['residual_sensitive']:.2f}"
                 if lv else "laya skipped")
    return {"decision": d, "laya": lv, "plan": plan, "policy": {"version": d["policy_version"]},
            "_summary": f"policy={d['policy_action']}{' (hard)' if d['hard'] else ''} -> final={d['final_action']} "
                        f"conf={d['confidence']} cites={d['citations']} | {laya_note}"}


def plan_for(action: str, direction: str, has_ref: bool) -> list[str]:
    if action == "block":
        return ["block", "notify_webhook"]
    if action == "escalate":
        return ["mask", "escalate_to_hitl", "notify_webhook"]
    steps = []
    if action == "redact":
        steps.append("pseudonymize" if direction == "prompt" else "mask")
    if direction == "response" and has_ref:
        steps.append("rehydrate")
    return steps


def suggested_action(d: dict[str, Any]) -> str:
    """The agent's best concrete suggestion for a reviewer (used to measure human override rate):
    for intent gray-band cases, block if Laya leans malicious else allow; otherwise the strictest
    non-escalate rule; else redact."""
    laya = d.get("laya")
    if laya and any(m["id"].startswith("INT-") and m["action"] == "escalate" for m in d["matched"]):
        return "block" if max(laya["injection"], laya["exfiltration"]) >= 0.75 else "allow"
    concrete = [m["action"] for m in d["matched"] if m["action"] != "escalate"]
    return strictest(*concrete) if concrete else "redact"


@node("act")
def act(state: GuardState) -> dict[str, Any]:
    inp, d = state["inp"], state["decision"]
    stage, rid, findings = inp["direction"], state["request_id"], state["findings"]
    s = get_settings()
    trace: list[dict[str, Any]] = []
    out: str | None = state["text"] if d["final_action"] in ("allow", "redact") else None
    placeholders: dict[str, str] = {}
    case_id = None
    for step in state["plan"]:
        if step == "pseudonymize":
            r = tools.run_tool(stage, "pseudonymize", trace, text=state["text"], findings=findings, request_id=rid)
            out, placeholders = r["text"], r["placeholders"]
        elif step == "mask":
            to_mask = [f for f in findings if stage != "response" or f.get("origin") == "new" or f["category"] == "secret"]
            r = tools.run_tool(stage, "mask", trace, text=state["text"], findings=to_mask)
            if d["final_action"] == "redact":
                out = r["text"]
        elif step == "rehydrate":
            r = tools.run_tool(stage, "rehydrate", trace, text=out or "", request_id=inp["request_ref"])
            out = r["text"]
        elif step == "escalate_to_hitl":
            r = tools.run_tool(stage, "escalate_to_hitl", trace, request_id=rid,
                               reason=d["rationale"][:2000], proposed_action=suggested_action(d),
                               signature=state["signature"], redacted_text=state["redacted_preview"],
                               context={"semantic": state["semantic"], "citations": d["citations"],
                                        "laya": d["laya"], "request_ref": inp.get("request_ref")})
            case_id = r["case_id"]
            obs.ESCALATIONS.labels(",".join(d["codes"]) or d["policy_action"]).inc()
        elif step == "block":
            tools.run_tool(stage, "block", trace, reason=d["rationale"][:500])
        elif step == "notify_webhook":
            if s_url := s.alert_webhook_url:
                with contextlib.suppress(Exception):  # alerting is best-effort; the decision itself holds
                    tools.run_tool(stage, "notify_webhook", trace, url=s_url,
                                   payload={"request_id": rid, "action": d["final_action"], "citations": d["citations"]})
    return {"output_text": out, "placeholders": placeholders, "hitl_case_id": case_id, "_tool_trace": trace,
            "_summary": f"executed {state['plan'] or ['pass-through']}"}


@node("verify")
def verify(state: GuardState) -> dict[str, Any]:
    out = state.get("output_text")
    d = dict(state["decision"])
    if d["final_action"] == "allow" and state["inp"]["direction"] == "prompt":
        # allowed by policy (e.g. internal LLM): released verbatim, nothing to verify
        return {"_summary": "allowed by policy; no transformation to verify"}
    leftovers = guardrails.verify_output(out) if d["final_action"] == "redact" else []
    if leftovers:
        d.update(final_action="block", codes=d["codes"] + ["AGENT-VERIFY"],
                 citations=d["citations"] + ["AGENT-VERIFY"],
                 rationale=d["rationale"] + f"; AGENT-VERIFY: output still contained {leftovers}")
        return {"decision": d, "output_text": None, "_summary": f"FAILED: leftovers {leftovers} -> block"}
    return {"_summary": "output clean"}


@node("fail_closed")
def fail_closed(state: GuardState) -> dict[str, Any]:
    stage = (state.get("error") or "unknown").split(":")[0]
    obs.FAIL_CLOSED.labels(stage).inc()
    d = {"policy_action": "block", "hard": True, "matched": [], "policy_version": "n/a", "laya": None,
         "codes": ["AGENT-FAILCLOSED"], "final_action": "block", "confidence": 1.0, "citations": ["AGENT-FAILCLOSED"],
         "rationale": f"AGENT-FAILCLOSED: {state.get('error')}", "signature": state.get("signature", ""), "precedent": None}
    return {"decision": d, "output_text": None, "plan": ["block"], "_summary": state.get("error")}


@node("outcome")
def outcome(state: GuardState) -> dict[str, Any]:
    inp, d = state["inp"], state["decision"]
    s = get_settings()
    final = d["final_action"]
    shadow = s.shadow_mode
    output = state.get("output_text")
    if shadow and final != "allow":
        d = {**d, "codes": d["codes"] + ["AGENT-SHADOW"]}
        output = inp["text"]
    status = "pending_review" if final == "escalate" and not shadow else "done"
    latency = (time.perf_counter() - state["t0"]) * 1000
    rid = state.get("request_id")
    trace = state.get("trace", [])
    if rid:
        with session_scope() as db:
            req = db.get(GuardRequest, rid)
            if req is not None:
                req.redacted_input = state.get("redacted_preview", "")[:20000]
                req.status, req.final_action, req.shadow, req.latency_ms = status, final, shadow, latency
                req.trace = trace
                for f in state.get("findings", []):
                    db.add(FindingRow(request_id=rid, entity=f["entity"], category=f["category"], severity=f["severity"],
                                      start=f["start"], end=f["end"], score=f["score"], source=f["source"],
                                      preview=f["preview"], decoded_from=f.get("decoded_from")))
                db.add(DecisionRow(request_id=rid, policy_version=d["policy_version"], policy_action=d["policy_action"],
                                   hard_floor=d["hard"], matched_rules=d["matched"], laya=d["laya"],
                                   precedent=d.get("precedent"), final_action=final, confidence=d["confidence"],
                                   rationale=d["rationale"]))
                for t in trace:
                    if t.get("type") == "tool":
                        db.add(ToolCallRow(request_id=rid, tool=t["tool"], status=t["status"],
                                           latency_ms=t["latency_ms"], detail=t.get("detail", {})))
                        obs.TOOL_CALLS.labels(t["tool"], t["status"]).inc()
                chain.append(db, "decision", {"final_action": final, "policy_action": d["policy_action"],
                                              "citations": d["citations"], "confidence": d["confidence"],
                                              "policy_version": d["policy_version"], "shadow": shadow,
                                              "findings": [f"{f['entity']}:{f['preview']}" for f in state.get("findings", [])],
                                              "hitl_case": state.get("hitl_case_id"),
                                              "output_sha256": hashlib.sha256(output.encode()).hexdigest() if output else None}, rid)
    obs.REQUESTS.labels(inp["direction"], final).inc()
    obs.LATENCY.labels(inp["direction"]).observe(latency / 1000)
    return {"decision": d, "final_action": final, "status": status, "output_text": output, "latency_ms": latency,
            "_summary": f"{final} ({status}) in {latency:.1f} ms"}


# ------------------------------------------------------------------------------------------- graph
def _next(target: str) -> Callable[[GuardState], str]:
    return lambda s: "fail_closed" if s.get("error") else target


def build_graph():
    g = StateGraph(GuardState)
    for name, fn in [("observe", observe), ("perceive", perceive), ("semantic_state", semantic_state),
                     ("recall", recall), ("reason", reason_node), ("act", act), ("verify", verify),
                     ("fail_closed", fail_closed), ("outcome", outcome)]:
        g.add_node(name, fn)
    g.set_entry_point("observe")
    order = ["observe", "perceive", "semantic_state", "recall", "reason", "act", "verify", "outcome"]
    for a, b in zip(order[:-2], order[1:-1], strict=True):
        g.add_conditional_edges(a, _next(b), {b: b, "fail_closed": "fail_closed"})
    g.add_conditional_edges("verify", _next("outcome"), {"outcome": "outcome", "fail_closed": "fail_closed"})
    g.add_edge("fail_closed", "outcome")
    g.add_edge("outcome", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_guard(inp: Mapping[str, Any]) -> dict[str, Any]:
    """Run the closed loop on one message. `inp` follows GuardInput; missing destination/role get defaults."""
    full = cast(GuardInput, {"destination": "external", "role": "employee", **inp})
    state: GuardState = {"inp": full, "trace": [], "t0": time.perf_counter()}
    return dict(get_graph().invoke(state))

