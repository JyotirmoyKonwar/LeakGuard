"""Read/review endpoints for the dashboard: requests, HITL queue, audit log, policies, stats."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from ... import observability as obs
from ...agent.graph import get_active_policy, set_active_policy
from ...agent.memory import remember
from ...audit import chain
from ...policy.engine import STRICTNESS, load_policy
from ...storage.db import session_scope
from ...storage.models import AuditEvent, DecisionRow, GuardRequest, HitlCase, PolicyVersion, ToolCallRow
from ..deps import require_api_key
from ..schemas import ResolveRequest

router = APIRouter(prefix="/v1", tags=["admin"], dependencies=[Depends(require_api_key)])


def _req_dict(r: GuardRequest, full: bool = False) -> dict[str, Any]:
    d = {"id": r.id, "created_at": r.created_at.isoformat(), "direction": r.direction, "destination": r.destination,
         "role": r.role, "final_action": r.final_action, "status": r.status, "latency_ms": round(r.latency_ms, 1),
         "shadow": r.shadow, "redacted_input": r.redacted_input[:300]}
    if full:
        d["redacted_input"] = r.redacted_input
        d["trace"] = r.trace
        d["findings"] = [{"entity": f.entity, "category": f.category, "severity": f.severity, "score": round(f.score, 3),
                          "source": f.source, "preview": f.preview, "decoded_from": f.decoded_from} for f in r.findings]
        if r.decision:
            dc = r.decision
            d["decision"] = {"policy_version": dc.policy_version, "policy_action": dc.policy_action, "hard": dc.hard_floor,
                             "matched_rules": dc.matched_rules, "laya": dc.laya, "precedent": dc.precedent,
                             "final_action": dc.final_action, "confidence": dc.confidence, "rationale": dc.rationale}
        d["tool_calls"] = [{"tool": t.tool, "status": t.status, "latency_ms": t.latency_ms, "detail": t.detail}
                           for t in r.tool_calls]
    return d


@router.get("/requests")
def list_requests(limit: int = Query(50, le=500), action: str | None = None, direction: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(GuardRequest).order_by(GuardRequest.created_at.desc()).limit(limit)
        if action:
            q = q.where(GuardRequest.final_action == action)
        if direction:
            q = q.where(GuardRequest.direction == direction)
        return [_req_dict(r) for r in s.execute(q).scalars()]


@router.get("/requests/{request_id}")
def get_request(request_id: str) -> dict:
    with session_scope() as s:
        r = s.get(GuardRequest, request_id)
        if r is None:
            raise HTTPException(404, "request not found")
        return _req_dict(r, full=True)


# ---------------------------------------------------------------------------------------------- HITL
@router.get("/hitl")
def list_cases(status: str = "pending", limit: int = Query(100, le=500)) -> list[dict]:
    with session_scope() as s:
        q = select(HitlCase).order_by(HitlCase.created_at.asc()).limit(limit)
        if status != "all":
            q = q.where(HitlCase.status == status)
        rows = s.execute(q).scalars().all()
        obs.PENDING_REVIEW.set(s.scalar(select(func.count()).select_from(HitlCase).where(HitlCase.status == "pending")) or 0)
        return [{"id": c.id, "request_id": c.request_id, "created_at": c.created_at.isoformat(), "status": c.status,
                 "reason": c.reason, "proposed_action": c.proposed_action, "redacted_text": c.redacted_text,
                 "context": c.context, "reviewer": c.reviewer, "reviewer_action": c.reviewer_action,
                 "reviewer_note": c.reviewer_note} for c in rows]


@router.post("/hitl/{case_id}/resolve")
def resolve_case(case_id: str, body: ResolveRequest) -> dict:
    """Human decision. Recorded in the audit chain, counted as an override if it differs from the agent's
    proposal, and written to memory so identical future situations are resolved the same way (feedback loop)."""
    with session_scope() as s:
        case = s.get(HitlCase, case_id)
        if case is None:
            raise HTTPException(404, "case not found")
        if case.status != "pending":
            raise HTTPException(409, f"case already {case.status}")
        req = s.get(GuardRequest, case.request_id)
        dec = s.execute(select(DecisionRow).where(DecisionRow.request_id == case.request_id)).scalar_one_or_none()
        # guardrail: a reviewer cannot loosen a HARD policy floor either
        if dec and dec.hard_floor and STRICTNESS[body.action] < STRICTNESS[dec.policy_action]:
            raise HTTPException(422, f"hard policy floor '{dec.policy_action}' cannot be loosened by review")
        case.status = {"allow": "approved", "redact": "edited", "block": "rejected"}[body.action]
        case.reviewer, case.reviewer_action, case.reviewer_note = body.reviewer, body.action, body.note
        case.resolved_at = datetime.now(UTC)
        if req is not None:
            req.status = "done"
            req.final_action = body.action
        override = body.action != case.proposed_action
        if override:
            obs.OVERRIDES.labels(case.proposed_action, body.action).inc()
        chain.append(s, "hitl_resolved", {"case_id": case.id, "reviewer": body.reviewer, "action": body.action,
                                          "proposed": case.proposed_action, "override": override, "note": body.note},
                     case.request_id)
        signature, summary = case.signature, case.redacted_text[:500]
        released = case.redacted_text if body.action == "redact" else None
    remember(signature, summary, body.action, body.note, case_id)
    return {"case_id": case_id, "status": "resolved", "action": body.action, "override": override,
            "released_text": released,
            "note": "Resubmit the original request to have it allowed under this precedent." if body.action == "allow" else None}


# ---------------------------------------------------------------------------------------------- audit
@router.get("/audit")
def audit(limit: int = Query(100, le=1000), request_id: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(limit)
        if request_id:
            q = q.where(AuditEvent.request_id == request_id)
        return [{"seq": e.seq, "created_at": e.created_at.isoformat(), "request_id": e.request_id,
                 "event_type": e.event_type, "payload": e.payload, "hash": e.hash, "prev_hash": e.prev_hash}
                for e in s.execute(q).scalars()]


@router.get("/audit/verify")
def audit_verify() -> dict:
    with session_scope() as s:
        return chain.verify_chain(s)


# ---------------------------------------------------------------------------------------------- policies
@router.get("/policies/active")
def active_policy() -> dict:
    p = get_active_policy()
    return {"name": p.name, "version": p.version, "description": p.description, "settings": p.settings,
            "rules": [r.model_dump() for r in p.rules]}


@router.post("/policies")
def upload_policy(body: dict[str, str]) -> dict:
    """Body: {"yaml": "...", "author": "..."} -- validated, versioned, then activated."""
    text = body.get("yaml", "")
    try:
        policy = load_policy(text=text)
    except Exception as e:
        raise HTTPException(422, f"invalid policy: {e}") from e
    if not any(r.match.default for r in policy.rules):
        raise HTTPException(422, "policy must contain a default rule")
    with session_scope() as s:
        for pv in s.execute(select(PolicyVersion).where(PolicyVersion.active)).scalars():
            pv.active = False
        s.add(PolicyVersion(name=policy.name, version=policy.version, yaml_text=text, active=True, author=body.get("author")))
        chain.append(s, "policy_activated", {"name": policy.name, "version": policy.version, "author": body.get("author"),
                                             "rules": len(policy.rules)})
    set_active_policy(policy)
    return {"activated": f"{policy.name}@v{policy.version}", "rules": len(policy.rules)}


@router.get("/policies/yaml")
def policy_yaml() -> dict:
    with session_scope() as s:
        pv = s.execute(select(PolicyVersion).where(PolicyVersion.active)).scalar_one_or_none()
        if pv:
            return {"yaml": pv.yaml_text}
    p = get_active_policy()
    return {"yaml": yaml.safe_dump(p.model_dump(exclude_defaults=True), sort_keys=False)}


# ---------------------------------------------------------------------------------------------- stats
@router.get("/stats")
def stats(last: int = Query(2000, le=20000)) -> dict:
    with session_scope() as s:
        reqs = s.execute(select(GuardRequest).where(GuardRequest.status != "processing")
                         .order_by(GuardRequest.created_at.desc()).limit(last)).scalars().all()
        tools_ = s.execute(select(ToolCallRow.tool, ToolCallRow.status, func.count())
                           .group_by(ToolCallRow.tool, ToolCallRow.status)).all()
        cases = s.execute(select(HitlCase.status, func.count()).group_by(HitlCase.status)).all()
        overrides = s.scalar(select(func.count()).select_from(HitlCase)
                             .where(HitlCase.reviewer_action.is_not(None), HitlCase.reviewer_action != HitlCase.proposed_action)) or 0
        resolved = s.scalar(select(func.count()).select_from(HitlCase).where(HitlCase.reviewer_action.is_not(None))) or 0
    lat = sorted(r.latency_ms for r in reqs)

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else 0.0

    by_action: dict[str, int] = {}
    by_dir: dict[str, dict[str, int]] = {}
    for r in reqs:
        by_action[r.final_action] = by_action.get(r.final_action, 0) + 1
        by_dir.setdefault(r.direction, {}).setdefault(r.final_action, 0)
        by_dir[r.direction][r.final_action] += 1
    tool_total = sum(n for _, _, n in tools_)
    tool_ok = sum(n for _, st, n in tools_ if st == "success")
    n = len(reqs) or 1
    return {
        "requests": len(reqs), "by_action": by_action, "by_direction": by_dir,
        "escalation_rate": round(by_action.get("escalate", 0) / n, 4),
        "block_rate": round(by_action.get("block", 0) / n, 4),
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99),
                       "mean": round(statistics.fmean(lat), 1) if lat else 0.0},
        "tools": {f"{t}:{st}": c for t, st, c in tools_},
        "tool_success_rate": round(tool_ok / tool_total, 4) if tool_total else None,
        "hitl": dict(cases), "override_rate": round(overrides / resolved, 4) if resolved else None,
    }
