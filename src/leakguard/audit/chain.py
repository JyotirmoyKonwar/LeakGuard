"""Tamper-evident audit log: each event stores sha256(prev_hash + canonical_json(event)).
Editing or deleting any row breaks every hash after it, which `verify_chain` detects."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..storage.models import AuditEvent

GENESIS = "0" * 64
_lock = threading.Lock()  # serialise appends within a process (DB unique constraint guards across processes)


def _canonical(request_id: str | None, event_type: str, payload: dict[str, Any]) -> str:
    return json.dumps({"request_id": request_id, "event_type": event_type, "payload": payload},
                      sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(prev_hash: str, request_id: str | None, event_type: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + _canonical(request_id, event_type, payload)).encode()).hexdigest()


def append(session: Session, event_type: str, payload: dict[str, Any], request_id: str | None = None) -> AuditEvent:
    with _lock:
        last = session.execute(select(AuditEvent).order_by(AuditEvent.seq.desc()).limit(1)).scalar_one_or_none()
        prev = last.hash if last else GENESIS
        # round-trip through JSON so what we hash is exactly what the DB will return
        payload = json.loads(json.dumps(payload, default=str))
        ev = AuditEvent(request_id=request_id, event_type=event_type, payload=payload, prev_hash=prev,
                        hash=compute_hash(prev, request_id, event_type, payload))
        session.add(ev)
        session.flush()
        return ev


def verify_chain(session: Session) -> dict[str, Any]:
    prev = GENESIS
    n = 0
    for ev in session.execute(select(AuditEvent).order_by(AuditEvent.seq)).scalars():
        expected = compute_hash(prev, ev.request_id, ev.event_type, ev.payload)
        if ev.prev_hash != prev or ev.hash != expected:
            return {"valid": False, "events_checked": n, "broken_at_seq": ev.seq}
        prev = ev.hash
        n += 1
    return {"valid": True, "events_checked": n, "head": prev}
