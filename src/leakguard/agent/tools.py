"""The agent's tool belt. Every call goes through `run_tool`, which enforces the allowlist,
times the call, catches errors (fail closed) and records a tool-call trace for audit."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..detection.types import SECRET_ENTITIES
from ..llm.client import get_llm, limiter
from ..storage.db import session_scope
from ..storage.models import HitlCase
from . import vault

log = logging.getLogger(__name__)


class ToolDenied(Exception):
    pass


def sha(v: str) -> str:
    return hashlib.sha256(v.encode()).hexdigest()


# ---------------------------------------------------------------------------- transformations
def pseudonymize(text: str, findings: list[dict[str, Any]], request_id: str, reversible: bool = True) -> dict[str, Any]:
    """Replace each sensitive span with a typed placeholder (<EMAIL_1>). Same value -> same placeholder.
    PII originals are vaulted (encrypted, TTL) so the response can be re-hydrated; secrets are NOT vaulted
    (data minimisation: a secret is never needed again)."""
    counters: dict[str, int] = {}
    by_value: dict[str, str] = {}
    vault_map: dict[str, str] = {}
    placeholders: dict[str, str] = {}
    out, last = [], 0
    for f in sorted(findings, key=lambda f: f["start"]):
        if f["start"] < last:
            continue
        val = text[f["start"]:f["end"]]
        ph = by_value.get(val)
        if ph is None:
            counters[f["entity"]] = counters.get(f["entity"], 0) + 1
            ph = f"<{f['entity']}_{counters[f['entity']]}>"
            by_value[val] = ph
            placeholders[ph] = f["entity"]
            if reversible and f["entity"] not in SECRET_ENTITIES:
                vault_map[ph] = val
        out.append(text[last:f["start"]])
        out.append(ph)
        last = f["end"]
    out.append(text[last:])
    stored = vault.store(request_id, vault_map) if reversible else 0
    return {"text": "".join(out), "placeholders": placeholders, "vaulted": stored,
            "known_values": [sha(v) for v in by_value]}


def mask(text: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Irreversible masking (used for logs, HITL views and responses)."""
    out, last = [], 0
    for f in sorted(findings, key=lambda f: f["start"]):
        if f["start"] < last:
            continue
        out += [text[last:f["start"]], f"[{f['entity']}]"]
        last = f["end"]
    out.append(text[last:])
    return {"text": "".join(out)}


def rehydrate(text: str, request_id: str) -> dict[str, Any]:
    """Restore ONLY placeholders this request created (guardrail: a response can't pull other vault entries)."""
    mapping = vault.load(request_id)
    n = 0
    for ph, val in mapping.items():
        if ph in text:
            n += text.count(ph)
            text = text.replace(ph, val)
    return {"text": text, "restored": n}


# ---------------------------------------------------------------------------- side-effect tools
def escalate_to_hitl(request_id: str, reason: str, proposed_action: str, signature: str,
                     redacted_text: str, context: dict[str, Any]) -> dict[str, Any]:
    with session_scope() as s:
        case = HitlCase(request_id=request_id, reason=reason, proposed_action=proposed_action,
                        signature=signature, redacted_text=redacted_text, context=context)
        s.add(case)
        s.flush()
        return {"case_id": case.id}


def notify_webhook(url: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Alert a SOC channel (Slack/Teams-compatible webhook). No-op when not configured."""
    if not url:
        return {"sent": False, "reason": "no webhook configured"}
    r = httpx.post(url, json=payload, timeout=5.0)
    r.raise_for_status()
    return {"sent": True, "status": r.status_code}


def forward_to_llm(messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
    refused = limiter.acquire()
    if refused:
        raise RuntimeError(f"downstream LLM unavailable: {refused}")
    try:
        r = get_llm().chat(messages, **kw)
    except Exception:
        limiter.record(False)
        raise
    limiter.record(True)
    return {"content": r.content, "model": r.model, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "llm_latency_ms": r.latency_ms}


def block(reason: str) -> dict[str, Any]:
    return {"blocked": True, "reason": reason}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "pseudonymize": pseudonymize,
    "mask": mask,
    "rehydrate": rehydrate,
    "escalate_to_hitl": escalate_to_hitl,
    "notify_webhook": notify_webhook,
    "forward_to_llm": forward_to_llm,
    "block": block,
}

# Guardrail layer 4: which tools may run in which stage. Anything else is refused.
ALLOWED: dict[str, set[str]] = {
    "prompt": {"pseudonymize", "mask", "escalate_to_hitl", "notify_webhook", "block"},
    "response": {"mask", "pseudonymize", "rehydrate", "escalate_to_hitl", "notify_webhook", "block"},
    "tool_payload": {"mask", "pseudonymize", "escalate_to_hitl", "notify_webhook", "block"},
    "gateway": {"forward_to_llm"},
}

# what we persist about a call (never raw text)
_SAFE_KEYS = {"placeholders", "vaulted", "restored", "case_id", "sent", "status", "blocked", "reason",
              "model", "prompt_tokens", "completion_tokens", "llm_latency_ms"}


def run_tool(stage: str, name: str, trace: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    entry: dict[str, Any] = {"type": "tool", "tool": name, "stage": stage}
    try:
        if name not in ALLOWED.get(stage, set()):
            raise ToolDenied(f"tool '{name}' is not allowed in stage '{stage}'")
        result = TOOLS[name](**kwargs)
        entry.update(status="success", detail={k: v for k, v in result.items() if k in _SAFE_KEYS})
        return result
    except ToolDenied as e:
        entry.update(status="denied", detail={"error": str(e)})
        raise
    except Exception as e:
        log.exception("tool %s failed", name)
        entry.update(status="error", detail={"error": f"{type(e).__name__}: {e}"[:300]})
        raise
    finally:
        entry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        trace.append(entry)
