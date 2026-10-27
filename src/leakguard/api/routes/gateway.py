"""OpenAI-compatible protected gateway. Point any OpenAI SDK at http://<leakguard>/v1 and every call is:
prompt guard -> forward sanitised prompt (tool) -> response guard -> re-hydrate placeholders -> caller."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header
from starlette.concurrency import run_in_threadpool

from ...agent import tools
from ...agent.graph import run_guard
from ...detection.rules import detect_rules
from ...observability import LLM_TOKENS
from ..deps import require_api_key
from ..schemas import ChatRequest

router = APIRouter(prefix="/v1", tags=["gateway"], dependencies=[Depends(require_api_key)])


def _completion(content: str, model: str, finish: str, meta: dict[str, Any], usage: dict[str, int] | None = None) -> dict:
    return {
        "id": f"chatcmpl-lg-{uuid.uuid4().hex[:20]}", "object": "chat.completion", "created": int(time.time()),
        "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                     "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "leakguard": meta,
    }


def _scrub_history(text: str) -> str:
    """Earlier turns get the fast rules-only pass (they were guarded when first sent)."""
    fs = [f.to_dict(include_text=False) for f in detect_rules(text)]
    return tools.mask(text, fs)["text"] if fs else text


def _pipeline(req: ChatRequest, role: str, destination: str, conversation_id: str | None) -> dict:
    if destination not in ("internal", "external"):
        destination = "external"
    msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=None)
    if last_user is None:
        return _completion("LeakGuard: no user message to guard.", req.model or "leakguard", "stop", {"action": "allow"})
    system_prompt = "\n".join(m.get("content") or "" for m in msgs if m["role"] == "system") or None

    p = run_guard({"text": msgs[last_user]["content"] or "", "direction": "prompt", "destination": destination,
                   "role": role, "user_id": req.user, "conversation_id": conversation_id})
    pd = p["decision"]
    meta: dict[str, Any] = {"prompt_request_id": p["request_id"], "prompt_action": pd["final_action"],
                            "prompt_citations": pd["citations"], "findings": [f["entity"] for f in p["findings"]]}
    if pd["final_action"] == "block":
        return _completion(f"Blocked by LeakGuard policy ({', '.join(pd['citations'])}). Request id {p['request_id']}.",
                           req.model or "leakguard", "content_filter", {**meta, "action": "block"})
    if pd["final_action"] == "escalate" and p.get("status") == "pending_review":
        return _completion(f"Your request is held for human review (case {p.get('hitl_case_id')}). "
                           "You will be able to resubmit once it is approved.",
                           req.model or "leakguard", "content_filter", {**meta, "action": "escalate",
                                                                        "hitl_case_id": p.get("hitl_case_id")})

    safe = []
    for i, m in enumerate(msgs):
        if i == last_user:
            safe.append({"role": "user", "content": p["output_text"] or ""})
        elif m["role"] == "user":
            safe.append({"role": "user", "content": _scrub_history(m.get("content") or "")})
        else:
            safe.append({"role": m["role"], "content": m.get("content") or ""})

    trace: list[dict[str, Any]] = []
    kw = {k: v for k, v in {"temperature": req.temperature, "max_tokens": req.max_tokens}.items() if v is not None}
    try:
        r = tools.run_tool("gateway", "forward_to_llm", trace, messages=safe, **kw)
    except Exception as e:  # fail closed: never retry with the raw prompt, never fall back to another provider
        return _completion(f"LeakGuard: downstream model unavailable ({type(e).__name__}).", req.model or "leakguard",
                           "error", {**meta, "action": "error", "tool_trace": trace})
    LLM_TOKENS.labels("prompt").inc(r["prompt_tokens"])
    LLM_TOKENS.labels("completion").inc(r["completion_tokens"])

    known: list[str] = []  # the LLM only ever saw placeholders, so any raw PII in its answer is 'new'
    resp = run_guard({"text": r["content"], "direction": "response", "destination": destination, "role": role,
                      "request_ref": p["request_id"], "known_values": known, "system_prompt": system_prompt,
                      "conversation_id": conversation_id})
    rd = resp["decision"]
    meta.update(response_request_id=resp["request_id"], response_action=rd["final_action"],
                response_citations=rd["citations"], llm_model=r["model"], tool_trace=trace,
                guard_latency_ms=round(p["latency_ms"] + resp["latency_ms"], 1), llm_latency_ms=round(r["llm_latency_ms"], 1))
    usage = {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
             "total_tokens": r["prompt_tokens"] + r["completion_tokens"]}
    if rd["final_action"] in ("block", "escalate"):
        return _completion(f"LeakGuard withheld the model's answer ({', '.join(rd['citations'])}).", r["model"],
                           "content_filter", {**meta, "action": rd["final_action"]}, usage)
    return _completion(resp["output_text"] or "", r["model"], "stop", {**meta, "action": pd["final_action"]}, usage)


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest, x_leakguard_role: str = Header(default="employee"),
                           x_leakguard_destination: str = Header(default="external"),
                           x_leakguard_conversation: str | None = Header(default=None)) -> dict:
    return await run_in_threadpool(_pipeline, req, x_leakguard_role, x_leakguard_destination, x_leakguard_conversation)


@router.get("/models")
async def models() -> dict:
    from ...config import get_settings

    _, model = get_settings().provider_defaults()
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "leakguard-proxy"}]}
