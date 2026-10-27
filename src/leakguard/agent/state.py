"""Explicit agent state carried through the LangGraph closed loop."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Direction = Literal["prompt", "response", "tool_payload"]


class GuardInput(TypedDict, total=False):
    text: str
    direction: Direction
    destination: Literal["internal", "external"]
    role: str
    user_id: str | None
    conversation_id: str | None
    tool_name: str | None  # for tool_payload
    tool_target: str | None  # URL / email / host the tool will send to
    request_ref: str | None  # for responses: id of the prompt request (allows rehydration + origin check)
    known_values: list[str]  # sha256 of values the user supplied in the prompt (response origin check)
    system_prompt: str | None


class GuardState(TypedDict, total=False):
    # goal: deliver the message with zero sensitive-data leakage and minimal false blocks
    goal: str
    request_id: str
    inp: GuardInput
    text: str  # normalised text
    input_sha256: str
    # perception
    findings: list[dict[str, Any]]
    detection_ms: float
    decoded_methods: list[str]
    signals: list[str]
    # semantic state
    semantic: dict[str, Any]
    redacted_preview: str
    signature: str
    # memory / RAG
    retrieved_rules: list[dict[str, Any]]
    precedents: list[dict[str, Any]]
    # reasoning / plan
    policy: dict[str, Any]
    laya: dict[str, Any] | None
    decision: dict[str, Any]
    plan: list[str]
    # actions
    tool_results: list[dict[str, Any]]
    output_text: str | None
    placeholders: dict[str, str]  # placeholder -> entity (never the raw value)
    hitl_case_id: str | None
    # outcome
    status: str
    final_action: str
    error: str | None
    trace: list[dict[str, Any]]
    latency_ms: float
    t0: float
