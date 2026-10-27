from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class InspectRequest(BaseModel):
    text: str = Field(..., description="Prompt, model response or serialized tool-call arguments")
    direction: Literal["prompt", "response", "tool_payload"] = "prompt"
    destination: Literal["internal", "external"] = "external"
    role: str = "employee"
    user_id: str | None = None
    conversation_id: str | None = None
    tool_name: str | None = None
    tool_target: str | None = Field(None, description="URL / email / host a tool will send data to")
    request_ref: str | None = Field(None, description="For responses: request_id of the guarded prompt (enables re-hydration)")
    system_prompt: str | None = None


class InspectResponse(BaseModel):
    request_id: str
    action: str
    status: str
    output_text: str | None
    findings: list[dict[str, Any]]
    citations: list[str]
    rationale: str
    confidence: float
    laya: dict[str, Any] | None
    hitl_case_id: str | None
    latency_ms: float
    trace: list[dict[str, Any]]


class ResolveRequest(BaseModel):
    reviewer: str
    action: Literal["allow", "redact", "block"]
    note: str = Field(..., min_length=3, description="Required justification (stored in the audit log)")


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None
