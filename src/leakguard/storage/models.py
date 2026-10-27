"""Persistent state. Raw sensitive values are NEVER stored in plain text:
- previews are masked, inputs are stored redacted + SHA-256 hashed,
- reversible pseudonymisation originals live encrypted in `vault_entries` with a TTL."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid.uuid4().hex


class GuardRequest(Base):
    __tablename__ = "requests"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(16))
    destination: Mapped[str] = mapped_column(String(16))
    role: Mapped[str] = mapped_column(String(32))
    user_id: Mapped[str | None] = mapped_column(String(64))
    input_sha256: Mapped[str] = mapped_column(String(64))
    redacted_input: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="done")  # done | pending_review
    final_action: Mapped[str] = mapped_column(String(16), index=True)
    shadow: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    trace: Mapped[list] = mapped_column(JSON, default=list)

    findings: Mapped[list[FindingRow]] = relationship(back_populates="request", cascade="all, delete-orphan")
    decision: Mapped[DecisionRow | None] = relationship(back_populates="request", cascade="all, delete-orphan")
    tool_calls: Mapped[list[ToolCallRow]] = relationship(back_populates="request", cascade="all, delete-orphan")


class FindingRow(Base):
    __tablename__ = "findings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), index=True)
    entity: Mapped[str] = mapped_column(String(32), index=True)
    category: Mapped[str] = mapped_column(String(8))
    severity: Mapped[int] = mapped_column(Integer)
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))
    preview: Mapped[str] = mapped_column(String(64))
    decoded_from: Mapped[str | None] = mapped_column(String(16))
    request: Mapped[GuardRequest] = relationship(back_populates="findings")


class DecisionRow(Base):
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), unique=True)
    policy_version: Mapped[str] = mapped_column(String(64))
    policy_action: Mapped[str] = mapped_column(String(16))
    hard_floor: Mapped[bool] = mapped_column(Boolean)
    matched_rules: Mapped[list] = mapped_column(JSON)
    laya: Mapped[dict | None] = mapped_column(JSON)
    precedent: Mapped[dict | None] = mapped_column(JSON)
    final_action: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)
    request: Mapped[GuardRequest] = relationship(back_populates="decision")


class ToolCallRow(Base):
    __tablename__ = "tool_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), index=True)
    tool: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16))  # success | error | denied
    latency_ms: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    request: Mapped[GuardRequest] = relationship(back_populates="tool_calls")


class HitlCase(Base):
    __tablename__ = "hitl_cases"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|approved|rejected|edited
    reason: Mapped[str] = mapped_column(Text)
    proposed_action: Mapped[str] = mapped_column(String(16))
    signature: Mapped[str] = mapped_column(String(256))
    redacted_text: Mapped[str] = mapped_column(Text)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    reviewer: Mapped[str | None] = mapped_column(String(64))
    reviewer_action: Mapped[str | None] = mapped_column(String(16))
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryItem(Base):
    """Reviewer verdicts used as precedents by the agent (feedback loop)."""
    __tablename__ = "memory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    signature: Mapped[str] = mapped_column(String(256), index=True)
    summary: Mapped[str] = mapped_column(Text)
    verdict: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text, default="")
    source_case: Mapped[str | None] = mapped_column(String(32))


class VaultEntry(Base):
    __tablename__ = "vault_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(32), index=True)
    placeholder: Mapped[str] = mapped_column(String(64))
    ciphertext: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AuditEvent(Base):
    __tablename__ = "audit_log"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    request_id: Mapped[str | None] = mapped_column(String(32), index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)


class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    yaml_text: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    author: Mapped[str | None] = mapped_column(String(64))
