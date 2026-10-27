"""Policy-as-code engine. Pure function of (semantic state, policy) -> decision with cited rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ..config import get_settings

Action = Literal["allow", "redact", "escalate", "block"]
STRICTNESS: dict[str, int] = {"allow": 0, "redact": 1, "escalate": 2, "block": 3}


def strictest(*actions: str) -> str:
    return max(actions, key=lambda a: STRICTNESS[a])


class Match(BaseModel):
    entities: list[str] | None = None
    category: Literal["secret", "pii"] | None = None
    directions: list[str] | None = None
    destinations: list[str] | None = None  # "internal" | "external"
    roles: list[str] | None = None
    text_kinds: list[str] | None = None  # prose | code
    min_severity: int | None = None
    max_severity: int | None = None
    min_count_setting: str | None = None
    origin: Literal["new", "any"] | None = None
    intent: Literal["exfiltration", "injection"] | None = None
    min_prob_setting: str | None = None
    signal: str | None = None
    requires_signals: list[str] | None = None  # all must be present (corroboration)
    default: bool = False


class Rule(BaseModel):
    id: str
    description: str
    match: Match
    action: Action
    hard: bool = False


class Policy(BaseModel):
    name: str
    version: int
    description: str = ""
    settings: dict[str, float] = Field(default_factory=dict)
    rules: list[Rule]

    @property
    def rule_ids(self) -> set[str]:
        return {r.id for r in self.rules}

    def rule(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rule_id), None)


@dataclass
class EntityHit:
    entity: str
    category: str
    severity: int
    origin: str = "input"  # "input" | "new" (appeared in a response but not in the prompt)


@dataclass
class PolicyInput:
    direction: str  # prompt | response | tool_payload
    destination: str  # internal | external
    role: str
    entities: list[EntityHit]
    intents: dict[str, float] = field(default_factory=dict)  # exfiltration / injection probabilities
    signals: set[str] = field(default_factory=set)
    text_kind: str = "prose"


@dataclass
class PolicyDecision:
    action: str
    hard: bool
    matched: list[dict[str, Any]]  # [{id, action, hard, description, entities}]
    policy_version: str

    @property
    def rule_ids(self) -> list[str]:
        return [m["id"] for m in self.matched]


def load_policy(path: str | Path | None = None, text: str | None = None) -> Policy:
    if text is None:
        text = Path(path or get_settings().policy_path).read_text()
    return Policy.model_validate(yaml.safe_load(text))


def _rule_hits(rule: Rule, p: PolicyInput, policy: Policy) -> list[EntityHit] | None:
    """Return the entity hits that triggered the rule ([] for non-entity rules), or None if no match."""
    m = rule.match
    if m.default:
        return None
    if m.directions and p.direction not in m.directions:
        return None
    if m.destinations and p.destination not in m.destinations:
        return None
    if m.roles and p.role not in m.roles:
        return None
    if m.text_kinds and p.text_kind not in m.text_kinds:
        return None
    if m.requires_signals and not set(m.requires_signals) <= p.signals:
        return None
    if m.signal:
        return [] if m.signal in p.signals else None
    if m.intent:
        thr = policy.settings.get(m.min_prob_setting or "", 0.5)
        return [] if p.intents.get(m.intent, 0.0) >= thr else None

    hits = [
        e for e in p.entities
        if (m.entities is None or e.entity in m.entities)
        and (m.category is None or e.category == m.category)
        and (m.min_severity is None or e.severity >= m.min_severity)
        and (m.max_severity is None or e.severity <= m.max_severity)
        and (m.origin in (None, "any") or e.origin == m.origin)
    ]
    if not hits:
        return None
    if m.min_count_setting:
        need = int(policy.settings.get(m.min_count_setting, 10))
        if len(hits) < need:
            return None
    return hits


def evaluate(p: PolicyInput, policy: Policy) -> PolicyDecision:
    matched: list[dict[str, Any]] = []
    for rule in policy.rules:
        hits = _rule_hits(rule, p, policy)
        if hits is None:
            continue
        matched.append({
            "id": rule.id, "action": rule.action, "hard": rule.hard, "description": rule.description,
            "entities": sorted({h.entity for h in hits}),
        })

    # Entities not covered by any rule fall back to redact (fail-safe, never silently allowed).
    covered = {e for m in matched for e in m["entities"]}
    uncovered = sorted({e.entity for e in p.entities} - covered)
    if uncovered and not any(m["action"] == "allow" for m in matched):
        matched.append({"id": "DEF-002", "action": "redact", "hard": False,
                        "description": "Sensitive entity with no explicit rule: redact by default.",
                        "entities": uncovered})

    if not matched:
        default = next(r for r in policy.rules if r.match.default)
        matched.append({"id": default.id, "action": default.action, "hard": False,
                        "description": default.description, "entities": []})

    # 'allow' rules only win when nothing stricter matched; strictest action wins overall.
    action = strictest(*[m["action"] for m in matched])
    hard = any(m["hard"] and m["action"] == action for m in matched)
    return PolicyDecision(action, hard, matched, f"{policy.name}@v{policy.version}")
