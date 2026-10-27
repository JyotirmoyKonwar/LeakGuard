"""Guardrails that sit around the agent's own reasoning and actions."""

from __future__ import annotations

import re

from ..detection.rules import detect_rules
from ..policy.engine import STRICTNESS

SYSTEM_LEAK_PHRASES = re.compile(r"(?i)(my (?:system )?prompt (?:says|is)|my instructions (?:are|say)|here (?:is|are) my (?:system )?(?:prompt|instructions)|as an ai i was instructed to)")
PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>|\[[A-Z_]+\]")


class GuardrailViolation(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


def check_input(text: str, max_chars: int) -> list[str]:
    """Layer 1 hygiene. Returns signals; raises on hard violations."""
    signals = []
    if len(text) > max_chars:
        raise GuardrailViolation("INPUT-SIZE", f"input of {len(text)} chars exceeds limit {max_chars}")
    if "\x00" in text:
        signals.append("binary_content")
    return signals


def system_prompt_leak(response: str, system_prompt: str | None, ngram: int = 8) -> bool:
    """Layer 2: response reproduces the system prompt (long shared n-gram) or announces doing so."""
    if SYSTEM_LEAK_PHRASES.search(response):
        return True
    if not system_prompt:
        return False
    sw = system_prompt.lower().split()
    rw = " ".join(response.lower().split())
    return any(" ".join(sw[i:i + ngram]) in rw for i in range(0, max(0, len(sw) - ngram + 1)))


def enforce_floor(policy_action: str, hard: bool, proposed: str) -> str:
    """Layer 4 invariant: nothing may loosen a hard policy floor."""
    if hard and STRICTNESS[proposed] < STRICTNESS[policy_action]:
        return policy_action
    return proposed


def verify_output(text: str | None) -> list[str]:
    """Post-action check: re-scan what we are about to release. Any surviving secret or high-severity
    identifier means a transformation failed, so the caller must fail closed."""
    if not text:
        return []
    leftovers = []
    for f in detect_rules(PLACEHOLDER.sub(" ", text)):
        if f.category == "secret" or f.severity >= 3:
            leftovers.append(f.entity)
    return leftovers
