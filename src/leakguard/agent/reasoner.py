"""Reason / plan: combine the deterministic policy floor with Laya's calibrated judgement and precedents.

Rules of combination (the agent's 'constitution'):
1. Policy is evaluated first. A *hard* rule is a floor nothing can loosen.
2. If the policy already says hard-block, or the message is not a natural-language prompt and already has findings,
   Laya is skipped (it cannot change the outcome; saves ~0.6 s on CPU).
3. Laya's intent probabilities (exfiltration / injection) feed the policy's INT-* rules. The band between the
   escalate and block thresholds is exactly "Laya is unsure" -> human review.
4. Laya may *relax* a bulk-PII escalation (PII-005) to redaction when it is confident the request is benign.
5. If detectors found nothing but Laya is very sure the text is sensitive, escalate (detector miss).
6. An exact reviewer precedent resolves a soft escalation the same way a human did before.
Every final decision cites policy rule IDs and AGENT-* reason codes (checked for groundedness in eval).
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..decision.laya_client import DecisionModel
from ..policy.engine import Policy, PolicyInput, evaluate
from .guardrails import enforce_floor

AGENT_CODES = {
    "AGENT-LAYA-RELAX": "Laya was confident the request is benign; a bulk-data escalation is handled by redaction.",
    "AGENT-RESIDUAL": "Laya found sensitive content the detectors did not mark.",
    "AGENT-PRECEDENT": "A reviewer resolved an identical situation before; the same verdict is applied.",
    "AGENT-SHORTCUT": "Hard policy block; the decision model was not needed.",
    "AGENT-FAILCLOSED": "A component failed; the guard failed closed.",
    "AGENT-VERIFY": "Post-action verification found sensitive data in the output.",
    "AGENT-SHADOW": "Shadow mode: the decision was logged but not enforced.",
}

RELAX_MAX_INTENT = 0.10   # Laya must be this sure the request is benign to relax a bulk-PII escalation


def reason(pinput: PolicyInput, policy: Policy, model: DecisionModel | None, model_text: str,
           precedents: list[dict[str, Any]], signature: str) -> dict[str, Any]:
    pre = evaluate(pinput, policy)
    codes: list[str] = []
    laya: dict[str, Any] | None = None

    if pre.action == "block" and pre.hard:
        codes.append("AGENT-SHORTCUT")
        post, final = pre, "block"
    else:
        # Laya is consulted only where its answer can change the outcome: intent on natural-language prompts,
        # or a residual-sensitivity check when the detectors found nothing.
        needed = (pinput.direction == "prompt" and pinput.text_kind == "prose") or not pinput.entities
        if model is not None and needed:
            v = model.decide(model_text)
            laya = v.to_dict()
            pinput.intents = {"exfiltration": v.exfiltration, "injection": v.injection}
        post = evaluate(pinput, policy)
        final = post.action

        if laya is not None:
            escalations = {m["id"] for m in post.matched if m["action"] == "escalate"}
            if final == "escalate" and not post.hard and escalations == {"PII-005"} \
                    and max(laya["injection"], laya["exfiltration"]) <= RELAX_MAX_INTENT:
                final = "redact"
                codes.append("AGENT-LAYA-RELAX")
            if not pinput.entities and final == "allow" and laya["residual_sensitive"] >= get_settings().laya_residual_threshold:
                final = "escalate"
                codes.append("AGENT-RESIDUAL")

        exact = next((p for p in precedents if p.get("exact")), None)
        if exact and final == "escalate" and not post.hard:
            final = exact["verdict"]
            codes.append("AGENT-PRECEDENT")

    final = enforce_floor(post.action, post.hard, final)
    # 1.0 for purely rule-driven outcomes; otherwise Laya's certainty about the riskiest intent
    conf = 1.0 if laya is None or "AGENT-PRECEDENT" in codes else float(laya["confidence"])

    rationale = "; ".join(
        [f"{m['id']} ({m['action']}{', hard' if m['hard'] else ''}): {m['description']}" for m in post.matched]
        + [f"{c}: {AGENT_CODES[c]}" for c in codes]
    )
    return {
        "policy_action": post.action, "hard": post.hard, "matched": post.matched, "policy_version": post.policy_version,
        "laya": laya, "codes": codes, "final_action": final, "confidence": round(conf, 4),
        "citations": post.rule_ids + codes, "rationale": rationale, "signature": signature,
        "precedent": next((p for p in precedents if p.get("exact")), None),
    }
