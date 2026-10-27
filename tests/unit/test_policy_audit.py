import pytest

from leakguard.audit import chain
from leakguard.policy.engine import EntityHit, PolicyInput, evaluate, load_policy, strictest
from leakguard.storage.db import session_scope
from leakguard.storage.models import AuditEvent

P = load_policy()


def pin(entities=(), direction="prompt", destination="external", role="employee", intents=None, signals=()):
    from leakguard.detection.types import SEVERITY, category_of

    return PolicyInput(direction, destination, role,
                       [EntityHit(e, category_of(e), SEVERITY.get(e, 2), "new" if direction == "response" else "input")
                        for e in entities], intents or {}, set(signals))


def test_strictest_order():
    assert strictest("allow", "redact", "escalate") == "escalate"
    assert strictest("block", "escalate") == "block"


@pytest.mark.parametrize("entities,kw,action,rule", [
    ((), {}, "allow", "DEF-001"),
    (("EMAIL",), {}, "redact", "PII-002"),
    (("EMAIL",), {"destination": "internal"}, "allow", "PII-003"),
    (("AADHAAR",), {"destination": "internal"}, "redact", "PII-004"),
    (("PRIVATE_KEY",), {}, "block", "SEC-001"),
    (("GITHUB_TOKEN",), {}, "redact", "SEC-002"),
    (("GITHUB_TOKEN",), {"direction": "response"}, "block", "SEC-003"),
    (("EMAIL",), {"direction": "tool_payload"}, "escalate", "PII-008"),
    (("PAN",), {"direction": "tool_payload"}, "block", "PII-007"),
    (("EMAIL",), {"direction": "tool_payload", "destination": "internal"}, "allow", "PII-010"),
    (("PHONE",), {"role": "contractor"}, "block", "PII-009"),
    (("EMAIL",) * 8, {}, "escalate", "PII-005"),
    ((), {"intents": {"exfiltration": 0.98}, "signals": ["bulk_request"]}, "block", "INT-001"),
    ((), {"intents": {"exfiltration": 0.96}, "signals": ["bulk_request"]}, "escalate", "INT-002"),
    ((), {"intents": {"injection": 0.99}}, "block", "INT-003"),
    ((), {"intents": {"injection": 0.96}}, "escalate", "INT-004"),
    ((), {"direction": "response", "signals": ["system_prompt_leak"]}, "block", "RESP-001"),
])
def test_policy_matrix(entities, kw, action, rule):
    d = evaluate(pin(entities, **kw), P)
    assert d.action == action
    assert rule in d.rule_ids


def test_unknown_entity_defaults_to_redact():
    d = evaluate(PolicyInput("prompt", "external", "employee", [EntityHit("NEW_THING", "pii", 2)]), P)
    assert d.action == "redact" and "DEF-002" in d.rule_ids


def test_hard_flag():
    assert evaluate(pin(("PRIVATE_KEY",)), P).hard
    assert not evaluate(pin(("EMAIL",)), P).hard


def test_audit_chain_detects_tampering():
    with session_scope() as s:
        for i in range(3):
            chain.append(s, "test_event", {"i": i})
    with session_scope() as s:
        assert chain.verify_chain(s)["valid"]
        ev = s.query(AuditEvent).filter_by(event_type="test_event").order_by(AuditEvent.seq).first()
        ev.payload = {"i": 999}  # an attacker edits history
    with session_scope() as s:
        res = chain.verify_chain(s)
        assert not res["valid"] and res["broken_at_seq"] == ev.seq
        # restore so other tests see a valid chain
        s.get(AuditEvent, ev.seq).payload = {"i": 0}


def test_intent_rules_only_for_prose_prompts():
    hot = {"injection": 0.99, "exfiltration": 0.99}
    assert evaluate(pin(intents=hot), P).action == "block"
    code = PolicyInput("prompt", "external", "employee", [], hot, set(), text_kind="code")
    assert evaluate(code, P).action == "allow"
    assert evaluate(pin(intents=hot, direction="tool_payload"), P).action == "allow"


def test_exfiltration_needs_bulk_cue():
    assert evaluate(pin(intents={"exfiltration": 0.99}), P).action == "allow"
    assert evaluate(pin(intents={"exfiltration": 0.99}, signals=["bulk_request"]), P).action == "block"
