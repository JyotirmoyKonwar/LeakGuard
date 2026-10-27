import pytest
from fastapi.testclient import TestClient

from leakguard.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_metrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    m = client.get("/metrics").text
    assert "leakguard_requests_total" in m


def test_inspect(client):
    r = client.post("/v1/guard/inspect", json={"text": "mail jane@acme.in", "direction": "prompt"})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] == "redact" and "<EMAIL_1>" in body["output_text"]
    assert any(t.get("node") == "reason" for t in body["trace"])
    detail = client.get(f"/v1/requests/{body['request_id']}").json()
    assert detail["decision"]["final_action"] == "redact"
    assert detail["tool_calls"][0]["tool"] == "pseudonymize"


def test_gateway_roundtrip_rehydrates(client):
    r = client.post("/v1/chat/completions", json={"model": "x", "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Draft a reply to priya@acme.in"}]})
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["leakguard"]["prompt_action"] == "redact"
    # MockLLM echoes the placeholder; the gateway must put the real value back for the caller
    assert "priya@acme.in" in body["choices"][0]["message"]["content"]


def test_gateway_blocks_secret_leak_in_response(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "__leak_secret__ please"}]})
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert "AKIA" not in body["choices"][0]["message"]["content"]


def test_gateway_blocks_private_key_prompt(client):
    key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234567890abcdef\n-----END RSA PRIVATE KEY-----"
    body = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": key}]}).json()
    assert body["leakguard"]["action"] == "block"


def test_hitl_resolve_flow(client):
    r = client.post("/v1/guard/inspect", json={"text": '{"body": "call meera@acme.in"}', "direction": "tool_payload",
                                               "tool_name": "send_email", "tool_target": "a@outside.io"}).json()
    case_id = r["hitl_case_id"]
    assert r["status"] == "pending_review" and case_id
    assert any(c["id"] == case_id for c in client.get("/v1/hitl").json())
    res = client.post(f"/v1/hitl/{case_id}/resolve", json={"reviewer": "secops", "action": "redact", "note": "mask and send"})
    assert res.json()["override"] is False or res.json()["action"] == "redact"
    assert client.post(f"/v1/hitl/{case_id}/resolve", json={"reviewer": "x", "action": "allow", "note": "again"}).status_code == 409
    stats = client.get("/v1/stats").json()
    assert stats["requests"] > 0 and "latency_ms" in stats


def test_audit_verify_and_policy(client):
    assert client.get("/v1/audit/verify").json()["valid"] is True
    pol = client.get("/v1/policies/active").json()
    assert any(r["id"] == "SEC-001" for r in pol["rules"])
    bad = client.post("/v1/policies", json={"yaml": "name: x\nversion: 1\nrules: []"})
    assert bad.status_code == 422
