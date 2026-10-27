import base64

import pytest

from leakguard.detection import validators as v
from leakguard.detection.decoders import decode_segments, normalise
from leakguard.detection.ensemble import Detector
from leakguard.detection.entropy import looks_random
from leakguard.detection.rules import detect_rules

# Values below are public test fixtures (documented example numbers / obviously fake keys), written inline.
VALID_CARD = "4532015112830366"
VALID_AADHAAR = "234567890124"


def ents(text):
    return {f.entity for f in detect_rules(text)}


class TestValidators:
    def test_luhn(self):
        assert v.luhn_valid(VALID_CARD)
        assert not v.luhn_valid("4532015112830367")

    def test_verhoeff_aadhaar(self):
        assert v.aadhaar_valid(VALID_AADHAAR)
        assert not v.aadhaar_valid("234567890123")
        assert not v.aadhaar_valid("134567890124")  # cannot start with 0/1

    def test_check_digit_roundtrip(self):
        assert v.verhoeff_valid("23456789012" + v.verhoeff_check_digit("23456789012"))
        assert v.luhn_valid("453201511283036" + v.luhn_check_digit("453201511283036"))

    def test_iban_pan_ssn(self):
        assert v.iban_valid("GB82 WEST 1234 5698 7654 32")
        assert not v.iban_valid("GB82 WEST 1234 5698 7654 33")
        assert v.pan_valid("ABCPE1234F")
        assert not v.ssn_valid("666-12-3456")

    def test_jwt_and_ip(self):
        assert v.jwt_valid("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghij")
        assert not v.ip_is_public("10.1.2.3") and v.ip_is_public("8.8.8.8")


class TestRules:
    @pytest.mark.parametrize("text,entity", [
        ("key AKIAQWERTYUIOPASDFGH here", "AWS_ACCESS_KEY"),
        ("token ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8", "GITHUB_TOKEN"),
        ("stripe " + "sk_live_" + "abcdefghijklmnopqrstuvwx", "STRIPE_KEY"),
        ("url postgres://admin:S3cretPw@db.internal:5432/app", "DB_CONNECTION_STRING"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234567890abcdef\n-----END RSA PRIVATE KEY-----", "PRIVATE_KEY"),
        ("dop_v1_" + "a" * 64, "DIGITALOCEAN_TOKEN"),
        ("shpss_" + "0123456789abcdef" * 2, "SHOPIFY_TOKEN"),
        ("Authorization: Basic YWRtaW46aHVudGVyMg==", "BASIC_AUTH"),
        ("mail me at jane.doe@acme.in", "EMAIL"),
        (f"card {VALID_CARD}", "CREDIT_CARD"),
        (f"aadhaar {VALID_AADHAAR[:4]} {VALID_AADHAAR[4:8]} {VALID_AADHAAR[8:]}", "AADHAAR"),
        ("PAN ABCPE1234F", "PAN"),
        ("call +91 98765 43210", "PHONE"),
        ("password = Tr0ub4dor&3", "PASSWORD"),
    ])
    def test_positive(self, text, entity):
        assert entity in ents(text)

    @pytest.mark.parametrize("text", [
        "docs example AKIAIOSFODNN7EXAMPLE",
        "commit 3f2a9c1e5b7d8f0a1c2e4b6d8f0a2c4e6b8d0f1a",
        "uuid 550e8400-e29b-41d4-a716-446655440000",
        "order 234567890123 shipped",  # 12 digits but fails Verhoeff
        "card 4532015112830367",  # fails Luhn
        "set OPENAI_API_KEY=<your-api-key>",
        "server at 10.0.0.1",
    ])
    def test_hard_negatives(self, text):
        assert ents(text) == set()

    def test_generic_high_entropy_assignment(self):
        assert "GENERIC_SECRET" in ents('api_key = "Zx8Qw2Lp9Rt4Vb7Nm3Kd6Hs1Fg5Jc0Ty"')

    def test_entropy(self):
        assert looks_random("Zx8Qw2Lp9Rt4Vb7Nm3Kd6Hs1Fg5Jc0Ty")
        assert not looks_random("a" * 30)
        assert not looks_random("3f2a9c1e5b7d8f0a1c2e4b6d8f0a2c4e6b8d0f1a")  # hex hash


class TestDeobfuscation:
    def test_zero_width_removed(self):
        assert normalise("gh\u200bp_x") == "ghp_x"

    def test_base64_secret_found(self):
        blob = base64.b64encode(b"postgres://admin:S3cretPw@db.internal:5432/app").decode()
        r = Detector(mode="rules").detect(f"config: {blob}")
        assert any(f.entity == "DB_CONNECTION_STRING" and f.decoded_from == "base64" for f in r.findings)

    def test_concat_segments(self):
        segs = decode_segments('k = "AKIAQWER" + "TYUIOPASDFGH"')
        assert any(s.method == "concat" and s.decoded == "AKIAQWERTYUIOPASDFGH" for s in segs)

    def test_json_fields_not_merged(self):
        text = '{"to": "partner@vendor-mail.com", "body": "hi"}'
        r = Detector(mode="rules").detect(text)
        emails = [f for f in r.findings if f.entity == "EMAIL"]
        assert len(emails) == 1 and r.text[emails[0].start:emails[0].end] == "partner@vendor-mail.com"

    def test_split_key_detected(self):
        r = Detector(mode="rules").detect('k = "AKIAQWER" + "TYUIOPASDFGH"')
        assert any(f.entity == "AWS_ACCESS_KEY" for f in r.findings)


@pytest.mark.ml
def test_gliner_finds_names():
    r = Detector(mode="rules+gliner").detect("Please email Priya Sharma about the refund.")
    assert any(f.entity == "PERSON" for f in r.findings)
