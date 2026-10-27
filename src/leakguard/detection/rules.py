"""Rule layer: regex + validators + entropy. Deterministic, fast (~1 ms) and explainable.
This is also the *baseline* system in the benchmark."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from . import validators as v
from .entropy import looks_random
from .types import Finding


@dataclass(frozen=True)
class Rule:
    entity: str
    pattern: re.Pattern
    score: float = 0.9
    group: int = 0  # capture group holding the sensitive value
    validator: Callable[[str], bool] | None = None
    validated_score: float = 0.99


def _r(p: str, flags: int = 0) -> re.Pattern:
    return re.compile(p, flags)


RULES: list[Rule] = [
    # ---------------- secrets (formats with distinctive prefixes) ----------------
    Rule("AWS_ACCESS_KEY", _r(r"\b((?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[A-Z0-9]{16})\b"), 0.97, 1),
    Rule("AWS_SECRET_KEY", _r(r"(?i)aws.{0,20}?(?:secret|private)?.{0,10}?(?:key)?.{0,5}?[=:\"'\s]+([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+])"), 0.95, 1),
    Rule("GCP_API_KEY", _r(r"\b(AIza[0-9A-Za-z_\-]{35})\b"), 0.97, 1),
    Rule("GOOGLE_OAUTH_SECRET", _r(r"\b(GOCSPX-[0-9A-Za-z_\-]{28})\b"), 0.97, 1),
    Rule("GITHUB_TOKEN", _r(r"\b((?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{36}|github_pat_[0-9A-Za-z_]{82})\b"), 0.98, 1),
    Rule("GITLAB_TOKEN", _r(r"\b(glpat-[0-9A-Za-z_\-]{20})\b"), 0.98, 1),
    Rule("SLACK_TOKEN", _r(r"\b(xox[baprs]-[0-9A-Za-z\-]{10,72})\b"), 0.97, 1),
    Rule("SLACK_WEBHOOK", _r(r"(https://hooks\.slack\.com/services/T[0-9A-Z]{8,10}/B[0-9A-Z]{8,10}/[0-9A-Za-z]{24})"), 0.98, 1),
    Rule("STRIPE_KEY", _r(r"\b((?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{24,99})\b"), 0.98, 1),
    Rule("ANTHROPIC_KEY", _r(r"\b(sk-ant-(?:api|admin)\d{2}-[0-9A-Za-z_\-]{80,100})"), 0.99, 1),
    Rule("OPENAI_KEY", _r(r"\b(sk-(?:proj-|svcacct-|admin-)?[0-9A-Za-z_\-]{40,164})\b"), 0.93, 1),
    Rule("GROQ_KEY", _r(r"\b(gsk_[0-9A-Za-z]{52})\b"), 0.98, 1),
    Rule("HF_TOKEN", _r(r"\b(hf_[0-9A-Za-z]{34,40})\b"), 0.97, 1),
    Rule("NPM_TOKEN", _r(r"\b(npm_[0-9A-Za-z]{36})\b"), 0.97, 1),
    Rule("SENDGRID_KEY", _r(r"\b(SG\.[0-9A-Za-z_\-]{22}\.[0-9A-Za-z_\-]{43})\b"), 0.98, 1),
    Rule("TWILIO_KEY", _r(r"\b(SK[0-9a-f]{32})\b"), 0.85, 1),
    Rule("AZURE_STORAGE_KEY", _r(r"AccountKey=([A-Za-z0-9+/]{86}==)"), 0.97, 1),
    Rule("PYPI_TOKEN", _r(r"\b(pypi-AgEIcHlwaS5vcmc[0-9A-Za-z_\-]{50,})"), 0.99, 1),
    Rule("DIGITALOCEAN_TOKEN", _r(r"\b(do[pro]_v1_[0-9a-f]{64})\b"), 0.99, 1),
    Rule("SHOPIFY_TOKEN", _r(r"\b(shp(?:at|ss|ca|pa)_[0-9a-fA-F]{32})\b"), 0.98, 1),
    Rule("TELEGRAM_TOKEN", _r(r"(?<![\w:])(\d{8,10}:[0-9A-Za-z_\-]{35})(?![\w\-])"), 0.92, 1),
    Rule("DISCORD_TOKEN", _r(r"(?<![\w.\-])([MNO][0-9A-Za-z_\-]{23,25}\.[0-9A-Za-z_\-]{6}\.[0-9A-Za-z_\-]{27,38})(?![\w\-])"), 0.93, 1),
    Rule("BASIC_AUTH", _r(r"(?i)(Authorization:\s*Basic\s+[A-Za-z0-9+/]{8,}={0,2})"), 0.96, 1),
    Rule("DATADOG_KEY", _r(r"(?i)(?:dd|datadog)[_\-]?(?:api|app)?[_\-]?key[\"']?\s*[:=]\s*[\"']?([0-9a-f]{32}(?:[0-9a-f]{8})?)\b"), 0.95, 1),
    Rule("JWT", _r(r"\b(eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})\b"), 0.85, 1, v.jwt_valid, 0.97),
    Rule("PRIVATE_KEY", _r(r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----[\s\S]{20,}?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----)"), 0.99, 1),
    Rule("DB_CONNECTION_STRING", _r(r"\b((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql)://[^\s:@/]+:[^\s@/]{3,}@[^\s/'\"]+(?:/[^\s'\"]*)?)"), 0.96, 1),
    Rule("PASSWORD", _r(r"(?i)\b(?:password|passwd|pwd|pass|passcode|secret)\b[\"']?\s*(?:[:=]|is|was)\s*[\"']?([^\s\"',;]{6,64})"), 0.80, 1),
    # ---------------- PII ----------------
    Rule("EMAIL", _r(r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"), 0.95, 1),
    Rule("CREDIT_CARD", _r(r"(?<![\d\-])((?:\d[ \-]?){12,18}\d)(?![\d\-])"), 0.40, 1, v.luhn_valid, 0.97),
    Rule("AADHAAR", _r(r"(?<!\d)([2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4})(?!\d)"), 0.40, 1, v.aadhaar_valid, 0.97),
    Rule("PAN", _r(r"\b([A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z])\b"), 0.90, 1, v.pan_valid, 0.95),
    Rule("SSN", _r(r"(?<!\d)(\d{3}-\d{2}-\d{4})(?!\d)"), 0.60, 1, v.ssn_valid, 0.92),
    Rule("IBAN", _r(r"\b([A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?)\b"), 0.40, 1, v.iban_valid, 0.97),
    Rule("PHONE", _r(r"(?<![\w+])((?:\+91[\-\s]?|0)?[6-9]\d{4}[\-\s]?\d{5}|\+\d{1,3}[\-\s]?\(?\d{2,4}\)?[\-\s]?\d{3,4}[\-\s]?\d{3,4}|\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4})(?!\d)"), 0.80, 1),
    Rule("IP_ADDRESS", _r(r"\b((?:\d{1,3}\.){3}\d{1,3})\b"), 0.30, 1, v.ip_is_public, 0.75),
    Rule("PASSPORT", _r(r"(?i)passport\s*(?:no\.?|number|#)?\s*[:#]?\s*([A-Z][0-9]{7}|[A-Z]{2}[0-9]{7})\b"), 0.92, 1),
    Rule("DATE_OF_BIRTH", _r(r"(?i)(?:dob|date of birth|born on|birth ?date)\s*[:\-]?\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]{3,9},? \d{4})"), 0.92, 1),
    Rule("BANK_ACCOUNT", _r(r"(?i)(?:a/?c|account)\s*(?:no\.?|number|#)?\s*[:#]?\s*(\d{9,18})\b"), 0.85, 1),
]

# A generic catch-all for 'key = <random>' assignments that don't match a known format.
GENERIC_ASSIGN = _r(
    r"(?i)\b(?:api[_\-]?key|secret|token|auth|access[_\-]?key|client[_\-]?secret|private[_\-]?key|bearer)\b"
    r"[\"']?\s*(?:[:=]|=>)?\s*[\"']?(?:Bearer\s+)?([A-Za-z0-9_\-+/=.]{20,})"
)

# Placeholders that look like secrets but are documented examples. Matching these must NOT block.
KNOWN_EXAMPLES = {
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "4111111111111111",
    "4242424242424242",
    "123-45-6789",
}
PLACEHOLDER_HINT = re.compile(r"(?i)(example|sample|dummy|placeholder|your[_\-]?(?:api[_\-]?)?key|xxxx|<[^>]+>|\*{4,}|redacted)")


def _is_placeholder(value: str) -> bool:
    return value in KNOWN_EXAMPLES or bool(PLACEHOLDER_HINT.search(value))


def detect_rules(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for rule in RULES:
        for m in rule.pattern.finditer(text):
            value = m.group(rule.group)
            start, end = m.start(rule.group), m.end(rule.group)
            if _is_placeholder(value):
                continue
            score, validated = rule.score, False
            if rule.validator is not None:
                if rule.validator(value):
                    score, validated = rule.validated_score, True
                elif rule.score < 0.5:
                    continue  # pure-shape match that failed its checksum: drop it
            findings.append(Finding(rule.entity, start, end, value, score, "rule", validated=validated))
    for m in GENERIC_ASSIGN.finditer(text):
        value = m.group(1)
        if looks_random(value) and not _is_placeholder(value):
            findings.append(Finding("GENERIC_SECRET", m.start(1), m.end(1), value, 0.75, "rule"))
    return resolve_overlaps(findings)


def resolve_overlaps(findings: list[Finding]) -> list[Finding]:
    """Keep the most specific/confident finding when spans overlap.
    Priority: higher score, then longer span."""
    findings = sorted(findings, key=lambda f: (-f.score, -(f.end - f.start)))
    kept: list[Finding] = []
    for f in findings:
        if any(f.start < k.end and k.start < f.end for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f.start)
