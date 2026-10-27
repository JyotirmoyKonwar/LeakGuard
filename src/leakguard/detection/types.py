"""Shared types and the canonical entity taxonomy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Category = Literal["secret", "pii"]

# Canonical entity labels used everywhere (detectors, policies, datasets, metrics).
SECRET_ENTITIES = {
    "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "GCP_API_KEY", "GITHUB_TOKEN", "GITLAB_TOKEN", "SLACK_TOKEN",
    "SLACK_WEBHOOK", "STRIPE_KEY", "OPENAI_KEY", "ANTHROPIC_KEY", "GROQ_KEY", "HF_TOKEN", "NPM_TOKEN",
    "SENDGRID_KEY", "TWILIO_KEY", "AZURE_STORAGE_KEY", "GOOGLE_OAUTH_SECRET", "JWT", "PRIVATE_KEY",
    "DB_CONNECTION_STRING", "PASSWORD", "GENERIC_SECRET", "BASIC_AUTH", "DATADOG_KEY", "DIGITALOCEAN_TOKEN",
    "DISCORD_TOKEN", "PYPI_TOKEN", "SHOPIFY_TOKEN", "TELEGRAM_TOKEN",
}
PII_ENTITIES = {
    "EMAIL", "PHONE", "CREDIT_CARD", "IBAN", "AADHAAR", "PAN", "SSN", "IP_ADDRESS", "PERSON",
    "ADDRESS", "DATE_OF_BIRTH", "PASSPORT", "BANK_ACCOUNT", "USERNAME", "GOV_ID",
}
ALL_ENTITIES = SECRET_ENTITIES | PII_ENTITIES


def category_of(entity: str) -> Category:
    return "secret" if entity in SECRET_ENTITIES else "pii"


# Severity drives the default policy floor (1 = low, 4 = critical).
SEVERITY: dict[str, int] = {
    **{e: 4 for e in SECRET_ENTITIES},
    "GENERIC_SECRET": 3,
    "PASSWORD": 4,
    "CREDIT_CARD": 4, "AADHAAR": 4, "GOV_ID": 4, "SSN": 4, "PASSPORT": 4, "BANK_ACCOUNT": 4, "IBAN": 4,
    "PAN": 3, "DATE_OF_BIRTH": 3,
    "EMAIL": 2, "PHONE": 2, "ADDRESS": 2, "PERSON": 1, "IP_ADDRESS": 1, "USERNAME": 1,
}


@dataclass
class Finding:
    entity: str
    start: int
    end: int
    text: str
    score: float
    source: str  # "rule", "gliner", "rule+gliner"
    decoded_from: str | None = None  # e.g. "base64" if found inside an encoded blob
    validated: bool = False  # passed a checksum/structure validator
    meta: dict = field(default_factory=dict)

    @property
    def category(self) -> Category:
        return category_of(self.entity)

    @property
    def severity(self) -> int:
        return SEVERITY.get(self.entity, 2)

    def masked(self) -> str:
        """Safe preview for logs: never the raw value."""
        t = self.text
        if len(t) <= 4:
            return "*" * len(t)
        return t[:2] + "*" * min(len(t) - 4, 12) + t[-2:]

    def to_dict(self, include_text: bool = False) -> dict:
        d = asdict(self)
        d["category"] = self.category
        d["severity"] = self.severity
        d["preview"] = self.masked()
        if not include_text:
            d.pop("text")
        return d
