"""Model 1: GLiNER span-level NER for PII and secrets that rules can't express
(names, addresses, free-form passwords, multilingual text)."""

from __future__ import annotations

import logging
import threading

from .types import Finding

log = logging.getLogger(__name__)

# natural-language label (what GLiNER sees) -> canonical entity
LABELS: dict[str, str] = {
    "person": "PERSON",
    "email": "EMAIL",
    "phone number": "PHONE",
    "address": "ADDRESS",
    "date of birth": "DATE_OF_BIRTH",
    "credit card number": "CREDIT_CARD",
    "passport number": "PASSPORT",
    "bank account number": "BANK_ACCOUNT",
    "password": "PASSWORD",
    "api key": "GENERIC_SECRET",
    "username": "USERNAME",
    "aadhaar number": "AADHAAR",
    "pan number": "PAN",
    "social security number": "SSN",
    "iban": "IBAN",
    "national id number": "GOV_ID",
    "tax id number": "GOV_ID",
    "driver license number": "GOV_ID",
}

# GLiNER's encoder window is ~384 words; chunk longer text on whitespace with overlap.
CHUNK_CHARS = 1500
OVERLAP = 200


class GlinerDetector:
    _instances: dict[str, GlinerDetector] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str, threshold: float = 0.45):
        from gliner import GLiNER  # heavy import, keep lazy

        self.model_name = model_name
        self.threshold = threshold
        self.model = GLiNER.from_pretrained(model_name)
        self.model.eval()

    @classmethod
    def get(cls, model_name: str, threshold: float = 0.45) -> GlinerDetector:
        with cls._lock:
            if model_name not in cls._instances:
                log.info("loading GLiNER model %s", model_name)
                cls._instances[model_name] = cls(model_name, threshold)
            inst = cls._instances[model_name]
            inst.threshold = threshold
            return inst

    def _chunks(self, text: str):
        if len(text) <= CHUNK_CHARS:
            yield 0, text
            return
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_CHARS)
            if end < len(text):
                ws = text.rfind(" ", start + CHUNK_CHARS // 2, end)
                end = ws if ws > 0 else end
            yield start, text[start:end]
            if end >= len(text):
                break
            start = max(end - OVERLAP, start + 1)

    def detect(self, text: str) -> list[Finding]:
        out: list[Finding] = []
        seen: set[tuple[int, int, str]] = set()
        labels = list(LABELS)
        for offset, chunk in self._chunks(text):
            for e in self.model.predict_entities(chunk, labels, threshold=self.threshold):
                ent = LABELS.get(e["label"])
                if ent is None:
                    continue
                s, t = offset + e["start"], offset + e["end"]
                key = (s, t, ent)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Finding(ent, s, t, text[s:t], float(e["score"]), "gliner"))
        return out
