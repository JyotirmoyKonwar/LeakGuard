"""Perception layer entry point: normalise -> de-obfuscate -> rules (+ GLiNER) -> merge -> context scoring."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from ..config import get_settings
from .decoders import decode_segments, normalise
from .entropy import shannon_entropy
from .rules import _is_placeholder, detect_rules, resolve_overlaps
from .types import Finding

log = logging.getLogger(__name__)

# words near a hit that suggest it's test/demo data rather than a live value
SUPPRESS_CONTEXT = re.compile(r"(?i)\b(example|sample|dummy|fake|placeholder|test(?:ing)? (?:key|card|data)|lorem)\b")
BOOST_CONTEXT = re.compile(r"(?i)\b(prod(?:uction)?|live|real|my|our|customer|patient|confidential|internal)\b")

# GLiNER is good at names/addresses, weaker at structured IDs that rules already cover.
# Per-entity minimum score for GLiNER-only findings (set from dev-split error analysis).
GLINER_MIN_SCORE: dict[str, float] = {
    "PERSON": 0.55, "ADDRESS": 0.50, "USERNAME": 0.70, "PASSWORD": 0.60, "GENERIC_SECRET": 0.70,
    "DATE_OF_BIRTH": 0.55, "EMAIL": 0.50, "PHONE": 0.55, "CREDIT_CARD": 0.70, "AADHAAR": 0.70,
    "PAN": 0.70, "GOV_ID": 0.60, "PASSPORT": 0.65, "BANK_ACCOUNT": 0.65, "SSN": 0.70, "IBAN": 0.70,
}


@dataclass
class DetectionResult:
    text: str  # normalised text the spans refer to
    findings: list[Finding]
    latency_ms: float
    detectors: list[str] = field(default_factory=list)
    decoded_methods: list[str] = field(default_factory=list)


def _context_adjust(text: str, f: Finding) -> None:
    window = text[max(0, f.start - 40): f.end + 40]
    if SUPPRESS_CONTEXT.search(window) and not f.validated:
        f.score *= 0.6
        f.meta["context"] = "suppressed"
    elif BOOST_CONTEXT.search(window):
        f.score = min(1.0, f.score + 0.05)
        f.meta["context"] = "boosted"


NATURAL_TOKEN = re.compile(r"^[A-Za-z]+(?:[._\-/][A-Za-z0-9]+)+$")  # region.staging.users.36, service_gateway_2
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
HEX_ONLY = re.compile(r"^(?:0x)?[0-9a-fA-F]+$")
ROLE_WORDS = {"customer", "user", "admin", "client", "team", "support", "manager", "agent", "sir", "madam",
              "developer", "engineer", "owner", "operator", "bot", "assistant", "everyone", "hi", "hello", "dear"}


def plausible(f: Finding) -> bool:
    """Sanity checks for GLiNER-only hits. GLiNER is context-driven: anything assigned to `API_KEY = ...`
    looks like an api key to it. Real credentials are long, high-entropy and not dotted/underscored words."""
    v = f.text.strip().strip("\"'")
    if f.entity in ("GENERIC_SECRET", "PASSWORD"):
        min_len = 8 if f.entity == "GENERIC_SECRET" else 6
        if len(v) < min_len or " " in v or NATURAL_TOKEN.match(v) or UUID_RE.match(v):
            return False
        if f.entity == "GENERIC_SECRET" and (HEX_ONLY.match(v) or shannon_entropy(v) < 3.0):
            return False
    if f.entity in ("PERSON", "USERNAME") and v.lower() in ROLE_WORDS:
        return False
    return len(v) >= 2


def merge(rule_f: list[Finding], model_f: list[Finding]) -> list[Finding]:
    merged = list(rule_f)
    for g in model_f:
        overlap = [r for r in merged if g.start < r.end and r.start < g.end]
        if overlap:
            for r in overlap:  # agreement -> more confidence; keep the rule's precise entity/span
                r.score = min(1.0, max(r.score, g.score) + 0.05)
                if "gliner" not in r.source:
                    r.source = f"{r.source}+gliner"
            continue
        if g.score >= GLINER_MIN_SCORE.get(g.entity, 0.6) and not _is_placeholder(g.text) and plausible(g):
            merged.append(g)
    return resolve_overlaps(merged)


class Detector:
    """Configurable detector. mode='rules' is the baseline; 'rules+gliner' is the improved system."""

    def __init__(self, mode: str | None = None, gliner_model: str | None = None, min_score: float = 0.5):
        s = get_settings()
        self.mode = mode or (s.detector_mode if s.use_gliner else "rules")
        self.gliner_model = gliner_model or s.gliner_model
        self.min_score = min_score
        self._gliner = None
        if self.mode == "rules+gliner":
            try:
                from .gliner_detector import GlinerDetector

                self._gliner = GlinerDetector.get(self.gliner_model, s.gliner_threshold)
            except Exception as e:  # model unavailable -> degrade to rules, but say so
                log.warning("GLiNER unavailable (%s); falling back to rules-only", e)
                self.mode = "rules"

    def detect(self, raw_text: str) -> DetectionResult:
        t0 = time.perf_counter()
        text = normalise(raw_text)
        findings = detect_rules(text)
        detectors = ["rules"]
        if self._gliner is not None:
            findings = merge(findings, self._gliner.detect(text))
            detectors.append(f"gliner:{self.gliner_model}")

        # look inside encoded/obfuscated blobs
        methods = []
        for seg in decode_segments(text):
            inner = detect_rules(seg.decoded)
            for f in inner:
                if f.text in text:  # value already visible in plain text -> the direct pass handles it precisely
                    continue
                methods.append(seg.method)
                findings.append(
                    Finding(f.entity, seg.start, seg.end, text[seg.start:seg.end], min(1.0, f.score + 0.02),
                            f.source, decoded_from=seg.method, validated=f.validated,
                            meta={"decoded_preview": f.masked()})
                )
        findings = resolve_overlaps(findings)
        for f in findings:
            _context_adjust(text, f)
        findings = [f for f in findings if f.score >= self.min_score]
        return DetectionResult(text, findings, (time.perf_counter() - t0) * 1000, detectors, sorted(set(methods)))


_default: Detector | None = None


def get_detector() -> Detector:
    global _default
    if _default is None:
        _default = Detector()
    return _default
