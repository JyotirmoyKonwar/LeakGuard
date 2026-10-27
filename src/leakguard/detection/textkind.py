"""Is this text a natural-language request ("prose") or code / config / log ("code")?

Intent questions (injection, exfiltration) are properties of a *request to an assistant*. Zero-shot Laya scores
code and config text as 'instructions to the AI' (dev error analysis, docs/report.md 4.3), so INT-* rules
only apply to prose prompts. Deterministic and explainable on purpose."""

from __future__ import annotations

import re

STRONG_START = re.compile(r"^\s*(import \w|from \S+ import|def \w|class \w+[:(]|curl |wget |SELECT |INSERT |UPDATE |"
                          r"#!|\$ \w|>>> |FROM \S+:|RUN |apiVersion:|git |docker |kubectl |npm |pip )", re.M)
CODE_START = re.compile(r"^\s*(const |let |var |function |public |private |String |export |\{|\[|<\?|<[a-zA-Z]+[ >]|"
                        r"version:|[\w.-]+:\s*$)", re.M)
ONE_ASSIGN = re.compile(r"^\s*[\w.\-\"']+\s*(?::=|[=:])\s*\S+\s*;?\s*$")
PLACEHOLDER = re.compile(r"<[A-Z_]+_\d+>|\[[A-Z_]+\]")
# `key = anything` or `key: single_token` (so "Note: the meeting is at 10" stays prose)
ASSIGN_LINE = re.compile(r"^\s*[\w.\-\"']+\s*(=\s*\S|:\s*\S+\s*$)", re.M)
LOG_LINE = re.compile(r"^\s*(\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\[\d{2}:\d{2}\]|(INFO|WARN|ERROR|DEBUG)\b)", re.M)
SYMBOLS = set("{}[]();=<>$#\\|`")


def text_kind(text: str) -> str:
    t = PLACEHOLDER.sub("x", text).strip()
    if not t:
        return "prose"
    if STRONG_START.search(t) or ONE_ASSIGN.match(t):
        return "code"
    lines = [ln for ln in t.splitlines() if ln.strip()]
    sym_ratio = sum(c in SYMBOLS for c in t) / len(t)
    assign = len(ASSIGN_LINE.findall(t)) / max(1, len(lines))
    words = re.findall(r"[^\W\d_]{2,}", t)
    prose_words = len(words) / max(1, len(t.split()))
    if LOG_LINE.search(t) and len(LOG_LINE.findall(t)) >= max(1, len(lines) // 2):
        return "code"
    if CODE_START.search(t) and (sym_ratio > 0.03 or assign >= 0.3):
        return "code"
    if len(lines) >= 2 and assign >= 0.6:  # multi-line config / YAML / .env
        return "code"
    if sym_ratio > 0.08 or (assign >= 0.5 and prose_words < 0.8):
        return "code"
    return "prose"


# Corroborating cue for exfiltration intent: the request names a bulk quantity / export verb.
BULK_CUE = re.compile(
    r"(?i)\b(all|every|entire|whole|bulk|complete list|full list|dump|export|download|extract|"
    r"todos|todas|completa|alle|gesamte|sämtliche|tous|toutes|entière|tutti|tutte|intero|सभी|पूरा|पूरी)\b")


def bulk_request(text: str) -> bool:
    return bool(BULK_CUE.search(text))
