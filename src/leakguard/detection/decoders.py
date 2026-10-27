"""De-obfuscation. Attackers (or careless users) hide secrets in base64, hex, URL-encoding,
string concatenation, or with zero-width characters. We normalise text and surface decoded
variants so the detectors can see through them."""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
B64_BLOB = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
HEX_BLOB = re.compile(r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{2}){16,}(?![0-9a-fA-F])")
URLENC = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")
# "AKIA" + "ABCD..." / 'sk-' . 'abc' style concatenations (explicit operator only; commas are JSON/lists)
CONCAT = re.compile(r"""(["'])([^"'\n]{1,80})\1\s*(?:\+|\.)\s*(["'])([^"'\n]{1,80})\3""")


@dataclass
class DecodedSegment:
    start: int  # span in the *normalised* text the segment came from
    end: int
    decoded: str
    method: str


def normalise(text: str) -> str:
    """Remove zero-width chars and fold full-width/compat characters (NFKC).
    Length can change, so detectors run on the normalised text and spans refer to it."""
    return unicodedata.normalize("NFKC", text.translate(ZERO_WIDTH))


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(ch.isprintable() or ch in "\n\r\t" for ch in s) / len(s)


def decode_segments(text: str, max_segments: int = 20) -> list[DecodedSegment]:
    out: list[DecodedSegment] = []
    for m in B64_BLOB.finditer(text):
        blob = m.group(0)
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False)
            dec = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if len(dec) >= 8 and _printable_ratio(dec) > 0.95:
            out.append(DecodedSegment(m.start(), m.end(), dec, "base64"))
    for m in HEX_BLOB.finditer(text):
        try:
            dec = bytes.fromhex(m.group(0)).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if len(dec) >= 8 and _printable_ratio(dec) > 0.95:
            out.append(DecodedSegment(m.start(), m.end(), dec, "hex"))
    for m in URLENC.finditer(text):
        dec = urllib.parse.unquote(m.group(0))
        if dec != m.group(0):
            out.append(DecodedSegment(m.start(), m.end(), dec, "urlencode"))
    for m in CONCAT.finditer(text):
        out.append(DecodedSegment(m.start(), m.end(), m.group(2) + m.group(4), "concat"))
    return out[:max_segments]
