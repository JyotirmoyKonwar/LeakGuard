"""Shannon-entropy heuristics for generic, format-less secrets."""

from __future__ import annotations

import math
import re
from collections import Counter

HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
B64_RE = re.compile(r"^[A-Za-z0-9+/=_\-]+$")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def looks_random(token: str, min_len: int = 20) -> bool:
    """True if a token looks like a randomly generated credential.

    Hex strings (hashes, git SHAs, UUIDs without dashes) have lower max entropy (4 bits/char)
    and are very common in benign text, so we require more from them.
    """
    if len(token) < min_len:
        return False
    has_upper = any(c.isupper() for c in token)
    has_lower = any(c.islower() for c in token)
    has_digit = any(c.isdigit() for c in token)
    if HEX_RE.match(token):
        return False  # bare hex is far more likely a hash/commit id than a secret
    if not B64_RE.match(token):
        return False
    classes = has_upper + has_lower + has_digit
    if classes < 3:
        return False
    return shannon_entropy(token) >= 4.0
