"""Checksum / structure validators. These turn a regex 'maybe' into a high-confidence hit
and are the main tool for keeping the false-block rate low."""

from __future__ import annotations

import base64
import json
import re

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_valid(number: str) -> bool:
    d = digits(number)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def luhn_check_digit(partial: str) -> str:
    for c in "0123456789":
        if luhn_valid(partial + c):
            return c
    raise ValueError("no check digit")


def verhoeff_valid(number: str) -> bool:
    d = digits(number)
    c = 0
    for i, ch in enumerate(reversed(d)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(partial: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(digits(partial))):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def aadhaar_valid(number: str) -> bool:
    d = digits(number)
    # 12 digits, first digit 2-9, Verhoeff checksum.
    return len(d) == 12 and d[0] not in "01" and verhoeff_valid(d)


def iban_valid(iban: str) -> bool:
    s = re.sub(r"\s", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(num) % 97 == 1


def pan_valid(pan: str) -> bool:
    # AAAPL1234C: 4th char is holder type
    return bool(re.fullmatch(r"[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]", pan.upper()))


def ssn_valid(ssn: str) -> bool:
    d = digits(ssn)
    if len(d) != 9:
        return False
    area, group, serial = d[:3], d[3:5], d[5:]
    return area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"


def _b64url_decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def jwt_valid(token: str) -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False
    try:
        header = json.loads(_b64url_decode(parts[0]))
        return isinstance(header, dict) and "alg" in header
    except Exception:
        return False


def ipv4_valid(ip: str) -> bool:
    parts = ip.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def ip_is_public(ip: str) -> bool:
    """Private / loopback / doc ranges are not PII-worthy."""
    if not ipv4_valid(ip):
        return False
    a, b, *_ = (int(p) for p in ip.split("."))
    if a in (10, 127, 0) or a >= 224:
        return False
    if (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or (a == 169 and b == 254):
        return False
    return not ip.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
