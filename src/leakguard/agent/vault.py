"""Encrypted, expiring store for reversible pseudonymisation (guardrail layer 6)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import delete, select

from ..config import get_settings
from ..storage.db import session_scope
from ..storage.models import VaultEntry

log = logging.getLogger(__name__)
_fernet: Fernet | None = None


def _f() -> Fernet:
    global _fernet
    if _fernet is None:
        key = get_settings().vault_key
        if not key:
            log.warning("LG_VAULT_KEY not set; using an ephemeral key (vault entries die with the process)")
            key = Fernet.generate_key().decode()
        _fernet = Fernet(key.encode())
    return _fernet


def store(request_id: str, mapping: dict[str, str]) -> int:
    """mapping: placeholder -> original value."""
    if not mapping:
        return 0
    exp = datetime.now(UTC) + timedelta(seconds=get_settings().vault_ttl_s)
    with session_scope() as s:
        for ph, val in mapping.items():
            s.add(VaultEntry(request_id=request_id, placeholder=ph,
                             ciphertext=_f().encrypt(val.encode()).decode(), expires_at=exp))
    return len(mapping)


def load(request_id: str) -> dict[str, str]:
    now = datetime.now(UTC)
    out: dict[str, str] = {}
    with session_scope() as s:
        rows = s.execute(select(VaultEntry).where(VaultEntry.request_id == request_id)).scalars().all()
        for r in rows:
            exp = r.expires_at if r.expires_at.tzinfo else r.expires_at.replace(tzinfo=UTC)
            if exp < now:
                continue
            try:
                out[r.placeholder] = _f().decrypt(r.ciphertext.encode()).decode()
            except InvalidToken:
                log.error("vault entry for %s could not be decrypted (key rotated?)", request_id)
    return out


def purge_expired() -> int:
    with session_scope() as s:
        res = s.execute(delete(VaultEntry).where(VaultEntry.expires_at < datetime.now(UTC)))
        return getattr(res, "rowcount", 0) or 0
