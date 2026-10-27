"""Thin client the Streamlit pages use to talk to the LeakGuard API."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API = os.environ.get("LEAKGUARD_API", "http://localhost:8000").rstrip("/")
KEY = os.environ.get("LEAKGUARD_API_KEY")

ACTION_COLORS = {"allow": "#2e7d32", "redact": "#1565c0", "escalate": "#ef6c00", "block": "#c62828",
                 "pending": "#757575"}


def _h() -> dict[str, str]:
    return {"X-API-Key": KEY} if KEY else {}


def get(path: str, **params: Any) -> Any:
    r = requests.get(f"{API}{path}", params=params, headers=_h(), timeout=30)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict, headers: dict | None = None, timeout: int = 120) -> requests.Response:
    return requests.post(f"{API}{path}", json=body, headers={**_h(), **(headers or {})}, timeout=timeout)


def badge(action: str) -> str:
    c = ACTION_COLORS.get(action, "#555")
    return (f"<span style='background:{c};color:white;padding:2px 10px;border-radius:10px;"
            f"font-weight:600;font-size:0.85rem'>{action.upper()}</span>")


def api_status() -> None:
    try:
        h = get("/healthz")
        try:
            ready = requests.get(f"{API}/readyz", timeout=5).status_code == 200
        except requests.RequestException:
            ready = False
        st.sidebar.success(f"API {h['version']} · {'models ready' if ready else 'models loading…'}")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"API unreachable at {API}: {type(e).__name__}")
    st.sidebar.caption(f"API: `{API}`")
