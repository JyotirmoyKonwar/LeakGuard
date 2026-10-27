from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import JSONResponse, Response

from .. import __version__
from ..config import get_settings
from ..observability import setup_logging
from ..storage.db import init_db
from .routes import admin, gateway, guard

log = logging.getLogger(__name__)
_ready = {"models": False}


def _warmup() -> None:
    """Load GLiNER + Laya in the background so the first request isn't slow; /readyz reports progress."""
    try:
        from ..decision.laya_client import get_decision_model
        from ..detection.ensemble import get_detector

        get_detector()
        get_decision_model()
    finally:
        _ready["models"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level)
    init_db()
    threading.Thread(target=_warmup, daemon=True).start()
    log.info("LeakGuard %s up (llm=%s, gliner=%s, laya=%s)", __version__, s.llm_provider, s.use_gliner, s.use_laya)
    yield


app = FastAPI(
    title="LeakGuard",
    version=__version__,
    description="Agentic data-leakage prevention, redaction and policy-enforcement guard for LLM apps.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(guard.router)
app.include_router(gateway.router)
app.include_router(admin.router)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/readyz", tags=["ops"])
def readyz() -> JSONResponse:
    ok = _ready["models"]
    return JSONResponse({"ready": ok}, status_code=200 if ok else 503)


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
