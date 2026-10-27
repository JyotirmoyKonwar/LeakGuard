"""Memory / RAG without an extra model: BM25 over (a) policy clauses and (b) past reviewer verdicts.

(a) grounds the decision in the exact policy text that applies; (b) is the feedback loop: once a human has
ruled on a situation, the agent re-uses that precedent instead of escalating the same thing again."""

from __future__ import annotations

import re
import threading
import time

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from ..policy.engine import Policy
from ..storage.db import session_scope
from ..storage.models import MemoryItem

_tok = re.compile(r"[a-z0-9_]+")


def tokenize(s: str) -> list[str]:
    return _tok.findall(s.lower().replace("-", "_"))


class PolicyIndex:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.docs = [f"{r.id} {r.description} {' '.join(r.match.entities or [])} {r.match.category or ''} "
                     f"{' '.join(r.match.directions or [])} {r.match.intent or ''} {r.action}" for r in policy.rules]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs])

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"id": self.policy.rules[i].id, "text": self.policy.rules[i].description,
                 "score": round(float(scores[i]), 3)} for i in top if scores[i] > 0]


class PrecedentMemory:
    """Cached BM25 index over reviewer verdicts; refreshed when new verdicts land."""

    def __init__(self, ttl_s: float = 5.0):
        self.ttl_s = ttl_s
        self._built = 0.0
        self._items: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        self._built = 0.0

    def _refresh(self) -> None:
        if time.monotonic() - self._built < self.ttl_s:
            return
        with session_scope() as s:
            rows = s.execute(select(MemoryItem).order_by(MemoryItem.id.desc()).limit(5000)).scalars().all()
            self._items = [{"signature": r.signature, "summary": r.summary, "verdict": r.verdict,
                            "note": r.note, "case": r.source_case} for r in rows]
        corpus = [tokenize(i["signature"] + " " + i["summary"]) for i in self._items]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._built = time.monotonic()

    def search(self, signature: str, summary: str, k: int = 3) -> list[dict]:
        with self._lock:
            self._refresh()
            # exact situation match first (most recent verdict wins) -- independent of BM25, whose IDF
            # degenerates on tiny corpora
            exact = [{**it, "score": None, "exact": True} for it in self._items if it["signature"] == signature][:1]
            if not self._bm25:
                return exact
            scores = self._bm25.get_scores(tokenize(signature + " " + summary))
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            similar = [{**self._items[i], "score": round(float(scores[i]), 3), "exact": False}
                       for i in top if scores[i] > 0 and self._items[i]["signature"] != signature]
            return (exact + similar)[:k]


precedents = PrecedentMemory()


def remember(signature: str, summary: str, verdict: str, note: str = "", case_id: str | None = None) -> None:
    with session_scope() as s:
        s.add(MemoryItem(signature=signature, summary=summary, verdict=verdict, note=note, source_case=case_id))
    precedents.invalidate()
