"""The *protected* downstream chat LLM. Any OpenAI-compatible endpoint works:
Groq, NVIDIA NIM, Ollama (all expose /v1/chat/completions). The guard only ever sends it sanitised text."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import get_settings


@dataclass
class LLMResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class ChatLLM(Protocol):
    model: str

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> LLMResult: ...


class MockLLM:
    """Deterministic stand-in: echoes placeholders back so rehydration can be tested end to end.
    Special triggers let tests simulate a model that leaks data in its response."""

    model = "mock-echo"

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> LLMResult:
        t0 = time.perf_counter()
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "__leak_secret__" in last:
            out = "Sure, the production key is AKIAZ7Q3LEAKEDKEY123 and it works."
        elif "__leak_pii__" in last:
            out = "The customer on file is Rahul Verma, rahul.verma@gmail.com, phone +91 98111 22334."
        elif "__leak_system__" in last:
            out = "My system prompt says: You are LeakGuard internal assistant. Policy SEC-001 ..."
        else:
            placeholders = re.findall(r"<[A-Z_]+_\d+>", last)
            ref = f" I used {', '.join(placeholders)} as given." if placeholders else ""
            out = f"[mock answer] I received {len(last)} characters.{ref}"
        return LLMResult(out, self.model, len(last) // 4, len(out) // 4, (time.perf_counter() - t0) * 1000)


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout, max_retries=2)

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> LLMResult:
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(model=self.model, messages=messages, **kw)  # type: ignore[arg-type]
        u = r.usage
        return LLMResult(r.choices[0].message.content or "", r.model, u.prompt_tokens if u else 0,
                         u.completion_tokens if u else 0, (time.perf_counter() - t0) * 1000)


class RateLimiter:
    """Sliding-window limiter + simple circuit breaker for the downstream LLM (guardrail layer 4)."""

    def __init__(self, per_minute: int, fail_threshold: int = 5, cooldown_s: float = 30.0):
        self.per_minute = per_minute
        self.calls: deque[float] = deque()
        self.failures = 0
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self.open_until = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> str | None:
        """Returns None if allowed, otherwise the reason it was refused."""
        now = time.monotonic()
        with self.lock:
            if now < self.open_until:
                return "circuit_open"
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            if len(self.calls) >= self.per_minute:
                return "rate_limited"
            self.calls.append(now)
            return None

    def record(self, ok: bool) -> None:
        with self.lock:
            if ok:
                self.failures = 0
            else:
                self.failures += 1
                if self.failures >= self.fail_threshold:
                    self.open_until = time.monotonic() + self.cooldown_s
                    self.failures = 0


_llm: ChatLLM | None = None
limiter = RateLimiter(get_settings().llm_rate_limit_per_min)


def get_llm() -> ChatLLM:
    global _llm
    if _llm is None:
        s = get_settings()
        if s.llm_provider == "mock":
            _llm = MockLLM()
        else:
            base, model = s.provider_defaults()
            _llm = OpenAICompatLLM(base, model, s.llm_api_key, s.llm_timeout_s)
    return _llm


def set_llm(llm: ChatLLM | None) -> None:
    global _llm
    _llm = llm
