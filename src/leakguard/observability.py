"""Prometheus metrics + structured JSON logging."""

from __future__ import annotations

import json
import logging
import sys

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("leakguard_requests_total", "Guard decisions", ["direction", "action"])
FINDINGS = Counter("leakguard_findings_total", "Sensitive entities detected", ["entity", "source"])
TOOL_CALLS = Counter("leakguard_tool_calls_total", "Agent tool calls", ["tool", "status"])
ESCALATIONS = Counter("leakguard_escalations_total", "Cases sent to human review", ["reason"])
OVERRIDES = Counter("leakguard_hitl_overrides_total", "Reviewer decisions that changed the agent's proposal", ["from_action", "to_action"])
FAIL_CLOSED = Counter("leakguard_fail_closed_total", "Requests blocked because a component failed", ["stage"])
LATENCY = Histogram("leakguard_latency_seconds", "End-to-end guard latency", ["direction"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5))
STAGE_LATENCY = Histogram("leakguard_stage_latency_seconds", "Per-node latency", ["node"],
                          buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2))
LLM_TOKENS = Counter("leakguard_llm_tokens_total", "Tokens sent to / received from the downstream LLM", ["kind"])
PENDING_REVIEW = Gauge("leakguard_hitl_pending", "Open human-review cases")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in ("request_id", "trace_id", "direction", "action"):
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


def setup_logging(level: str = "INFO") -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    for noisy in ("httpx", "urllib3", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
