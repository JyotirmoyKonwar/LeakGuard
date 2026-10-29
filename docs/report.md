# LeakGuard: Agentic LLM Data-Leakage Prevention, Redaction and Policy-Enforcement Guard

**Final report, HSR-AGAI-048 · Advanced AI (industry-linked agentic AI project) · Jyotirmoy Konwar · 2026**

## 1. Problem

Enterprises adopting LLMs leak sensitive data through three channels:

1. **Prompts:** employees paste customer records, government IDs and credentials into chat assistants.
2. **Responses:** models repeat memorised or tool-retrieved data, or reveal their system prompt.
3. **Agent tool calls:** an autonomous agent emails, posts or uploads data to external destinations.

Industry players such as Matters.ai build AI-assisted platforms to discover and protect this data. The brief asks for a *working, closed-loop agent* that inspects all three channels, detects secrets, PII and policy violations, decides to **block / redact / allow / escalate**, applies safe transformations and logs policy evidence. It must be measured by **leakage recall, false-block rate and latency**, plus agent-level metrics: action and tool success, groundedness, escalation and override rates, and end-to-end latency and cost.

## 2. Data

Only existing Hugging Face datasets are used; no synthetic text was written. The full dataset card is in [dataset.md](dataset.md).

| Source | Role |
|---|---|
| ai4privacy/open-pii-masking-500k | Multilingual PII spans (8 languages incl. Hindi and Telugu) |
| Podric/prowl-secrets-corpus | 31 credential types in code, chat, tickets and logs, plus 13 kinds of look-alike negatives |
| deepset/prompt-injections | Injection vs benign prompts |
| v1adam/Prompt_injection_and_Sensitive_Data_exposure_detection | Benign / sensitive / jailbreak prompts |
| nisaefendioglu/synthetic-sensitive-data-in-source-code-n300 | Source code with secrets and PII, used as tool payloads |

The data is split 60/20/20 by a hash of the example ID. Both models run zero-shot, so **train** is unused. **Dev** is for every threshold and rule decision, and **test** is only for the final numbers. Gold actions are a fixed, documented function of each dataset's own labels, so the evaluation is not circular.

## 3. Method

### 3.1 Perception (baseline → improved)

* **Baseline: rule layer.** About 40 regexes for credential formats and Indian/US/EU identifiers, checksum validators (Luhn, Verhoeff for Aadhaar, IBAN mod-97, JWT header), Shannon-entropy checks for generic secrets, a known-placeholder list, and **de-obfuscation** (base64, hex, URL-encoding, string concatenation, zero-width and full-width characters).
* **Improved: + GLiNER** (`urchade/gliner_multi_pii-v1`, a multilingual span-based NER model), queried with 18 natural-language labels. Rule and model spans are merged:
  * When the two agree, the rule's precise type and span are kept and the score is boosted.
  * GLiNER-only spans must pass per-entity score thresholds and **plausibility filters**, which were added after error analysis (§4.4).

### 3.2 Decision: policy + Laya

* **Policy-as-code:** a YAML file of about 25 rules over entity, category, severity, direction, destination, role, intent and signal.
  * The strictest matching action wins.
  * *Hard* rules are floors.
  * Unknown entities fall back to redact, so nothing is silently allowed.
* **Laya-multilingual** (322M, mmBERT): one forward pass answers four calibrated yes/no questions about the *redacted* message (jailbreak, prompt injection, bulk exfiltration, residual sensitive content).
  * The INT-* rules map the probabilities to actions: above the block threshold → block; in the band between the escalate and block thresholds → **escalate to a human**.
  * Thresholds were tuned on dev (§4.3).
* **Combination rules** (`agent/reasoner.py`):
  * If a hard block applies, Laya is skipped.
  * Laya may relax a bulk-PII escalation to redaction when it is confident the request is benign.
  * Laya may escalate when the detectors found nothing but the content is very likely sensitive.
  * An exact reviewer precedent resolves a soft escalation.
  * The hard floor is always re-applied last.

### 3.3 Agent loop

A LangGraph state machine with an explicit `GuardState` (see [agent_state_action.md](agent_state_action.md)):

`observe → perceive → semantic_state → recall (BM25 policy RAG + reviewer precedents) → reason → act (tools) → verify → outcome`, plus `fail_closed`.

Tools: `pseudonymize` (reversible, Fernet-encrypted vault with TTL), `mask`, `rehydrate` (only this request's placeholders), `escalate_to_hitl`, `notify_webhook`, `forward_to_llm` (rate-limited, circuit-breaker) and `block`. Every call is checked against a per-stage allowlist, timed and persisted.

### 3.4 Guardrails and HITL

Six layers, each with dedicated tests:

1. **Prompt:** de-obfuscation, detection, intent, size limits.
2. **Response:** new-PII and secret blocking, system-prompt-leak detection, scoped re-hydration.
3. **Tool payload:** destination allowlist; field-level rules.
4. **Agent self-guard:** hard floor, fail-closed on any exception, post-action verification, tool allowlist, rate limit and circuit breaker, shadow mode.
5. **Human review queue:** a mandatory justification, override metric, and precedent memory.
6. **Data handling:** no raw values at rest, an encrypted vault, and a SHA-256 hash-chained audit log with a verify endpoint.

### 3.5 System

* FastAPI with an OpenAI-compatible gateway; SQLAlchemy and Alembic on Postgres (SQLite for development).
* A Streamlit dashboard with six pages.
* Prometheus metrics and a provisioned Grafana dashboard; MLflow tracking.
* Docker multi-target images and Docker Compose.
* GitHub Actions CI: lint, mypy, tests, Postgres integration and a Docker build.
* A Hugging Face Space deployment with Neon Postgres.

## 4. Evaluation

All numbers are on the **held-out test split** unless marked *dev*. Metric definitions are in `src/leakguard/eval/metrics.py`:

* **Leakage recall** is end-to-end: the share of gold sensitive values that did not reach the destination.
* **False-block rate (FBR)** is the share of gold-*allow* items that were blocked or held for review.
* **Protect accuracy** asks whether the guard intervened exactly when it should have.

Everything runs on a laptop CPU (8 threads), with no GPU.

### 4.1 Main ablation (all systems on the same 600 stratified test items)

| System | Leakage recall ↑ | FBR ↓ | Injections + jailbreaks stopped ↑ | Action acc. ↑ | Protect acc. ↑ | p50 / p95 latency |
|---|---|---|---|---|---|---|
| Baseline: rules + validators + de-obfuscation | 24.7% | 1.6% | 0% | 57.8% | 57.5% | 0.2 / 0.4 ms |
| + GLiNER zero-shot | 78.5% | 3.3% | 0% | 82.8% | 85.5% | 369 / 487 ms |
| **Full agent** (+ policy, Laya, memory, guardrails, tools) | **80.8%** | 4.9% | **36%** | 80.8% | **86.0%** | 1181 / 2264 ms* |

\* Measured on a busy laptop. An earlier run of the same agent, before the last change, measured 1032 / 1353 ms; GLiNER, whose code didn't change, also slowed from 492 to 844 ms p95 between the two runs. So the difference is machine load, not the change.

On the **full test split** (2,889 items), the detector systems score as follows:

* Rules: leakage recall 27.2%, FBR 1.7%.
* Rules + GLiNER: leakage recall 80.6%, FBR 3.1%, detection recall 85.6%, detection precision 65.9%.

**Agent-level metrics** (600 items):

* **Groundedness 100%:** every decision cites at least one valid policy rule, and every citation exists.
* **Tool success 100%:** 385 tool calls (pseudonymize 256, block 81, mask 38, escalate_to_hitl 10), with zero errors or denials.
* **Escalation rate 1.7%.**
* **No fail-closed events.**
* **Laya was consulted on 481 of 600 requests.** It was skipped for hard blocks and for non-prose or non-prompt traffic that already had findings.

**Cost:** the guard makes no paid API calls; both models run locally. In gateway mode the only paid tokens are the downstream LLM's, and those see only redacted text.

### 4.2 Intent detection on all test prompts (Laya)

To get past the small sample above, the agent was also run on **every** injection, jailbreak, benign and sensitive prompt in the test split:

| Scenario | n | Stopped (block + escalate) | Note |
|---|---|---|---|
| Prompt injection (deepset, en) | 40 | 8 (20%) | 6 of them sent to human review |
| Prompt injection (deepset, de) | 20 | 7 (35%) | multilingual checkpoint |
| Jailbreak (v1adam) | 63 | 24 (38%) | 8 sent to human review |
| Benign prompts (deepset + v1adam) | 114 | 2 (**1.8% false stops**) | |
| "Sensitive" prompts (v1adam, no spans) | 64 | 3 stopped, 37 redacted | 62.5% protected overall |

The detectors alone stop **none** of these, because they contain no PII or secrets. Zero-shot Laya, held to a strict false-block budget, stops 39 of 123 (32%) of them. The dev estimate (75% on 24 items) was optimistic. The gap is a small-sample effect plus mild overfitting of the thresholds to dev, and it is reported as such.

### 4.3 Laya prompt design and calibration (dev)

Laya is zero-shot here, so the *question design* is part of the method. Six benign and five malicious prompts were probed. The table shows the maximum intent probability Laya returned.

| Variant | State given to Laya | Benign (max prob) | Malicious (max prob) | ms/call |
|---|---|---|---|---|
| V1 | JSON with message + findings + **retrieved policy text** | 0.63–1.00 | 0.93–1.00 | 551 |
| V2 | `{"text": msg}`, custom questions | 0.01–1.00 | 0.67–1.00 | 382 |
| V4 | `{"prompt": msg}`, Laya's own `guard_questions()` wording | 0.00–0.04 (+1 outlier) | 0.00–1.00 (misses exfiltration) | 322 |
| **V5 (chosen)** | `{"prompt": msg}`, guard wording + an exfiltration question in the same style | 0.00–0.53 (+1 outlier) | **0.85–1.00** | 503 |
| V7 | V5 + residual + action questions | same as V5 | same as V5 | 1184 |

Findings:

* **(a) Policy text leaks into the judgement.** With the retrieved policy clauses in the state (V1), Laya answered "yes" to injection and exfiltration for everything, because the policy text itself talks about credentials and injection. Laya now sees only the redacted message; policy RAG grounds the *rationale*, not the model input.
* **(b) The action question was dropped.** Extra questions don't change the intent answers but cost about 170 ms each.
* **(c) Zero-shot Laya is unreliable on code, config and logs.** On a 600-item dev sample, secrets-corpus negatives that are pure code or config had a median exfiltration score of 0.75, and about a third scored ≥ 0.9 for injection.
  * Injection and exfiltration are properties of a *request to an assistant*, so the INT-* rules now apply only to **prose prompts**, using a deterministic `text_kind` classifier (`detection/textkind.py`).
  * Secrets and PII in code are still caught by the detectors and the hard rules.
  * As a side effect, Laya is skipped for most non-prompt traffic, saving about 0.5 s per request.
* **(d) Random tokens read as payloads.** Even on prose prompts, Laya gave injection ≈ 1.0 to any prompt containing a hash, base64 blob or session ID ("Customer reported 401. Their ref is eyJ…"). Such tokens are now replaced by `[token]` before Laya sees the text. The blobs themselves are still decoded and scanned by the perception layer.
* **(e) Exfiltration needs corroboration.** The exfiltration question also fired on benign prose ("our office will be closed on…" scored 0.9). INT-001/002 now additionally require a deterministic `bulk_request` cue: all / every / entire / export / dump / download, plus equivalents in es, de, fr, it and hi.

**Calibration ablation** (`python -m leakguard.eval.calibrate`; 600 stratified dev items; detectors + policy, with Laya intents applied as listed):

| Configuration | False-block rate | Protect acc. | Injection/jailbreak recall |
|---|---|---|---|
| No Laya intents (detectors + policy only) | 1.6% | 0.868 | 0% |
| Laya intents on all text, initial thresholds | 49.0% | 0.760 | – |
| + prose-only gate | 35.8% | 0.803 | – |
| + bulk-request corroboration | 25.9% | 0.813 | – |
| + token neutralisation, thresholds re-tuned (best possible) | **2.9%** | **0.877** | **75%** |
| same, ≤ 5% false-block budget | 4.9% | 0.882 | 83% |
| + residual-sensitive escalation (any cut-off 0.90–0.995) | ≥ 13.6% | ≤ 0.842 | 75% |

The chosen operating point is injection and exfiltration escalate at ≥ 0.95 and block at ≥ 0.98. The [0.95, 0.98) band goes to a human reviewer. The residual-sensitive check is implemented but **off by default**, because it never paid for itself on dev.

### 4.4 Error analysis that changed the system (dev)

1. **Gold-label gap in prowl.** Sampling *rows* left real secrets in "negative" texts unlabelled (for example, a config with a live-looking DB URL marked *allow*). The fix was to sample texts, then union the spans from all of their rows.
2. **GLiNER context bias.** GLiNER's "api key" label fired on anything assigned to an `API_KEY` variable (`"9f5"`, `region.staging.users.36`), and "person" fired on "Customer". The fix was plausibility filters for GLiNER-only hits: minimum length and entropy, no dotted-word tokens, no bare hex or UUIDs, and a stop-list of role words. On dev, FBR went from 6.9% to 2.9% and precision from 0.53 to 0.66, with leakage recall roughly unchanged (0.81 → 0.79).
3. **Mixed prompts.** A debugging prompt ("My script returns 401. Config:" followed by `OPENAI_API_KEY=…`) was blocked as an injection, because Laya read the config line as an instruction. Laya now judges only the natural-language lines of a mixed prompt (`prose_only` in `decision/laya_client.py`). It was validated on dev first (same 2.9% FBR and 75% injection recall), then re-run on test, where FBR fell from 5.3% to 4.9%.
4. **De-obfuscation over-reach.** Found by using the dashboard: the concatenation decoder merged two JSON strings separated by a comma, so masking swallowed the `"body"` key. The decoder now requires an explicit `+` or `.` operator, and there's a regression test.

### 4.5 Where it still fails (test; full list in [failure_analysis.md](failure_analysis.md))

* **Structured IDs in foreign formats.**
  * GOV_ID recall is 35%. ai4privacy ID numbers have country-specific formats that neither the rules nor zero-shot GLiNER know.
  * PASSPORT recall is 31%.
  * CREDIT_CARD recall is 21%, because ai4privacy card numbers fail Luhn, so the validator rejects them by design.
* **Languages.** Leakage recall is 88% for English and 74–79% for de, es, fr, hi and nl, but 60% for Telugu.
* **False blocks.** 10 of the agent's 12 false blocks are detector false positives on prowl's look-alike negatives, such as `secret: 90b7e0` or an AWS-shaped `AKIA…` string. These hit hard rules: a "secret" in a response or tool payload is blocked. The other 2 are Laya injection scores on garbled text.
* **Over-redaction** (16%) mostly comes from GLiNER marking person names in benign prompts. That is harmless to the user, since the names are re-hydrated in gateway mode, but it inflates the over-redaction count.

### 4.6 Latency

Per-node p50 / p95 on the 600-item agent run (CPU):

| Node | p50 / p95 |
|---|---|
| observe | 3 / 4 ms |
| perceive (rules + GLiNER) | **427 / 844 ms** |
| semantic_state | 0.1 / 0.2 ms |
| recall | 0.3 / 1.6 ms |
| reason (policy + Laya) | **734 / 1424 ms** |
| act | 0.0 / 2.0 ms |
| verify | 0.0 / 0.3 ms |

The rules-only path (CI and fallback) takes 0.2 ms. The two models account for more than 98% of latency; ONNX export and batching are the obvious next steps (§6).

## 5. Agentic requirements checklist

| Requirement | Where |
|---|---|
| Observe → perception → semantic state → memory/RAG → reason/plan → tool action → guardrails/HITL → outcome → feedback | `agent/graph.py`, [agent_state_action.md](agent_state_action.md) |
| Explicit state and goal | `agent/state.py` (`GuardState.goal`) |
| ≥ 2 safe tool/API actions | 7 tools, per-stage allowlist (`agent/tools.py`) |
| Memory / RAG | BM25 policy retrieval + reviewer-precedent memory (`agent/memory.py`) |
| Guardrails / HITL | 6 layers; Review Queue page; `/v1/hitl` |
| Outcome logging, audit | DB tables + SHA-256 hash chain (`audit/chain.py`) |
| Tool-call / action trace | `GuardState.trace` → `/v1/requests/{id}` → Audit Explorer |
| Benchmark and failure analysis | `eval/benchmark.py`, [results.md](results.md), [failure_analysis.md](failure_analysis.md) |
| API, persistent storage, dashboard, tests, containers, monitoring | FastAPI, Postgres/Alembic, Streamlit, 112 tests, Docker/Compose, Prometheus/Grafana |

## 6. Limitations and future work

* **Latency on CPU.** GLiNER (about 0.4 s) and Laya (about 0.6 s) dominate. Next steps: ONNX export (both libraries support it), batching in the gateway, and running Laya only on prompts and tool payloads above a rule-based risk score.
* **Gold-label coverage.** Prowl labels only secrets, so unlabelled names in its chat logs count against precision. No dataset yields `escalate` gold labels at scale.
* **Exfiltration over long agent trajectories** (e.g. gloom-data-exfiltration) needs a trajectory-level memory rather than per-message checks.
* **Laya fine-tuning.** Zero-shot Laya stops about a third of injections and jailbreaks at a 1.8% false-stop rate (§4.2). Laya ships an RLCD fine-tuning recipe, and fine-tuning it on the *train* splits of deepset and v1adam (kept separate from test) is the most promising way to raise intent recall. It is the same Colab/Kaggle workflow as the GLiNER notebook.
* **Licences.** prowl is CC-BY-NC and ai4privacy has custom terms, so the trained artefacts are for research and education.
