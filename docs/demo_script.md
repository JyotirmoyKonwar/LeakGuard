# Demo video script (about 6 minutes)

**Setup before recording:** `make api` (real models) and `make ui`, a fresh `leakguard.db`, and a Groq key in `.env` for the gateway segment. Open browser tabs for the Playground, Review Queue, Audit Explorer and Monitoring, plus a terminal.

| Time | Show | Say |
|---|---|---|
| 0:00 | README architecture diagram | "Companies want LLMs, but prompts, answers and agent tool calls leak customer data and credentials. LeakGuard is a guard agent that sits in front of any LLM. It detects sensitive data, decides what to do, acts, and records evidence." |
| 0:30 | Playground → *PII in a prompt* → Run | "Two small models. GLiNER finds the name, email, phone and Aadhaar number; rules and checksums confirm them. The policy says redact, and the prompt goes out as placeholders." Point at the citations PII-001/PII-002 and the agent trace. |
| 1:15 | *API key while debugging* | "A secret is never sent, not even reversibly. SEC-002 is a hard rule, so no model can override it." |
| 1:40 | *Hidden (base64) credential* | "It decodes base64, hex, split strings and invisible characters before detecting, so it catches a DB password hidden in base64 and blocks it." |
| 2:00 | *Prompt injection* and *Exfiltration* | "No PII here, so rules alone would allow it. Laya, the decision model, gives calibrated probabilities for injection and exfiltration, and the policy blocks at ≥ 0.98 and sends the 0.95–0.98 gray band to a human. Those thresholds were calibrated on dev." Show the Laya JSON. |
| 2:40 | Terminal: `curl` or the Python OpenAI SDK with `base_url=http://localhost:8000/v1` | "It's OpenAI-compatible: change one line in an existing app. The LLM sees `<EMAIL_1>`, and the user gets the real email back, because the response is re-hydrated from an encrypted vault, but only with this request's placeholders." Then send `__leak_secret__` using the mock provider to show the response guard. |
| 3:30 | Playground → tool_payload, `send_email` to an external address with a phone number | "Agents call tools too. Contact details going outside need a human, so it escalates." |
| 3:50 | Review Queue | "The reviewer sees only the masked text, Laya's view and the cited rules. They choose redact and must give a reason." Resolve it. Re-run the same request in the Playground: "AGENT-PRECEDENT: the agent learned from the reviewer." |
| 4:30 | Audit Explorer | "Every step is in a SHA-256 hash chain." Show the verify banner. Optional: edit a row in sqlite and show it turn red. |
| 5:00 | Benchmarks page | "Held-out test data, all from Hugging Face: ai4privacy in 8 languages, the prowl secrets corpus with look-alike negatives, and deepset injections. Leakage recall goes from 25% with rules to 79% with GLiNER and 81% with the full agent. The agent is the only system that stops injections and jailbreaks, at under 2% false stops on benign prompts. The trade-off is about 1 s of CPU latency, and the report says exactly where it goes." |
| 5:40 | Monitoring + Grafana | "Prometheus metrics for decisions, tools, escalations, overrides and fail-closed events. CI runs tests against real Postgres; Docker Compose and a Hugging Face Space deploy it." |

**Closing line:** "Two small CPU models, policy-as-code, human in the loop, full audit. Production-shaped AI security, with numbers to back it up."
