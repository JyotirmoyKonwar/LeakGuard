---
title: LeakGuard
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Agentic data-leak prevention guard for LLM apps
---

# LeakGuard on Hugging Face Spaces

This Space runs the LeakGuard API (internal) and the Streamlit dashboard (public, port 7860) in one container,
built from `deploy/Dockerfile --target space`. The free CPU tier (2 vCPU / 16 GB) fits both models.

Set these **Space secrets**:
- `LG_DATABASE_URL`: a Neon Postgres URL (persistent storage; the Space disk is ephemeral)
- `LG_VAULT_KEY`: a Fernet key (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)
- `LG_LLM_PROVIDER` / `LG_LLM_API_KEY`: e.g. `groq` + your key (optional; defaults to the mock LLM)

See `docs/deployment.md` for the full steps.
