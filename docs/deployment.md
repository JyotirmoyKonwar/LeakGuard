# Deployment

There are three ways to run LeakGuard, from quickest to most complete.

## 1. Local, no Docker (about 1 minute, mocks)

```bash
uv sync --all-extras
make demo        # API on :8000 with mock models (instant start)
make ui          # dashboard on :8501 (in a second terminal)
```

To use the real models, copy `.env.example` to `.env`, then run `make api`. The first start downloads GLiNER (about 1.1 GB) and Laya (about 1.3 GB) into the Hugging Face cache.

## 2. Full local stack with Docker Compose

Runs Postgres, the API, the UI, MLflow, Prometheus and Grafana.

```bash
cp .env.example .env            # set LG_VAULT_KEY at minimum
make up
```

| Service | URL |
|---|---|
| API + OpenAPI docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Grafana (dashboard "LeakGuard") | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| MLflow | http://localhost:5000 |

Apply migrations against Postgres with `LG_DATABASE_URL=postgresql+psycopg://leakguard:leakguard@localhost:5432/leakguard make migrate`. The API also creates tables on start.

## 3. Cloud: Hugging Face Spaces + Neon Postgres (free)

**Why Spaces instead of Render:** Render's free instance has 512 MB of RAM, but GLiNER plus Laya need about 3 GB. A free CPU Space has 16 GB.

1. **Database:** create a free project on [Neon](https://neon.tech) and copy the connection string (`postgresql://…?sslmode=require`).
2. **Space:** run
   ```bash
   HF_TOKEN=<write token> uv run python deploy/hf-space/push_space.py --space <your-user>/leakguard
   ```
   This creates a Docker Space and uploads the code with `deploy/Dockerfile` as its root `Dockerfile`. Spaces builds the last stage (`space`), which runs the API privately on `127.0.0.1:8000` and the dashboard publicly on port 7860. The first build takes about 10–15 minutes because it bakes in both models.
3. **Secrets** (Space → Settings → Variables and secrets):
   * `LG_DATABASE_URL`: the Neon URL
   * `LG_VAULT_KEY`: a Fernet key
   * `LG_LLM_PROVIDER=groq` and `LG_LLM_API_KEY=<groq key>` (optional)
   * `LG_API_KEY` (optional; protects the API)
4. **Migrations:** run `LG_DATABASE_URL=<neon url> uv run alembic upgrade head` once from your machine.
5. Open the Space URL and go to the Playground. `/v1` is internal to the container; the public demo is the dashboard.

## CI

`.github/workflows/ci.yml` runs three jobs on every push:

* lint (ruff), types (mypy), tests with coverage, and Alembic on a fresh SQLite DB
* the API and guardrail tests against a real **Postgres 16** service container
* a Docker build of the API image

CI uses mocks only (no model downloads), so it finishes in a few minutes.

## Configuration reference

All settings are environment variables prefixed `LG_`; see `src/leakguard/config.py` and `.env.example`. The most important ones:

| Variable | Default | Meaning |
|---|---|---|
| `LG_DATABASE_URL` | `sqlite:///leakguard.db` | SQLite for development, Postgres for production |
| `LG_LLM_PROVIDER` | `mock` | `groq`, `nim`, `ollama` or `openai_compat` (any OpenAI-compatible endpoint) |
| `LG_USE_GLINER` / `LG_GLINER_MODEL` | `true` / `urchade/gliner_multi_pii-v1` | Model 1 |
| `LG_USE_LAYA` / `LG_LAYA_MODEL` | `true` / `convaiinnovations/laya-multilingual` | Model 2 |
| `LG_LAYA_RESIDUAL_THRESHOLD` | `1.01` (off) | Escalate when detectors find nothing but Laya's residual-sensitive probability is at least this. Off by default: it raised false blocks on dev at every cut-off |
| `LG_SHADOW_MODE` | `false` | Log decisions but release everything (safe rollout) |
| `LG_VAULT_KEY` | random per process | Fernet key for reversible pseudonymisation |
| `LG_API_KEY` | unset | Require an `X-API-Key` header |
| `LG_ALERT_WEBHOOK_URL` | unset | Slack/Teams webhook for block and escalate alerts |
