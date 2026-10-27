"""Central configuration, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LG_", extra="ignore")

    # --- storage ---
    database_url: str = f"sqlite:///{ROOT / 'leakguard.db'}"

    # --- downstream (protected) chat LLM: any OpenAI-compatible endpoint ---
    llm_provider: Literal["mock", "groq", "nim", "ollama", "openai_compat"] = "mock"
    llm_base_url: str | None = None  # overrides the provider default
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float = 30.0

    # --- detection (model 1: GLiNER) ---
    detector_mode: Literal["rules", "rules+gliner"] = "rules+gliner"
    gliner_model: str = "urchade/gliner_multi_pii-v1"  # any GLiNER checkpoint (Hub id or local path)
    gliner_threshold: float = 0.45
    use_gliner: bool = True  # set False to run rules-only (CI, tiny containers)

    # --- decision (model 2: Laya) ---
    use_laya: bool = True
    laya_model: str = "convaiinnovations/laya-multilingual"
    # escalate when detectors found nothing but Laya is this sure the text is sensitive; >1 disables.
    # Off by default: on dev every cut-off in [0.90, 0.995] raised the false-block rate above 13%.
    laya_residual_threshold: float = 1.01

    # --- guardrails ---
    max_input_chars: int = 50_000
    shadow_mode: bool = False  # log what we *would* do, but always allow
    vault_key: str | None = None  # Fernet key; generated per-process if missing
    vault_ttl_s: int = 3600
    llm_rate_limit_per_min: int = 60
    internal_domains: list[str] = ["leakguard.local", "corp.example.com", "internal"]
    alert_webhook_url: str | None = None  # Slack/Teams-compatible SOC alert hook

    # --- misc ---
    policy_path: Path = ROOT / "src" / "leakguard" / "policy" / "policies" / "default.yaml"
    api_key: str | None = None  # if set, the API requires X-API-Key
    log_level: str = "INFO"

    def provider_defaults(self) -> tuple[str, str]:
        """(base_url, model) for the configured provider."""
        defaults = {
            "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
            "nim": ("https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct"),
            "ollama": ("http://localhost:11434/v1", "llama3.2"),
            "openai_compat": ("http://localhost:8001/v1", "default"),
            "mock": ("mock://", "mock-echo"),
        }
        base, model = defaults[self.llm_provider]
        return self.llm_base_url or base, self.llm_model or model


@lru_cache
def get_settings() -> Settings:
    return Settings()
