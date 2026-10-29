"""Publish LeakGuard to a Hugging Face Docker Space.

    HF_TOKEN=... uv run python deploy/hf-space/push_space.py --space <user>/leakguard

The Space builds deploy/Dockerfile with no --target, so the *last* stage (`space`: API + dashboard on :7860)
is what runs. Secrets (LG_DATABASE_URL, LG_VAULT_KEY, ...) are set in the Space settings, never uploaded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ["pyproject.toml", "uv.lock", "src", "ui", "eval/results", "deploy/hf-space/start.sh", "alembic.ini", "migrations"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. JyotirmoyKonwar/leakguard")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for item in INCLUDE:
            src = ROOT / item
            dst = t / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "records_*.jsonl"))
            elif src.exists():
                shutil.copy2(src, dst)
        shutil.copy2(ROOT / "deploy" / "Dockerfile", t / "Dockerfile")
        shutil.copy2(ROOT / "deploy" / "hf-space" / "README.md", t / "README.md")
        api.upload_folder(folder_path=str(t), repo_id=a.space, repo_type="space",
                          commit_message="Deploy LeakGuard")
    print(f"pushed -> https://huggingface.co/spaces/{a.space}  (set secrets in Settings)")


if __name__ == "__main__":
    main()
