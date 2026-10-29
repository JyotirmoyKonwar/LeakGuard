#!/usr/bin/env sh
# Hugging Face Space entrypoint: API on 127.0.0.1:8000 (private), Streamlit on 0.0.0.0:7860 (public).
set -e
uvicorn leakguard.api.main:app --host 127.0.0.1 --port 8000 &
exec streamlit run ui/Playground.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
