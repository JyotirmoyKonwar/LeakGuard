"""Policy-as-code: view rules and publish a new version."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from api_client import api_status, get, post

st.set_page_config(page_title="Policies · LeakGuard", page_icon="📐", layout="wide")
api_status()
st.title("📐 Policies")
p = get("/v1/policies/active")
st.markdown(f"Active policy **{p['name']} v{p['version']}**: {p['description']}")
st.json(p["settings"], expanded=False)
rows = [{"id": r["id"], "action": r["action"], "hard": r["hard"], "description": r["description"],
         "match": {k: v for k, v in r["match"].items() if v not in (None, False)}} for r in p["rules"]]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.markdown("### Edit & publish")
st.caption("A new version is validated, stored, recorded in the audit chain and applied to new requests immediately.")
y = st.text_area("Policy YAML", get("/v1/policies/yaml")["yaml"], height=420)
author = st.text_input("Author", "policy-admin")
if st.button("Validate & activate", type="primary"):
    r = post("/v1/policies", {"yaml": y, "author": author})
    (st.success if r.status_code == 200 else st.error)(r.json())
