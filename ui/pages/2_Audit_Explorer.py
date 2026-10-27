"""Request history, per-request agent traces and the tamper-evident audit log."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from api_client import api_status, badge, get

st.set_page_config(page_title="Audit · LeakGuard", page_icon="📜", layout="wide")
api_status()
st.title("📜 Audit explorer")

v = get("/v1/audit/verify")
if v["valid"]:
    st.success(f"Audit hash chain intact · {v['events_checked']} events · head `{v.get('head', '')[:16]}…`")
else:
    st.error(f"Audit chain BROKEN at event #{v['broken_at_seq']} (after {v['events_checked']} valid events)")

f1, f2, f3 = st.columns(3)
action = f1.selectbox("Action", ["any", "allow", "redact", "escalate", "block"])
direction = f2.selectbox("Direction", ["any", "prompt", "response", "tool_payload"])
limit = f3.slider("Rows", 10, 500, 100)
params = {"limit": limit}
if action != "any":
    params["action"] = action
if direction != "any":
    params["direction"] = direction
reqs = get("/v1/requests", **params)
if not reqs:
    st.info("No requests yet. Try the Playground.")
    st.stop()
df = pd.DataFrame(reqs)
st.dataframe(df[["created_at", "id", "direction", "destination", "role", "final_action", "status", "latency_ms", "redacted_input"]],
             hide_index=True, use_container_width=True)

rid = st.selectbox("Inspect request", [r["id"] for r in reqs])
d = get(f"/v1/requests/{rid}")
st.markdown(f"### {badge(d['final_action'])} &nbsp; `{rid}`", unsafe_allow_html=True)
a, b = st.columns(2)
with a:
    st.markdown("**Redacted input (what is stored)**")
    st.code(d["redacted_input"], language=None, wrap_lines=True)
    if d.get("decision"):
        dec = d["decision"]
        st.markdown(f"**Policy** `{dec['policy_version']}` → `{dec['policy_action']}`{' (hard)' if dec['hard'] else ''} · "
                    f"final `{dec['final_action']}` · confidence {dec['confidence']:.2f}")
        st.dataframe(pd.DataFrame(dec["matched_rules"]), hide_index=True, use_container_width=True)
with b:
    st.markdown("**Findings**")
    st.dataframe(pd.DataFrame(d["findings"]), hide_index=True, use_container_width=True)
    st.markdown("**Tool calls**")
    st.dataframe(pd.DataFrame(d["tool_calls"]), hide_index=True, use_container_width=True)
st.markdown("**Agent trace (state machine path)**")
st.dataframe(pd.DataFrame([{"step": t.get("node") or "tool:" + str(t.get("tool")), "ms": t.get("ms", t.get("latency_ms")),
                            "status": t.get("status"), "summary": t.get("summary") or str(t.get("detail"))}
                           for t in d["trace"]]), hide_index=True, use_container_width=True)
st.markdown("**Audit events for this request**")
for e in get("/v1/audit", request_id=rid):
    with st.expander(f"#{e['seq']} {e['event_type']} · hash {e['hash'][:12]}… ← prev {e['prev_hash'][:12]}…"):
        st.json(e["payload"])
