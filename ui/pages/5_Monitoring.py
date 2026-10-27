"""Live operational metrics from the API (also exported to Prometheus at /metrics)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from api_client import api_status, get

st.set_page_config(page_title="Monitoring · LeakGuard", page_icon="📈", layout="wide")
api_status()
st.title("📈 Monitoring")
s = get("/v1/stats")
m = st.columns(3) + st.columns(3)
m[0].metric("Requests", s["requests"])
m[1].metric("Block rate", f"{s['block_rate']:.1%}")
m[2].metric("Escalation rate", f"{s['escalation_rate']:.1%}")
m[3].metric("p95 latency", f"{s['latency_ms']['p95']:.0f} ms")
m[4].metric("Tool success", "–" if s["tool_success_rate"] is None else f"{s['tool_success_rate']:.1%}")
m[5].metric("Reviewer override rate", "–" if s["override_rate"] is None else f"{s['override_rate']:.1%}")

c1, c2 = st.columns(2)
if s["by_action"]:
    c1.plotly_chart(px.pie(names=list(s["by_action"]), values=list(s["by_action"].values()), title="Decisions",
                           color=list(s["by_action"]), color_discrete_map={"allow": "#2e7d32", "redact": "#1565c0",
                                                                           "escalate": "#ef6c00", "block": "#c62828"}),
                    use_container_width=True)
if s["by_direction"]:
    df = pd.DataFrame(s["by_direction"]).fillna(0).T
    c2.plotly_chart(px.bar(df, barmode="stack", title="Decisions by direction"), use_container_width=True)
st.markdown("**Latency (ms)**")
st.write(s["latency_ms"])
st.markdown("**Tool calls**")
st.dataframe(pd.DataFrame([{"tool:status": k, "count": v} for k, v in s["tools"].items()]), hide_index=True)
st.markdown("**Human review**")
st.write(s["hitl"])
st.caption("Prometheus scrapes /metrics; a Grafana dashboard is provisioned in deploy/grafana.")
