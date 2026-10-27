"""Offline benchmark results (produced by `python -m leakguard.eval.benchmark`)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Benchmarks · LeakGuard", page_icon="📊", layout="wide")
st.title("📊 Benchmarks")
root = Path(__file__).resolve().parents[2] / "eval" / "results"
files = sorted(root.rglob("summary_*.json"))
if not files:
    st.info("No results yet. Run `make bench`.")
    st.stop()
f = st.selectbox("Result file", files, format_func=lambda p: str(p.relative_to(root)))
res = json.loads(Path(f).read_text())
sys = res["systems"]
keys = ["leakage_recall", "false_block_rate", "over_redaction_rate", "detection_precision", "detection_recall",
        "action_accuracy", "protect_accuracy", "escalation_rate", "groundedness", "tool_success_rate",
        "latency_ms_p50", "latency_ms_p95"]
table = pd.DataFrame({s: {k: v.get(k) for k in keys} for s, v in sys.items()}).T
st.dataframe(table, use_container_width=True)

c1, c2 = st.columns(2)
plot = table.reset_index(names="system")
c1.plotly_chart(px.bar(plot, x="system", y=["leakage_recall", "false_block_rate"], barmode="group",
                       title="Leakage recall vs false-block rate"), use_container_width=True)
c2.plotly_chart(px.bar(plot, x="system", y="latency_ms_p95", title="p95 latency (ms, CPU)", log_y=True),
                use_container_width=True)

s = st.selectbox("Breakdown for system", list(sys))
t1, t2, t3 = st.tabs(["By scenario", "By language", "By entity"])
t1.dataframe(pd.DataFrame(sys[s]["by_scenario"]).T, use_container_width=True)
t2.dataframe(pd.DataFrame(sys[s]["by_lang"]).T, use_container_width=True)
t3.dataframe(pd.DataFrame(sys[s]["entity_recall"]).T, use_container_width=True)
conf = pd.DataFrame(sys[s]["confusion"]).T
st.plotly_chart(px.imshow(conf, text_auto=True, labels={"x": "predicted", "y": "gold"}, title="Action confusion matrix",
                          color_continuous_scale="Blues"), use_container_width=True)
