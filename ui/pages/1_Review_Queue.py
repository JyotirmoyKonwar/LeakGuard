"""Human-in-the-loop review queue."""

from __future__ import annotations

import streamlit as st
from api_client import api_status, badge, get, post

st.set_page_config(page_title="Review Queue · LeakGuard", page_icon="🧑‍⚖️", layout="wide")
api_status()
st.title("🧑‍⚖️ Human review queue")
st.caption("Cases the agent was not allowed or not confident enough to decide alone. Every verdict is audited, "
           "counted as an override if it differs from the proposal, and becomes a precedent the agent reuses.")

reviewer = st.sidebar.text_input("Reviewer name", "secops-analyst")
status = st.sidebar.selectbox("Show", ["pending", "approved", "edited", "rejected", "all"])
cases = get("/v1/hitl", status=status)
st.metric("Cases shown", len(cases))
if not cases:
    st.success("Queue is empty.")

for c in cases:
    ctx = c.get("context", {})
    sem = ctx.get("semantic", {})
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown(f"**Case `{c['id'][:10]}`** · {c['created_at'][:19]} · {sem.get('direction')} → "
                    f"{sem.get('destination')} · role `{sem.get('role')}`" + (f" · tool `{sem.get('tool')}`" if sem.get("tool") else ""))
        h2.markdown(f"proposed {badge(c['proposed_action'])}", unsafe_allow_html=True)
        st.code(c["redacted_text"], language=None, wrap_lines=True)
        st.caption("Entities: " + ", ".join(f"{k}×{v}" for k, v in sem.get("entity_counts", {}).items()) +
                   f" · citations: {', '.join(ctx.get('citations', []))}")
        with st.expander("Why was this escalated?"):
            for line in c["reason"].split("; "):
                st.write("• " + line)
            if ctx.get("laya"):
                st.json({k: ctx["laya"][k] for k in ("injection", "exfiltration", "residual_sensitive", "confidence")})
        if c["status"] == "pending":
            note = st.text_input("Justification (required)", key=f"n{c['id']}")
            b1, b2, b3 = st.columns(3)
            for col, action, label in [(b1, "allow", "✅ Allow"), (b2, "redact", "✂️ Redact & send"), (b3, "block", "⛔ Block")]:
                if col.button(label, key=f"{action}{c['id']}", use_container_width=True, disabled=len(note) < 3):
                    r = post(f"/v1/hitl/{c['id']}/resolve", {"reviewer": reviewer, "action": action, "note": note})
                    if r.status_code == 200:
                        st.toast(f"Resolved: {action}" + (" (override)" if r.json().get("override") else ""))
                        st.rerun()
                    else:
                        st.error(r.json().get("detail", r.text))
        else:
            st.markdown(f"Resolved by **{c['reviewer']}** → {badge(c['reviewer_action'])} · _{c['reviewer_note']}_",
                        unsafe_allow_html=True)
