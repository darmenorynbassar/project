"""Advisor prototype: chat with local Qwen, risk list, student review.

A human approves every action, and approving only writes a local simulated log.

Usage:
    streamlit run src/app.py --server.address localhost
"""
import ollama
import pandas as pd
import streamlit as st

import chatbot
import support
from features import CUTOFF_WEEKS

st.set_page_config(page_title="Academic Support Prototype", layout="wide")
st.session_state.setdefault("history", [])
st.session_state.setdefault("traces", {})

with st.sidebar:
    st.title("Academic support")
    week = st.selectbox("Cut-off week", CUTOFF_WEEKS, index=1)
    info = support.model_info(week)
    st.metric("Recall on unseen 2014 students", f"{info['recall']:.2f}")
    st.metric("Precision", f"{info['precision']:.2f}")
    st.metric("Students flagged", f"{info['share_of_students_flagged']:.0%}")
    st.caption(info["note"])
    st.caption(f"Chat model: {chatbot.MODEL}, running locally. Pseudonymised OULAD ids only.")
    if st.button("Clear chat"):
        st.session_state.history, st.session_state.traces = [], {}
        st.rerun()

chat_tab, list_tab, student_tab = st.tabs(["Chat", "Risk list", "Student review"])

with chat_tab:
    box = st.container(height=520)
    for i, message in enumerate(st.session_state.history):
        with box.chat_message(message["role"]):
            st.markdown(message["content"])
            if i in st.session_state.traces:
                with st.expander("Source data (from the model, not from the chatbot)"):
                    st.json(st.session_state.traces[i], expanded=False)

    if prompt := st.chat_input("Ask about students, risk, letters or meeting slots"):
        st.session_state.history.append({"role": "user", "content": prompt})
        try:
            with st.spinner("Thinking..."):
                reply, trace = chatbot.answer(st.session_state.history, week)
        except (ConnectionError, ollama.ResponseError) as e:
            st.session_state.history.pop()
            st.error(f"Cannot reach Ollama. Start it and pull {chatbot.MODEL}. ({e}) "
                     "The Risk list and Student review tabs still work.")
        else:
            st.session_state.history.append({"role": "assistant", "content": reply})
            if trace:
                st.session_state.traces[len(st.session_state.history) - 1] = trace
            st.rerun()

with list_tab:
    options = sorted(m["code_module"] for m in support.modules(week))
    module = st.selectbox("Module", ["All"] + list(dict.fromkeys(options)))
    risks = support.risk_list(week, None if module == "All" else module, limit=200)
    st.write(f"{risks['total_flagged']} students flagged at week {week}, showing the top "
             f"{len(risks['students'])}.")
    st.dataframe(pd.DataFrame(risks["students"]), hide_index=True, width="stretch")

with student_tab:
    student_id = st.number_input("Student id", min_value=0, step=1, value=None)
    if student_id is not None:
        result = support.explain(student_id, week)
        if not result["found"]:
            st.warning(result["note"])
        else:
            modules = [r["module"] for r in result["records"]]
            choice = st.selectbox("Module", modules) if len(modules) > 1 else modules[0]
            record = next(r for r in result["records"] if r["module"] == choice)
            st.subheader(f"{record['risk_level']} risk ({choice} {record['presentation']}, "
                         f"week {week})")
            st.table(pd.DataFrame(record["top_factors"]))

            draft = support.draft_letter(student_id, week, choice)
            letter = st.text_area("Letter to advisor (editable)", draft["letter"], height=300,
                                  key=f"letter_{student_id}_{choice}_{week}")
            slots = support.meeting_slots()
            slot = st.selectbox("Meeting slot (mock calendar)", slots) if slots else None
            outcome = st.selectbox("Outcome (simulated)",
                                   ["Approved", "Not needed", "Needs more review"])
            if st.button("Save decision (simulated, nothing is sent)"):
                support.record_action(student_id, choice, week, letter,
                                      slot if outcome == "Approved" else None, outcome)
                st.success("Saved to the local simulated log. Nothing was sent.")
