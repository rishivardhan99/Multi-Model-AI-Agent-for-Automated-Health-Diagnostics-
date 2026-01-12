# chatbot/ui/chat_ui.py
import streamlit as st
from pathlib import Path
import json
from chatbot.chatbot_runner import ChatbotRunner

BASE_DIR = Path(__file__).resolve().parents[2]

def render_chatbot_ui(manifest=None, index_dir=None):
    st.header("Ask Medicube — Report Q&A (Dev)")

    if manifest is None:
        st.info("No report selected. Please provide a manifest.")
        return

    # normalize manifest if path given
    if isinstance(manifest, str):
        try:
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except Exception:
            st.error("Failed to load manifest file.")
            return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_report_id" not in st.session_state:
        st.session_state.active_report_id = manifest.get("report_id")

    if st.session_state.active_report_id != manifest.get("report_id"):
        st.session_state.chat_history = []
        st.session_state.active_report_id = manifest.get("report_id")

    st.subheader(f"Report: {manifest.get('report_id')}")
    st.write("Canonical base:", manifest.get("canonical_base"))
    st.write("Artifacts:")
    for k, v in manifest.get("artifacts", {}).items():
        st.write(f"- {k}: {v}")

    # persist runner in session state
    if "chatbot_runner" not in st.session_state:
        st.session_state["chatbot_runner"] = ChatbotRunner(index_dir=index_dir)
    runner = st.session_state["chatbot_runner"]

    # input form
    with st.form("chat_form", clear_on_submit=False):
        question = st.text_input("Ask a question about this report (within scope):", key="chat_input")
        submitted = st.form_submit_button("Send")
        if submitted and question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Generating answer..."):
                answer, meta = runner.answer(question, manifest, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # display chat history
    if st.session_state.chat_history:
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(f"**You:** {turn['content']}")
            else:
                st.markdown(f"**Medicube:** {turn['content']}")
    else:
        st.info("No messages yet. Ask something about the selected report.")
