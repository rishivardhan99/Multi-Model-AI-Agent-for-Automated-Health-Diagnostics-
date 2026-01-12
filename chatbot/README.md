Chatbot module
=======================

Purpose
-------
This module provides a RAG-backed stateful chatbot for Medicube. It indexes
project docs + artifacts, then serves a Streamlit chat UI that answers
questions for a selected report using the manifest as authoritative context.

Quick start
-----------
1. Create virtualenv and install:
   pip install -r chatbot/requirements.txt

2. Create .env:
   cp chatbot/.env.example chatbot/.env
   Add keys if you want Gemini or Grok integration.

3. Index artifacts (offline):
   python chatbot/ingestion/index_builder.py \
       --source-docs "../medicube_project_memory.md" \
       --artifacts-dir "../outputs" \
       --index-dir "./vectorstore"

4. Run the Streamlit UI from your main app. The UI call provided is:
   from chatbot.ui.chat_ui import render_chatbot_ui
   render_chatbot_ui(manifest=st.session_state.last_manifest)

Notes
-----
- If LLM API keys are absent, the chatbot will still run but will return a
  deterministic fallback that includes retrieved context and a friendly
  message asking to configure the provider.
- The index is local (FAISS) stored at ./vectorstore/index.faiss and
  ./vectorstore/metadata.json
