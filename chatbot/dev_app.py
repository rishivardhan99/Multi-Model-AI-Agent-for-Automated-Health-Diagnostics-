# chatbot/dev_app.py
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# --- LOAD .env FIRST (CRITICAL FOR STREAMLIT) ---
ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

print("DEBUG ENV CHECK:")
print("MEDICUBE_LLM_PROVIDER =", os.getenv("MEDICUBE_LLM_PROVIDER"))
print("GEMINI_API_KEY SET =", bool(os.getenv("GEMINI_API_KEY")))
# Ensure project root is importable so `import chatbot.*` works both locally and in Docker
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import json
from chatbot.chatbot_runner import ChatbotRunner
from chatbot.ui.chat_ui import render_chatbot_ui

st.set_page_config(page_title="Medicube Chatbot (Dev Test)")

sample_manifest = Path(__file__).resolve().parent / "sample_data" / "test_manifest.json"

if not sample_manifest.exists():
    st.error(f"Missing sample manifest: {sample_manifest}")
    st.stop()

try:
    manifest = json.loads(sample_manifest.read_text(encoding="utf-8"))
except Exception as e:
    st.error(f"Failed to load sample manifest: {e}")
    st.stop()

# simulate pipeline completion
if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = True
if "last_manifest" not in st.session_state:
    st.session_state.last_manifest = manifest

# persist runner in session state to preserve caches across Streamlit reruns
if "chatbot_runner" not in st.session_state:
    st.session_state["chatbot_runner"] = ChatbotRunner(index_dir=None)

render_chatbot_ui(manifest=manifest)
