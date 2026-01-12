# chatbot/chatbot_runner.py
from pathlib import Path
from dotenv import load_dotenv

# Load chatbot/.env explicitly
ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

import yaml
from chatbot.retrieval.retriever import Retriever
from chatbot.context_builder import build_context
from chatbot.llm.factory import get_llm
from chatbot.safety_guard import check_and_refuse, refuse_response
from chatbot.deterministic_summary import generate_summary
from chatbot.utils.file_utils import read_text_file
from chatbot.config_loader import cfg

BASE_DIR = Path(__file__).resolve().parents[0]
CONFIG = cfg or {}

class ChatbotRunner:
    def __init__(self, index_dir: str = None):
        model = CONFIG.get("embedding_model", "all-MiniLM-L6-v2")
        idx_dir = index_dir or str((BASE_DIR / "vectorstore"))
        self.retriever = Retriever(index_dir=idx_dir, model_name=model)
        self.llm = get_llm()
        self._report_cache = {}

    def _get_report_text(self, manifest: dict):
        report_id = manifest.get("report_id")
        if not report_id:
            return None
        if report_id in self._report_cache:
            return self._report_cache[report_id].get("text")
        model3_path = manifest.get("artifacts", {}).get("model3_txt")
        text = read_text_file(model3_path)
        self._report_cache[report_id] = {"text": text, "manifest": manifest}
        return text

    def answer(self, user_question: str, manifest: dict, chat_history: list):
        # Safety first
        if check_and_refuse(user_question):
            return refuse_response(), {"refused": True, "sources": []}

        # Ensure report loaded (cached)
        self._get_report_text(manifest)

        # Optional retrieval (enabled via config)
        retrieved = []
        if getattr(self.retriever, "enabled", False) and CONFIG.get("retrieval_enabled", False):
            try:
                k = CONFIG.get("max_retrieved_chunks", 6)
                retrieved = self.retriever.similarity_search(user_question, k=k) or []
            except Exception:
                retrieved = []

        # Build prompt (report-bound)
        prompt = build_context(retrieved, manifest, chat_history, user_question)

        # LLM call
        try:
            resp = self.llm.generate(prompt, max_tokens=512)
            if not resp or resp.lower().startswith("[llm not configured]"):
                raise RuntimeError("LLM not configured or returned stub.")
            # guard: if LLM itself includes diagnosis/prescription, refuse
            if check_and_refuse(resp):
                return refuse_response(), {"refused": True, "sources": [r.get("chunk_id") for r in retrieved]}
            return resp, {"refused": False, "sources": [r.get("chunk_id") for r in retrieved]}
        except Exception:
            # deterministic fallback
            det = generate_summary(manifest)
            fallback_text = ("[LLM unavailable — deterministic fallback used]\n\n" + det)
            return fallback_text, {"refused": False, "sources": [r.get("chunk_id") for r in retrieved]}
