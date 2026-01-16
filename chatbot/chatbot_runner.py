# chatbot/chatbot_runner.py
from pathlib import Path
from dotenv import load_dotenv

# Load chatbot/.env explicitly
ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

from chatbot.retrieval.retriever import Retriever
from chatbot.context_builder import build_context
from chatbot.llm.factory import get_llm
from chatbot.safety_guard import (
    check_and_refuse,
    check_llm_output_and_refuse,
    refuse_response,
)
from chatbot.deterministic_summary import generate_summary
from chatbot.utils.file_utils import read_text_file
from chatbot.config_loader import cfg

BASE_DIR = Path(__file__).resolve().parent
CONFIG = cfg or {}


class ChatbotRunner:
    def __init__(self, index_dir: str = None):
        model = CONFIG.get("embedding_model", "all-MiniLM-L6-v2")
        idx_dir = index_dir or str(BASE_DIR / "vectorstore")

        self.retriever = Retriever(index_dir=idx_dir, model_name=model)
        self.llm = get_llm()
        self._report_cache = {}

    def _get_report_text(self, manifest: dict):
        report_id = manifest.get("report_id")
        if not report_id:
            return None

        if report_id in self._report_cache:
            return self._report_cache[report_id]["text"]

        model3_path = manifest.get("artifacts", {}).get("model3_txt")
        text = read_text_file(model3_path)

        self._report_cache[report_id] = {
            "text": text,
            "manifest": manifest,
        }
        return text

    def answer(self, user_question: str, manifest: dict, chat_history: list):
        # -------------------------
        # 1. USER INTENT CHECK
        # -------------------------
        intent = check_and_refuse(user_question)
        if intent == "HARD_REFUSE":
            return refuse_response(), {"refused": True, "sources": []}

        # -------------------------
        # 2. LOAD REPORT (Model-3 TXT)
        # -------------------------
        self._get_report_text(manifest)

        # -------------------------
        # 3. OPTIONAL RETRIEVAL (off by default)
        # -------------------------
        retrieved = []
        if (
            getattr(self.retriever, "enabled", False)
            and CONFIG.get("retrieval_enabled", False)
        ):
            try:
                k = CONFIG.get("max_retrieved_chunks", 6)
                retrieved = self.retriever.similarity_search(user_question, k=k) or []
            except Exception:
                retrieved = []

        # -------------------------
        # 4. BUILD PROMPT
        # -------------------------
        prompt = build_context(retrieved, manifest, chat_history, user_question)

        # -------------------------
        # 5. CALL LLM
        # -------------------------
        try:
            response = self.llm.generate(prompt, max_tokens=512)

            if not response or response.lower().startswith("[llm not configured]"):
                # Soft fallback — NOT an error
                response = generate_summary(manifest)


            # -------------------------
            # 6. OUTPUT SAFETY CHECK
            # -------------------------
            if check_llm_output_and_refuse(response):
                return refuse_response(), {
                    "refused": True,
                    "sources": [r.get("chunk_id") for r in retrieved],
                }

            return response, {
                "refused": False,
                "sources": [r.get("chunk_id") for r in retrieved],
            }

        except Exception:
            # -------------------------
            # 7. DETERMINISTIC FALLBACK
            # -------------------------
            det = generate_summary(manifest)
            fallback = (
                "[LLM unavailable — deterministic fallback used]\n\n" + det
            )
            return fallback, {
                "refused": False,
                "sources": [r.get("chunk_id") for r in retrieved],
            }
