# chatbot/retrieval/retriever.py
from pathlib import Path
import json
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parents[1]

class Retriever:
    """
    Minimal retriever: lazily loads sentence-transformers/faiss if available.
    If not installed or index missing, .enabled == False and similarity_search returns [].
    """
    def __init__(self, index_dir: str = None, model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir) if index_dir else (BASE_DIR / "vectorstore")
        self.model_name = model_name
        self.enabled = False
        self.metadata = []
        try:
            # optional heavy deps
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            # Attempt to load index and metadata
            idx_path = self.index_dir / "index.faiss"
            meta_path = self.index_dir / "metadata.json"
            if idx_path.exists() and meta_path.exists():
                self.embedder = SentenceTransformer(self.model_name)
                self.index = faiss.read_index(str(idx_path))
                with open(meta_path, "r", encoding="utf-8") as fh:
                    self.metadata = json.load(fh)
                self.enabled = True
        except Exception:
            self.enabled = False

    def similarity_search(self, query: str, k: int = 6) -> List[Dict]:
        if not self.enabled:
            return []
        emb = self.embedder.encode([query], convert_to_numpy=True)
        import faiss
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results
