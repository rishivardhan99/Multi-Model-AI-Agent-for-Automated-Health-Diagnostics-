import re
import uuid
from pathlib import Path
import json

def _normalize_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def chunk_text(text: str, max_chars: int = 2000):
    text = _normalize_whitespace(text)
    if len(text) <= max_chars:
        yield text
        return
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cur = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) + 1 <= max_chars:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            yield " ".join(cur)
            cur = [s]
            cur_len = len(s) + 1
    if cur:
        yield " ".join(cur)

def chunk_doc(source_path: str, doc_type: str, tags=None, max_chars=2000):
    tags = tags or []
    p = Path(source_path)
    if not p.exists():
        return []
    content = p.read_text(encoding="utf-8")
    chunks = []
    for i, c in enumerate(chunk_text(content, max_chars=max_chars)):
        chunk_id = f"{p.stem}__{i}__{uuid.uuid4().hex[:8]}"
        chunks.append({
            "chunk_id": chunk_id,
            "type": doc_type,
            "source": str(p.name),
            "content": c,
            "tags": tags,
            "priority": "high" if "decision" in tags or "rule" in tags else "medium"
        })
    return chunks

def write_chunks(chunks, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
