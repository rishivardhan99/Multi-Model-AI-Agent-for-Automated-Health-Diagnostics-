"""
Lightweight index builder. Use for offline indexing.
"""

import argparse
from pathlib import Path
import json
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
from chunker import chunk_doc
from loaders import load_manifest, load_artifacts_for_manifest

def build_index_from_files(docs, artifacts, index_dir: str, model_name="all-MiniLM-L6-v2", max_chars=2000):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except Exception as e:
        raise RuntimeError("Missing sentence-transformers/faiss. Install requirements to build index.") from e

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    all_texts = []
    # docs: list of (path, type)
    for path, doc_type in docs:
        chunks = chunk_doc(path, doc_type, tags=[doc_type], max_chars=max_chars)
        for c in chunks:
            metadata.append({"chunk_id": c["chunk_id"], "source": c["source"], "type": c["type"], "tags": c["tags"], "content": c["content"]})
            all_texts.append(c["content"])

    for path, art_name in artifacts:
        chunks = chunk_doc(path, f"artifact_{art_name}", tags=["artifact", art_name], max_chars=max_chars)
        for c in chunks:
            metadata.append({"chunk_id": c["chunk_id"], "source": c["source"], "type": c["type"], "tags": c["tags"], "content": c["content"]})
            all_texts.append(c["content"])

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    with open(index_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(f"Index built with {len(metadata)} chunks at {index_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-docs", nargs="+", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    docs = []
    for d in args.source_docs:
        docs.append((d, "project_doc"))

    artifacts = []
    manifests = list(Path(args.artifacts_dir).rglob("manifest*.json"))
    for m in manifests:
        try:
            man = load_manifest(str(m))
            arts = load_artifacts_for_manifest(man, args.artifacts_dir)
            for a in arts:
                artifacts.append(a)
        except Exception as e:
            print("Skipping manifest", m, e)

    build_index_from_files(docs, artifacts, args.index_dir, model_name=args.model)

if __name__ == "__main__":
    main()
