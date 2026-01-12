from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]

def load_manifest(manifest_path: str):
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(p.read_text(encoding="utf-8"))

def load_artifacts_for_manifest(manifest: dict, artifacts_dir: str):
    artifacts = []
    for name, relpath in manifest.get("artifacts", {}).items():
        p = Path(relpath)
        # if path is relative, try relative to artifacts_dir
        if not p.exists():
            candidate = Path(artifacts_dir) / relpath
            if candidate.exists():
                p = candidate
        if p.exists():
            artifacts.append((str(p), name))
    return artifacts

def load_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")
