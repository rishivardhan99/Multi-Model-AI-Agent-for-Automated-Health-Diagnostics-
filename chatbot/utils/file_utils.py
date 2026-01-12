# chatbot/utils/file_utils.py
from pathlib import Path
import json
from typing import Optional

# This file lives at: /app/chatbot/utils/file_utils.py inside the container.
# Chatbot root is one parent up.
CHATBOT_ROOT = Path(__file__).resolve().parents[1]

DEBUG_OUT = Path("/tmp/chatbot_path_debug.json")  # inspectable inside container

def _log_debug(entry: dict):
    try:
        existing = []
        if DEBUG_OUT.exists():
            try:
                existing = json.loads(DEBUG_OUT.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append(entry)
        DEBUG_OUT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        # never crash on logging
        pass

def resolve_artifact_path(p: Optional[str]) -> Optional[Path]:
    """
    Resolve artifact paths with clear priority:
      1) If the path is absolute and exists -> return it
      2) If the path exists relative to CHATBOT_ROOT -> return it
      3) If the path exists relative to CHATBOT_ROOT/'outputs' -> return it
      4) If the path exists when treated as relative to cwd -> return it
      5) Otherwise return None (and log attempted candidates)
    """
    tried = []
    if not p:
        _log_debug({"requested": p, "result": None, "tried": tried})
        return None

    requested = Path(p)
    # 1) absolute
    if requested.is_absolute():
        tried.append(str(requested))
        if requested.exists():
            _log_debug({"requested": p, "result": str(requested), "tried": tried})
            return requested.resolve()

    # 2) relative to chatbot root
    candidate = CHATBOT_ROOT / requested
    tried.append(str(candidate))
    if candidate.exists():
        _log_debug({"requested": p, "result": str(candidate), "tried": tried})
        return candidate.resolve()

    # 3) relative to chatbot root /outputs (common pattern)
    candidate2 = CHATBOT_ROOT / "sample_data" / "outputs" / requested.name
    tried.append(str(candidate2))
    if candidate2.exists():
        _log_debug({"requested": p, "result": str(candidate2), "tried": tried})
        return candidate2.resolve()

    # 4) relative to current working directory
    cwd_candidate = Path.cwd() / requested
    tried.append(str(cwd_candidate))
    if cwd_candidate.exists():
        _log_debug({"requested": p, "result": str(cwd_candidate), "tried": tried})
        return cwd_candidate.resolve()

    # 5) last attempt: check chatbot root / outputs / provided relative path as a whole
    candidate3 = CHATBOT_ROOT / "outputs" / requested
    tried.append(str(candidate3))
    if candidate3.exists():
        _log_debug({"requested": p, "result": str(candidate3), "tried": tried})
        return candidate3.resolve()

    # nothing found
    _log_debug({"requested": p, "result": None, "tried": tried})
    return None

def read_text_file(p: Optional[str]) -> Optional[str]:
    path = resolve_artifact_path(p)
    if not path:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text if text else None
    except Exception:
        return None

def read_json_file(p: Optional[str]) -> Optional[dict]:
    path = resolve_artifact_path(p)
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
