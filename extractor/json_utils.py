# extractor/json_utils.py
import json
from typing import Any

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
# keeping old name for compatibility
def load_json_file(path: str):
    return load_json(path)

def flatten_json_text(data: Any) -> str:
    """
    Convert a JSON object to a single text blob suitable for CSV 'raw_text' cell.
    Expands nested dicts (e.g., 'parameters') into 'key: value' lines so extractor regex can match.
    """
    def _emit_dict(d: dict, prefix: str = ""):
        parts = []
        for k, v in d.items():
            if isinstance(v, (str, int, float)):
                parts.append(f"{prefix}{k}: {v}")
            elif isinstance(v, dict):
                # recursive expansion, join with prefix if needed
                parts.extend(_emit_dict(v, prefix=f"{prefix}{k}."))
            elif isinstance(v, list):
                # try to expand list elements if they are dict-like, else join items
                if all(isinstance(item, dict) for item in v):
                    for i, item in enumerate(v, start=1):
                        parts.append(f"{prefix}{k}[{i}]:")
                        parts.extend(_emit_dict(item, prefix=f"{prefix}{k}[{i}]."))
                else:
                    # join list of primitives
                    parts.append(f"{prefix}{k}: " + ", ".join(str(x) for x in v))
            else:
                parts.append(f"{prefix}{k}: {v}")
        return parts

    parts = []
    if isinstance(data, dict):
        # prefer expanding everything
        parts.extend(_emit_dict(data))
    else:
        parts.append(str(data))
    return "\n".join(parts)
