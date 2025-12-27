import os
import json
import shutil
from typing import Any

def _ensure(folder: str):
    os.makedirs(folder, exist_ok=True)

def save_json(name: str, data: Any, folder: str = "outputs/model2_outputs") -> str:
    _ensure(folder)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def save_text(name: str, text: str, folder: str = "outputs/model2_outputs") -> str:
    _ensure(folder)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def save_prompt(name: str, text: str, folder: str = "outputs/model2_outputs") -> str:
    return save_text(name, text, folder=folder)

def append_escalation(row_summary: dict, path: str = "escalate.csv") -> str:
    import csv
    file_exists = os.path.exists(path)
    with open(path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "row_index", "reason"])
        writer.writerow([row_summary.get("filename", ""), row_summary.get("row_index", ""), row_summary.get("reason", "")])
    return path

def save_raw_input(input_path: str, folder: str = "outputs/model2_outputs", name: str = None) -> str:
    """
    Copy the raw input file into folder for traceability.
    name: filename to use inside folder
    """
    _ensure(folder)
    base = os.path.basename(input_path)
    dest_name = name if name else f"raw_input_{base}"
    dest = os.path.join(folder, dest_name)
    try:
        shutil.copy2(input_path, dest)
        return dest
    except Exception:
        # best-effort: if it's a stream or inaccessible, write a small marker file
        fallback = os.path.join(folder, dest_name + ".missing")
        with open(fallback, "w", encoding="utf-8") as f:
            f.write(f"Could not copy original file: {input_path}\n")
        return fallback
