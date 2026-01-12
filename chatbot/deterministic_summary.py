# chatbot/deterministic_summary.py
from chatbot.utils.file_utils import read_text_file, read_json_file, resolve_artifact_path
import textwrap
from pathlib import Path
import json

def _format_summary(body: str) -> str:
    return (
        "Clinician-style summary (deterministic):\n\n"
        f"{body}\n\n"
        "When to consult a clinician:\n"
        "- If you have symptoms such as dizziness, chest pain, or severe shortness of breath.\n"
        "- If abnormalities are new, worsening, or persistent.\n\n"
        "Disclaimer: This summary is informational and not a medical diagnosis."
    )

def _read_debug_log() -> str:
    dbg = Path("/tmp/chatbot_path_debug.json")
    if not dbg.exists():
        return "No debug path log found."
    try:
        j = json.loads(dbg.read_text(encoding="utf-8"))
        # show last 6 attempts
        last = j[-6:]
        pretty = json.dumps(last, indent=2)
        return f"Path resolution debug (last attempts):\n{pretty}"
    except Exception:
        return "Failed to read debug log."

def generate_summary(manifest: dict) -> str:
    model3_path = manifest.get("artifacts", {}).get("model3_txt")
    model3_text = read_text_file(model3_path)
    if model3_text:
        body = textwrap.shorten(model3_text, width=2000, placeholder="...")
        footer = f"\n\n[Resolved model3_txt path: {resolve_artifact_path(model3_path)}]"
        return _format_summary(body + footer)

    # try model2
    model2_path = manifest.get("artifacts", {}).get("model2_json")
    model2 = read_json_file(model2_path)
    if model2:
        parts = []
        patterns = model2.get("patterns", {})
        positives = [k for k, v in patterns.items() if v]
        negatives = [k for k, v in patterns.items() if not v]
        if positives:
            parts.append("Detected patterns: " + ", ".join(positives) + ".")
        if negatives:
            parts.append("No evidence of: " + ", ".join(negatives) + ".")
        derived = model2.get("derived_metrics", {})
        if derived:
            parts.append(
                "Derived metrics: "
                + ", ".join(f"{k} = {v}" for k, v in derived.items())
                + "."
            )
        notes = model2.get("notes")
        if notes:
            parts.append("Notes: " + notes)
        body = " ".join(parts) if parts else "No major abnormalities detected."
        footer = f"\n\n[Resolved model2_json path: {resolve_artifact_path(model2_path)}]"
        return _format_summary(body + footer)

    # nothing found — include debug log so we can see exactly what was tried.
    debug_info = _read_debug_log()
    return _format_summary("No detailed artifacts were available for deterministic analysis.\n\n" + debug_info)
