# extractor/postprocess.py
from typing import Dict, Any

def postprocess_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Postprocessing MUST be non-destructive.
    It may only attach notes, never modify numeric values.
    """
    if not isinstance(row, dict):
        return row

    # Hemoglobin annotation (DO NOT modify value)
    try:
        hb = row.get("Hemoglobin")
        if hb not in (None, "", "None"):
            try:
                hv = float(hb)
                if hv > 30:  # likely g/L misread
                    row.setdefault("_notes", {})
                    row["_notes"]["Hemoglobin"] = (
                        "Value unusually high; possible g/L → g/dL OCR scale issue"
                    )
            except Exception:
                pass
    except Exception:
        pass

    # Total Protein annotation (DO NOT modify value)
    try:
        tp = row.get("Total Protein")
        if tp not in (None, "", "None"):
            try:
                tv = float(tp)
                if tv > 100:
                    row.setdefault("_notes", {})
                    row["_notes"]["Total Protein"] = (
                        "Value unusually high; possible missing decimal OCR issue"
                    )
            except Exception:
                pass
    except Exception:
        pass

    return row
