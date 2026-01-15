# model2/pipeline/loader.py
import csv
import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

# local quality checks
from .quality_checks import detect_low_data_quality

# (only the ALIAS_MAP shown; replace this section in your loader)
ALIAS_MAP = {
    "hdl_cholesterol": "HDL",
    "total_cholesterol": "Total_Cholesterol",
    "triglycerides": "Triglycerides",
    "glucose_fasting": "Glucose_Fasting",
    "hba1c": "HbA1c",
    "creatinine": "Creatinine",
    "crp": "CRP",
    "hemoglobin": "Hemoglobin",
    "platelets": "Platelets",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "neutrophils": "Neutrophils",
    "lymphocytes": "Lymphocytes",
    "ldl": "LDL",
    "ldl_cholesterol": "LDL",       # added mapping to canonical LDL
    "vldl": "VLDL",
    "urea_bun": "Urea_BUN",
    "urea": "Urea_BUN",
}

META_KEYS = {"age", "gender", "patient_id", "filename", "report_date"}

def _normalize_header(h: str) -> str:
    if h is None:
        return ""
    s = h.strip()
    s = re.sub(r"\(([^)]+)\)", r" \1", s)            # parentheses -> text
    s = s.replace("%", " percent")
    s = s.replace(" percent", "_PERCENT")

    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s

def _canonical_key(h: str) -> str:
    n = _normalize_header(h)
    key = n.lower()
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    parts = n.split("_")
    parts = [p.upper() if p.lower() in ("hdl","ldl","vldl","crp","hba1c","rbc","wbc") else p.capitalize() for p in parts]
    return "_".join(parts)

def _cast_number(s: Any):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # remove common units and letters
    s = re.sub(r"[A-Za-z/°%]+", "", s).strip()
    s = s.replace(",", "")
    if s == "":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s

def _load_param_map_expected(path_candidates: Optional[List[str]] = None) -> List[str]:
    """
    Try to locate param_map.json and return canonical expected param keys (param_id or canonical names).
    Fallback: empty list.
    """
    if path_candidates is None:
        path_candidates = [
            Path(__file__).resolve().parents[2] / "extractor" / "param_map.json",
            Path(__file__).resolve().parents[1] / "extractor" / "param_map.json",
            Path(__file__).resolve().parents[3] / "param_map.json",
        ]
    for p in path_candidates:
        try:
            if p.exists():
                j = json.loads(p.read_text(encoding="utf-8"))
                # prefer param_id if present else key name
                keys = []
                for k, v in j.items():
                    if isinstance(v, dict) and v.get("param_id"):
                        keys.append(v.get("param_id"))
                    else:
                        # normalize key to canonical pattern used in loader
                        keys.append(k.replace(" ", "_"))
                return keys
        except Exception:
            continue
    return []

def load_input(path: str) -> Dict[str, Any]:
    """
    Load Model-1 output CSV (one-row) or JSON (testing).
    Returns a dict:
      {
        "age": ...,
        "gender": ...,
        "parameters": { <canonical param>: numeric_or_None, ... },
        "status": { <canonical param>: "LOW"/"HIGH"/"NORMAL", ... },
        "notes": { <canonical param>: "..." },
        "missing_params": [...],
        "quality": {...}   # new: quality flags from quality_checks.detect_low_data_quality
      }
    """
    result = {
        "parameters": {},
        "status": {},
        "notes": {},
    }

    expected_params = _load_param_map_expected()

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        # flatten tester JSON -> result
        for k, v in j.items():
            key = _canonical_key(k)
            if key.lower() in ("age", "gender", "patient_id", "filename", "report_date"):
                result[key] = v
                continue
            # try numeric cast
            num = _cast_number(v)
            if isinstance(num, (int, float)) or num is None:
                result["parameters"][key] = num
            else:
                # non-numeric strings could be statuses/notes
                up = str(v).strip().upper()
                if up in ("LOW", "HIGH", "NORMAL"):
                    result["status"][key] = up
                else:
                    result["notes"][key] = str(v)
        # quality checks
        quality = detect_low_data_quality(result["parameters"], expected_params=expected_params)
        result["quality"] = quality
        # missing params relative to expected_params
        missing = []
        if expected_params:
            for p in expected_params:
                if p not in result["parameters"] or result["parameters"].get(p) is None:
                    missing.append(p)
        result["missing_params"] = missing
        return result

    # CSV case
    with open(path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        try:
            row = next(reader)
        except StopIteration:
            # no row -> low quality
            result["quality"] = {"low_data_quality": True, "reason": "empty_csv"}
            result["missing_params"] = expected_params or []
            return result

    # process columns
    for raw_h, raw_v in row.items():
        norm = _normalize_header(raw_h)
        # detect status/note columns by suffix before canonicalization
        lowkey = norm.lower()
        if lowkey.endswith("_status") or lowkey.endswith(" status"):
            base = norm.rsplit("_", 1)[0]
            canonical = _canonical_key(base)
            val = (raw_v or "").strip()
            if val != "":
                result["status"][canonical] = val.upper()
            continue
        if lowkey.endswith("_note") or lowkey.endswith(" note"):
            base = norm.rsplit("_", 1)[0]
            canonical = _canonical_key(base)
            val = (raw_v or "").strip()
            if val != "":
                result["notes"][canonical] = val
            continue

        canonical = _canonical_key(norm)
        # metadata
        if canonical.lower() in ("age", "gender", "patient_id", "filename", "report_date"):
            if canonical.lower() == "age":
                result["age"] = _cast_number(raw_v)
            else:
                result[canonical] = (raw_v or "").strip() if raw_v is not None else None
            continue

        # numeric lab
        val = _cast_number(raw_v)

        # ONLY numeric values are allowed as parameters
        if isinstance(val, (int, float)) or val is None:
            result["parameters"][canonical] = val
        # everything else is ignored here (strings handled earlier as status/note)

    # Now compute quality flags and missing params
    quality = detect_low_data_quality(result["parameters"], expected_params=expected_params)
    result["quality"] = quality

    # missing params list
    missing = []
    if expected_params:
        for p in expected_params:
            if p not in result["parameters"] or result["parameters"].get(p) is None:
                missing.append(p)
    result["missing_params"] = missing

    return result
