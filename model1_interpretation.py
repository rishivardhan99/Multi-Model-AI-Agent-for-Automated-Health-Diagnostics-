"""
Model-1 interpretation (updated)
- Loads param_map.json ranges
- Classifies values into interpretation labels expected by table_utils:
    interpretation (LOW, LOW-BORDER, NORMAL, HIGH-BORDER, HIGH, UNKNOWN, INVALID)
  and interpretation_reason (human-friendly short reason)
- Saves CSV/JSON and units_used JSON and flushes files.
- Returns df_interp (with needed columns) and units_used dict.
"""

from pathlib import Path
import json
import math
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from validation_and_standardization import _norm_key_simple

ROOT = Path(__file__).resolve().parents[0]
PARAM_MAP_PATH = ROOT / "extractor" / "param_map.json"

# load PARAM_MAP (fall back to empty)
try:
    with open(PARAM_MAP_PATH, "r", encoding="utf-8") as fh:
        PARAM_MAP = json.load(fh)
except Exception:
    PARAM_MAP = {}

# Build normalized mapping for robust lookup (reuse same normalization rule)
NORMALIZED_TO_CANON: dict = {}
for canonical, info in PARAM_MAP.items():
    NORMALIZED_TO_CANON[_norm_key_simple(canonical)] = canonical
    aliases = info.get("aliases", []) if isinstance(info, dict) else []
    for a in aliases:
        NORMALIZED_TO_CANON[_norm_key_simple(a)] = canonical
    param_id = info.get("param_id") if isinstance(info, dict) else None
    if param_id:
        NORMALIZED_TO_CANON[_norm_key_simple(str(param_id))] = canonical


def _canonical_lookup(key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    s = str(key).strip()
    if not s:
        return None
    norm = _norm_key_simple(s)
    # prefer normalized mapping
    if norm in NORMALIZED_TO_CANON:
        return NORMALIZED_TO_CANON[norm]
    # fallback: exact-case match in PARAM_MAP
    for k in PARAM_MAP.keys():
        if k.lower() == s.lower():
            return k
    # fallback: try aliases
    for k, info in PARAM_MAP.items():
        aliases = info.get("aliases", []) if isinstance(info, dict) else []
        for a in aliases:
            if str(a).strip().lower() == s.lower():
                return k
    # last fallback: return original string (preserve)
    return key


def _get_range_and_unit_from_map(canonical: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if canonical is None:
        return None, None, None
    # prefer normalized lookup
    norm = _norm_key_simple(canonical)
    entry = None
    if norm in NORMALIZED_TO_CANON:
        entry = PARAM_MAP.get(NORMALIZED_TO_CANON[norm])
    else:
        for k, v in PARAM_MAP.items():
            if k.lower() == str(canonical).strip().lower():
                entry = v
                break
    if entry is None:
        return None, None, None
    rng = entry.get("range", {}) if isinstance(entry, dict) else {}
    lo = rng.get("min")
    hi = rng.get("max")
    units = rng.get("units") or entry.get("unit")
    try:
        lo = float(lo) if lo is not None else None
        hi = float(hi) if hi is not None else None
    except Exception:
        lo = hi = None
    return lo, hi, units


def _safe_isnan(x):
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def classify_value_and_reason(v: Optional[float], lo: Optional[float], hi: Optional[float], border_frac: float = 0.10) -> Tuple[str, str]:
    """
    Returns (interpretation_label, reason_text).
    Labels: LOW, LOW-BORDER, NORMAL, HIGH-BORDER, HIGH, UNKNOWN, INVALID
    """
    if v is None or _safe_isnan(v):
        return "UNKNOWN", "missing_value"
    if lo is None or hi is None:
        return "UNKNOWN", "no_ref_range"

    try:
        v_f = float(v)
        lo_f = float(lo)
        hi_f = float(hi)
    except Exception:
        return "UNKNOWN", "non_numeric"

    if lo_f >= hi_f:
        return "UNKNOWN", "invalid_ref_range"

    span = hi_f - lo_f
    margin = span * float(border_frac)

    low_border = lo_f
    low_border_zone_start = lo_f - margin
    high_border = hi_f
    high_border_zone_end = hi_f + margin

    # classification:
    if v_f < low_border_zone_start:
        return "LOW", f"{v_f} < {lo_f}"
    if low_border_zone_start <= v_f < lo_f:
        return "LOW-BORDER", f"{v_f} near lower bound {lo_f}"
    if lo_f <= v_f <= hi_f:
        return "NORMAL", f"{v_f} within [{lo_f},{hi_f}]"
    if hi_f < v_f <= high_border_zone_end:
        return "HIGH-BORDER", f"{v_f} near upper bound {hi_f}"
    if v_f > high_border_zone_end:
        return "HIGH", f"{v_f} > {hi_f}"
    return "UNKNOWN", "unhandled_case"


def interpret_dataframe(df_std: pd.DataFrame,
                        save_outputs: bool = True,
                        out_dir: Optional[Path] = None,
                        basename: Optional[str] = "model1_output",
                        border_frac: float = 0.10,
                        units_tracking: bool = True,
                        save_json: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    df_std expected to have columns:
      patient_id, filename, age, gender, parameter, canonical, raw_value, value_std, value_num, unit_std, valid, invalid_reason
    Returns df_interp (with interpretation columns) and units_used dict.
    """
    if out_dir is None:
        out_dir = ROOT / "outputs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defensive: ensure expected columns
    cols_expected = ["patient_id","filename","age","gender","parameter","canonical","raw_value","value_std","value_num","unit_std","valid","invalid_reason"]
    for c in cols_expected:
        if c not in df_std.columns:
            df_std[c] = None

    rows = []
    units_used = {}

    for _, r in df_std.iterrows():
        # canonical normalization: prefer actual param_map canonical name if possible
        canon_raw = r.get("canonical")
        canon = _canonical_lookup(canon_raw)
        # ensure we preserve canonical column in output exactly as used in your pipeline
        if canon is None and isinstance(canon_raw, str):
            canon = canon_raw

        value = r.get("value_num")
        observed_unit = r.get("unit_std") or ""
        lo, hi, std_unit = _get_range_and_unit_from_map(canon)

        # track units mapping
        if units_tracking:
            k = str(canon).strip() if canon else "UNKNOWN_PARAM"
            units_used.setdefault(k, {"observed_units": set(), "standard_unit": std_unit})
            if observed_unit:
                units_used[k]["observed_units"].add(observed_unit)

        # if row marked invalid, mark invalid interpretation
        if not bool(r.get("valid", True)):
            interp = "INVALID"
            reason = str(r.get("invalid_reason") or "marked_invalid")
        else:
            interp, reason = classify_value_and_reason(value, lo, hi, border_frac=border_frac)

        row_out = {
            "patient_id": r.get("patient_id"),
            "filename": r.get("filename"),
            "age": r.get("age"),
            "gender": r.get("gender"),
            "parameter": r.get("parameter"),
            "canonical": canon,
            "raw_value": r.get("raw_value"),
            "value_std": r.get("value_std"),
            "value_num": r.get("value_num"),
            "unit_std": observed_unit,
            "std_unit": std_unit,
            "ref_min": lo,
            "ref_max": hi,
            # These two columns are *required* by downstream code (table_utils)
            "interpretation": interp,
            "interpretation_reason": reason,
            # backward-compatible alias
            "classification": interp,
        }
        rows.append(row_out)

    df_interp = pd.DataFrame(rows)

    # make sets -> lists for units_used
    clean_units = {}
    for k, v in units_used.items():
        clean_units[k] = {
            "observed_units": sorted(list(v["observed_units"])) if v.get("observed_units") else [],
            "standard_unit": v.get("standard_unit")
        }

    # Save outputs if requested (paths match Streamlit usage)
    base = basename or "model1_output"
    csv_path = Path(out_dir) / f"{base}.model1_interpreted.csv"
    json_path = Path(out_dir) / f"{base}.model1_interpreted.json"
    units_path = Path(out_dir) / f"{base}.units_used.json"

    if save_outputs:
        try:
            df_interp.to_csv(csv_path, index=False)
            # flush to disk (best-effort)
            with open(csv_path, "a", encoding="utf-8") as fh:
                fh.flush()
            print(f"[Model1] Saved interpreted CSV → {csv_path}")
        except Exception as e:
            print(f"[Model1] ERROR saving CSV: {e}")

        if save_json:
            try:
                df_interp.to_json(json_path, orient="records", indent=2)
                print(f"[Model1] Saved interpreted JSON → {json_path}")
            except Exception as e:
                print(f"[Model1] ERROR saving JSON: {e}")

        try:
            with open(units_path, "w", encoding="utf-8") as fh:
                json.dump(clean_units, fh, indent=2)
                fh.flush()
            print(f"[Model1] Saved units_used.json → {units_path}")
        except Exception as e:
            print(f"[Model1] ERROR saving units_used.json: {e}")

    return df_interp, clean_units


# quick test when executed directly
if __name__ == "__main__":
    import pandas as pd
    # tiny smoke test
    df = pd.DataFrame([{
        "patient_id":"p1","filename":"s.pdf","age":30,"gender":"Male",
        "parameter":"Hemoglobin","canonical":"Hemoglobin","raw_value":"13","value_std":"13","value_num":13.0,"unit_std":"g/dL","valid":True
    }])
    df_i, units = interpret_dataframe(df, save_outputs=False, basename="smoke_test")
    print(df_i.head())
    print("units:", units)
