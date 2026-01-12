from pathlib import Path
import re
import json
import math
from typing import Tuple, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
PARAM_MAP_PATH = ROOT / "extractor" / "param_map.json"


def _norm_key_simple(s: str) -> str:
    if s is None:
        return ""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def is_implausible(canon: str, value: float) -> bool:
    """
    Conservative physiological sanity check.
    Drops OCR-scale errors (e.g. Free T4 = 14 instead of 1.4).
    Uses normalized canonical lookup to find ranges in PARAM_MAP.

    NEW: Also checks 'hard_limits' in PARAM_MAP entries (if present).
    If value lies strictly outside hard_limits -> mark implausible.
    """
    try:
        if value is None:
            return False
        # Normalize incoming canon key
        norm = _norm_key_simple(canon)

        # lookup canonical resolved key from NORMALIZED_TO_CANON (built below)
        canon_key = NORMALIZED_TO_CANON.get(norm)
        if canon_key is None:
            # no entry found -> cannot mark as implausible based on map
            return False

        info = PARAM_MAP.get(canon_key)
        if not isinstance(info, dict):
            return False

        # ----- NEW: check explicit hard_limits first -----
        hard = info.get("hard_limits") if isinstance(info, dict) else None
        if isinstance(hard, dict):
            hlo = hard.get("min")
            hhi = hard.get("max")
            try:
                if hlo is not None:
                    hlo = float(hlo)
                if hhi is not None:
                    hhi = float(hhi)
                # If hard limit exists and value outside -> implausible
                if hlo is not None and value < hlo:
                    return True
                if hhi is not None and value > hhi:
                    return True
            except Exception:
                # if casting fails, continue to the fallback checks below
                pass
        # -------------------------------------------------

        rng = info.get("range", {}) or {}
        lo = rng.get("min")
        hi = rng.get("max")
        if lo is None and hi is None:
            return False
        # cast
        try:
            if lo is not None:
                lo = float(lo)
        except Exception:
            lo = None
        try:
            if hi is not None:
                hi = float(hi)
        except Exception:
            hi = None

        # conservative 5x/10x checks similar to previous logic but robust
        if lo is not None and hi is not None:
            if value > hi * 5:
                return True
            if value < lo / 10:
                return True
        elif hi is not None:
            if value > hi * 5:
                return True
        elif lo is not None:
            if value < lo / 5:
                return True
    except Exception:
        return False

    return False


# Load param_map.json (used for canonical keys, preferred unit, ranges, aliases)
try:
    with open(PARAM_MAP_PATH, "r", encoding="utf-8") as fh:
        PARAM_MAP = json.load(fh)
except Exception:
    PARAM_MAP = {}

# Build canonical name lookup (lowercase)
CANONICAL_KEYS = {k.lower(): k for k in PARAM_MAP.keys()}

# Build alias -> canonical mapping from param_map (cover common alias lookups)
ALIAS_TO_CANON: dict = {}
for canonical, info in PARAM_MAP.items():
    aliases = info.get("aliases", []) if isinstance(info, dict) else []
    for a in aliases:
        ALIAS_TO_CANON[str(a).strip().lower()] = canonical
    # also map the canonical string itself lower->canonical
    ALIAS_TO_CANON[str(canonical).strip().lower()] = canonical

# Build a normalized -> canonical mapping for robust lookup
NORMALIZED_TO_CANON: dict = {}
for canonical, info in PARAM_MAP.items():
    k_norm = _norm_key_simple(canonical)
    NORMALIZED_TO_CANON[k_norm] = canonical
    # also map aliases normalized -> canonical
    aliases = info.get("aliases", []) if isinstance(info, dict) else []
    for a in aliases:
        NORMALIZED_TO_CANON[_norm_key_simple(a)] = canonical
    # also map param_id if present
    param_id = info.get("param_id") if isinstance(info, dict) else None
    if param_id:
        NORMALIZED_TO_CANON[_norm_key_simple(str(param_id))] = canonical


# ---------- helper: numeric parsing ----------
NUM_RE = re.compile(r'([<>]?\s*-?\d+(?:[.,]\d+)?)')


def parse_number(s: Optional[str]) -> Optional[float]:
    """Extract a numeric value from a string. Returns float or None."""
    if s is None:
        return None
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        try:
            if isinstance(s, float) and math.isnan(s):
                return None
            return float(s)
        except Exception:
            return None
    s = str(s).strip()
    if s == "":
        return None
    # Replace comma decimal
    s_clean = s.replace(" ", "").replace(",", ".")
    # remove any trailing non-numeric (like mg/dL)
    m = NUM_RE.search(s_clean)
    if not m:
        return None
    token = m.group(1)
    token = token.lstrip("<>").strip()
    try:
        return float(token)
    except Exception:
        # fallback: extract first float-like substring
        m2 = re.search(r'-?\d+(?:\.\d+)?', token)
        if m2:
            try:
                return float(m2.group(0))
            except:
                return None
    return None


# ---------- unit conversion helpers ----------
CONVERSION_FACTORS = {
    ("g/l", "g/dl"): 0.1,
    ("g/dl", "g/l"): 10.0,
    ("l/l", "%"): 100.0,
    ("%","l/l"): 0.01,
    ("10^6/ul", "10^12/l"): 1000.0,
    ("10^12/l", "10^6/ul"): 0.001,
    ("10^3/ul", "10^9/l"): 1.0,
    ("10^9/l", "10^3/ul"): 1.0,
    ("mg/dl", "umol/l"): 88.4,
    ("umol/l", "mg/dl"): 1.0/88.4,
    ("mg/dl", "mmol/l"): 0.01129,
    ("mmol/l", "mg/dl"): 1/0.01129,
}


def _norm_unit(u: Optional[str]) -> str:
    if u is None:
        return ""
    return str(u).strip().lower()


def convert_value(value: Optional[float], obs_unit: str, std_unit: str) -> Optional[float]:
    """Convert observed numeric value to standard unit if conversion factor exists."""
    if value is None:
        return None
    obs = _norm_unit(obs_unit)
    std = _norm_unit(std_unit)
    if obs == "" or std == "" or obs == std:
        return value
    f = CONVERSION_FACTORS.get((obs, std))
    if f is not None:
        try:
            return float(value) * float(f)
        except Exception:
            return value
    if std in obs or obs in std:
        return value
    return value


# ---------- read_structured_csv ----------
def read_structured_csv(structured_path: Path) -> pd.DataFrame:
    """
    Read a wide structured CSV (the one created from extraction) and convert to long format.
    """
    p = Path(structured_path)
    if not p.exists():
        raise FileNotFoundError(p)

    dfw = pd.read_csv(p, dtype=str).fillna("")
    # expected patient-level fields
    meta_cols = ["filename", "patient_id", "age", "gender"]
    for c in meta_cols:
        if c not in dfw.columns:
            dfw[c] = ""  # ensure present

    # find parameter columns (anything not in meta_cols)
    param_cols = [c for c in dfw.columns if c not in meta_cols]

    # melt
    rows = []
    for _, row in dfw.iterrows():
        meta = {c: row.get(c, "") for c in meta_cols}
        for param in param_cols:
            raw_val = row.get(param, "")
            # map param -> canonical if alias exists (use normalized map)
            pkey = str(param).strip()
            canonical = None
            norm = _norm_key_simple(pkey)
            if norm in NORMALIZED_TO_CANON:
                canonical = NORMALIZED_TO_CANON[norm]
            else:
                # fallback: try direct case-insensitive match
                if pkey.lower() in CANONICAL_KEYS:
                    canonical = CANONICAL_KEYS[pkey.lower()]
                else:
                    # try substring heuristic
                    for kk in CANONICAL_KEYS:
                        if kk in pkey.lower() and len(kk) >= 3:
                            canonical = CANONICAL_KEYS[kk]
                            break
            rows.append({
                "filename": meta["filename"],
                "patient_id": meta["patient_id"],
                "age": meta["age"],
                "gender": meta["gender"],
                "parameter": param,
                "canonical": canonical or param,
                "raw_value": raw_val,
            })

    df_long = pd.DataFrame(rows)
    cols = ["patient_id", "filename", "age", "gender", "parameter", "canonical", "raw_value"]
    df_long = df_long[cols]
    return df_long


# ---------- standardize_dataframe ----------
def standardize_dataframe(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric values, normalize units & produce:
      - value_std (string)
      - value_num (float)
      - unit_std (string)
      - valid (bool), invalid_reason (str)
    If a value is implausible per param_map -> mark invalid and clear raw_value to act like missing.
    """
    df = df_long.copy()

    # ensure columns exist
    for c in ["patient_id", "filename", "age", "gender", "parameter", "canonical", "raw_value"]:
        if c not in df.columns:
            df[c] = ""

    value_nums = []
    value_std = []
    unit_std = []
    valid_flags = []
    invalid_reasons = []
    scale_corrections = []  # 'div10','mul10' or ''
    raw_value_out = []

    # iterate rows and build outputs (exactly one append per input row)
    for idx, r in df.iterrows():
        raw = r.get("raw_value", "")
        canon_raw = r.get("canonical") or ""
        canon_key = _norm_key_simple(canon_raw)

        pm_entry = None
        # resolve canonical map entry if available
        mapped_canonical = NORMALIZED_TO_CANON.get(canon_key)
        if mapped_canonical:
            pm_entry = PARAM_MAP.get(mapped_canonical)

        # preferred unit from param map
        preferred_unit = ""
        if isinstance(pm_entry, dict):
            preferred_unit = pm_entry.get("unit") or pm_entry.get("range", {}).get("units") or ""

        # parse numeric
        num = parse_number(raw)

        # detect unit suffix
        detected_unit = ""
        if isinstance(raw, str) and raw.strip():
            m = re.search(r'[A-Za-z%µμμμ^0-9/\. ]+$', raw.strip())
            if m:
                candidate = m.group(0).strip()
                candidate = candidate.replace("µ", "u").replace("μ", "u")
                if any(ch.isalpha() or ch in "%/" for ch in candidate):
                    detected_unit = candidate

        detected_unit_norm = _norm_unit(detected_unit)
        preferred_unit_norm = _norm_unit(preferred_unit)

        converted_num = num
        unit_to_report = detected_unit_norm or preferred_unit_norm or ""

        # convert units if mapping exists
        if num is not None and preferred_unit_norm and detected_unit_norm and detected_unit_norm != preferred_unit_norm:
            conv = CONVERSION_FACTORS.get((detected_unit_norm, preferred_unit_norm))
            if conv is not None:
                try:
                    converted_num = float(num) * float(conv)
                    unit_to_report = preferred_unit_norm
                except Exception:
                    converted_num = num
            else:
                if detected_unit_norm == "l/l" and preferred_unit_norm == "%":
                    try:
                        converted_num = float(num) * 100.0
                        unit_to_report = preferred_unit_norm
                    except Exception:
                        pass

        # fallback parse if num None
        if num is None and isinstance(raw, str) and raw.strip():
            try:
                token = re.search(r'-?\d+(?:[.,]\d+)?', raw)
                if token:
                    num = float(token.group(0).replace(",", "."))
                    converted_num = num
            except Exception:
                pass

        # OCR DECIMAL SCALE CORRECTION (conservative)
        scale_note = ""
        try:
            if converted_num is not None and isinstance(pm_entry, dict):
                rng = pm_entry.get("range") or {}
                lo = rng.get("min")
                hi = rng.get("max")
                lof = None
                hif = None
                try:
                    if lo is not None:
                        lof = float(lo)
                except Exception:
                    lof = None
                try:
                    if hi is not None:
                        hif = float(hi)
                except Exception:
                    hif = None

                if lof is not None or hif is not None:
                    v = float(converted_num)
                    if lof is not None and hif is not None:
                        span = max(hif - lof, 1.0)
                    else:
                        if lof is not None:
                            span = max(abs(lof) * 0.2, 1.0)
                        elif hif is not None:
                            span = max(abs(hif) * 0.2, 1.0)
                        else:
                            span = 1.0
                    margin = 0.10 * span

                    applied = False
                    if hif is not None:
                        try:
                            if v > hif * 2:
                                v_div = v / 10.0
                                if (lof is None or (lof - margin) <= v_div) and (hif is None or v_div <= (hif + margin)):
                                    converted_num = v_div
                                    scale_note = "div10"
                                    applied = True
                        except Exception:
                            pass

                    if (not applied) and lof is not None:
                        try:
                            if v < lof / 2:
                                v_mul = v * 10.0
                                if (lof is None or (lof - margin) <= v_mul) and (hif is None or v_mul <= (hif + margin)):
                                    converted_num = v_mul
                                    scale_note = "mul10"
                                    applied = True
                        except Exception:
                            pass
        except Exception:
            scale_note = ""

        # determine validity & implausible-drop
        if num is None:
            valid = False
            reason = "no_numeric"
            converted_num = None
        else:
            try:
                if converted_num is None:
                    valid = False
                    reason = "no_numeric_after_processing"
                else:
                    # If implausible per param_map (10x/5x rule OR hard_limits) -> treat as missing (drop)
                    if is_implausible(canon_raw, float(converted_num)):
                        valid = False
                        reason = "implausible_value"
                        # key action: remove the raw_value so downstream behaves like missing
                        converted_num = None
                        raw = ""
                    else:
                        valid = True
                        reason = ""
            except Exception:
                valid = True
                reason = ""

        # append outputs (exactly one per row)
        if converted_num is None:
            value_nums.append(None)
            value_std.append("")
        else:
            try:
                value_nums.append(float(converted_num))
                value_std.append(str(converted_num))
            except Exception:
                value_nums.append(None)
                value_std.append("")

        unit_std.append(unit_to_report if unit_to_report else "")
        valid_flags.append(bool(valid))
        invalid_reasons.append(reason if reason else "")
        scale_corrections.append(scale_note)
        raw_value_out.append(raw if raw is not None else "")

    # assign columns
    df["value_std"] = value_std
    df["value_num"] = value_nums
    df["unit_std"] = unit_std
    df["valid"] = valid_flags
    df["invalid_reason"] = invalid_reasons
    df["scale_correction"] = scale_corrections
    # Overwrite raw_value with processed raw_value - this is important so downstream treats implausible as missing
    df["raw_value"] = raw_value_out

    # preserve other columns; ensure types
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df
