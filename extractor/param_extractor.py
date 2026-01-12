"""
extractor/param_extractor.py
Robust parameter extraction module.

Provides:
 - extract_params_from_text(text) -> list[dict]
 - fallback_line_scan(text) -> list[dict]

Each dict:
{
  "raw_name": str,
  "raw_value": str | None,
  "value": float | None,
  "unit": str | None,
  "canonical": str | None,
  "match_confidence": float,
  "source": str | None,   # e.g. "regex", "line_firstnum", "special_line_start", "line_token"
  "raw_row_text": str | None,
  "value_confidence": "high"|"medium"|"low"|"unknown",
  "suspect_reason": str | None
}
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ----------------- load param_map.json -----------------
BASE = Path(__file__).resolve().parents[0]
PMAP = BASE / "param_map.json"
if not PMAP.exists():
    PMAP = Path(__file__).resolve().parents[1] / "extractor" / "param_map.json"

if PMAP.exists():
    with open(PMAP, "r", encoding="utf-8") as f:
        PARAM_MAP = json.load(f)
else:
    PARAM_MAP = {}

# Build alias -> canonical lookup
CANONICAL_LOOKUP: Dict[str, str] = {}
for canon, info in PARAM_MAP.items():
    aliases = []
    if isinstance(info, dict):
        aliases = info.get("aliases", [])
    elif isinstance(info, list):
        aliases = info
    for a in aliases:
        CANONICAL_LOOKUP[str(a).lower()] = canon
    CANONICAL_LOOKUP[str(canon).lower()] = canon

# --------- patterns ----------
CANDIDATE_PARAM_PATTERN = re.compile(
    r'([A-Za-z0-9%()./\-\s]{3,40})\s*[:\-]?\s*([<>]?\s?-?\d+(?:[.,]\d+)?)\s*'
    r'(mg/dL|g/dL|g/L|mmol/L|µIU/mL|uIU/mL|IU/L|%|mmol/L|ng/mL|pg/mL|U/L|mm/hr)?',
    flags=re.IGNORECASE,
)

NUM_PATTERN = re.compile(r'([<>]?\s?-?\d+(?:[.,]\d+)?)')

# (old) line-start pattern is still kept for compatibility
LINE_START_PATTERN = re.compile(
    r'^\s*([A-Za-z][A-Za-z0-9 ()/%.,\-]{2,50})\s+([<>]?\s?-?\d+(?:[.,]\d+)?)\s*'
    r'(mg/dL|g/dL|g/L|mmol/L|µIU/mL|uIU/mL|IU/L|%|mmol/L|ng/mL|pg/mL|U/L|mm/hr)?',
    flags=re.IGNORECASE,
)

# New: pattern to detect explicit numeric ranges inside a line (e.g., "26-34", "26 – 34", "26 to 34")
RANGE_PATTERN = re.compile(r'(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s+to\s+(\d+(?:[.,]\d+)?)',
                           flags=re.IGNORECASE)

def _looks_like_name(s: str) -> bool:
    """Filter out junk names like '-' or mostly punctuation."""
    if not s:
        return False
    s = s.strip()
    if len(s) < 2:
        return False
    # must contain at least one letter
    if not any(c.isalpha() for c in s):
        return False
    # reject if letters are <30% of non-space chars
    letters = sum(1 for c in s if c.isalpha())
    total = len(s.replace(" ", ""))
    if total and letters / total < 0.3:
        return False
    return True

def normalize_number(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    # strip leading < or >
    s = s.lstrip("<>").strip()
    try:
        return float(s)
    except Exception:
        m = re.search(r'-?\d+(?:\.\d+)?', s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None

# Try rapidfuzz if available, else fallback
try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    process = None
    fuzz = None
    _HAS_RAPIDFUZZ = False

def fuzzy_canonical(name: str, threshold: int = 78) -> Tuple[Optional[str], float]:
    """
    Match extracted name -> canonical. Use rapidfuzz if available, else substring matching.
    Returns (canonical_name, score) where score is 0..1.
    """
    if not name:
        return None, 0.0
    name_clean = re.sub(r'[^A-Za-z0-9 ]', ' ', name).strip().lower()
    if not name_clean:
        return None, 0.0

    # direct lookup
    if name_clean in CANONICAL_LOOKUP:
        return CANONICAL_LOOKUP[name_clean], 1.0

    # token-level lookup
    tokens = [t for t in name_clean.split() if len(t) > 1]
    for token in tokens:
        if token in CANONICAL_LOOKUP:
            return CANONICAL_LOOKUP[token], 0.9

    # rapidfuzz matching
    if _HAS_RAPIDFUZZ and CANONICAL_LOOKUP and len(name_clean) >= 3:
        choices = list(CANONICAL_LOOKUP.keys())
        match = process.extractOne(name_clean, choices, scorer=fuzz.WRatio)
        if match:
            matched_str, score, _ = match
            if score >= threshold:
                return CANONICAL_LOOKUP[matched_str], score / 100.0
            return None, score / 100.0

    # fallback substring check
    for k in CANONICAL_LOOKUP:
        if k in name_clean and len(k) >= 3:
            return CANONICAL_LOOKUP[k], 0.6

    return None, 0.0

# Helper: detect numeric spans corresponding to explicit ranges in a line
def _find_range_spans(line: str) -> List[Tuple[int, int, float, float]]:
    spans = []
    for m in RANGE_PATTERN.finditer(line):
        try:
            # two alternative groups: (a-b) or (a to b)
            if m.group(1) and m.group(2):
                a = normalize_number(m.group(1))
                b = normalize_number(m.group(2))
                spans.append((m.start(), m.end(), a, b))
            elif m.group(3) and m.group(4):
                a = normalize_number(m.group(3))
                b = normalize_number(m.group(4))
                spans.append((m.start(), m.end(), a, b))
        except Exception:
            continue
    return spans

def _num_in_range_spans(num_span_start: int, num_span_end: int, range_spans: List[Tuple[int, int, float, float]]) -> bool:
    for rs, re, a, b in range_spans:
        # overlap test
        if not (num_span_end < rs or num_span_start > re):
            return True
    return False

def _select_best_number_from_line(line: str) -> Tuple[Optional[str], Optional[float], str]:
    """
    Choose the best numeric token from a line for 'result' determination.

    Returns (raw_num_text, numeric_value, reason)
      reason: "not_in_range_span", "first_nonrange", "first_fallback"
    """
    nums = list(re.finditer(NUM_PATTERN, line))
    if not nums:
        return None, None, "no_numbers"
    range_spans = _find_range_spans(line)

    # prefer the first numeric token that is NOT inside an explicit range span
    for m in nums:
        s, e = m.start(1), m.end(1)
        if not _num_in_range_spans(s, e, range_spans):
            raw = m.group(1).strip()
            val = normalize_number(raw)
            return raw, val, "not_in_range_span"

    # if all numbers are within range spans, try to select a number outside right-side (less likely)
    # fallback to first number but mark as suspect
    first = nums[0]
    raw = first.group(1).strip()
    val = normalize_number(raw)
    return raw, val, "first_fallback"

def extract_params_from_text(text: str) -> List[Dict]:
    """
    Global regex scan. (kept for backward compatibility)
    """
    out: List[Dict] = []
    if not text:
        return out

    for m in CANDIDATE_PARAM_PATTERN.finditer(text):
        raw_name = m.group(1).strip()
        if not _looks_like_name(raw_name):
            continue

        raw_value = m.group(2).strip()
        unit = m.group(3) if m.group(3) else None
        val = normalize_number(raw_value)
        canon, score = fuzzy_canonical(raw_name)

        entry = {
            "raw_name": raw_name,
            "raw_value": raw_value,
            "value": val,
            "unit": unit,
            "canonical": canon,
            "match_confidence": score,
            "source": "regex",
            "raw_row_text": None,
            "value_confidence": "high" if val is not None else "unknown",
            "suspect_reason": None
        }
        out.append(entry)
    return out

def fallback_line_scan(text: str) -> List[Dict]:
    """
    Secondary strategy, more robust for table rows like:
      'CK 1416 IU/L 38-204'
      'ALP 72 IU/L 40-129'

    Steps:
      A0) new: take substring before the FIRST number as name, and pick the best non-range number as value
          -> source = 'line_firstnum' (but with safeguards)
      A)  old special line-start regex (kept)          -> 'special_line_start'
      B)  generic line scanning with tokens           -> 'line_token'
    """
    out: List[Dict] = []
    if not text:
        return out

    for line in text.splitlines():
        if not line:
            continue
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # attach raw_row_text for debugging
        raw_row_text = line_stripped

        # ---------- A0) NEW: smarter first-number heuristic ----------
        # Attempt to choose a numeric token that is NOT part of an explicit reference range
        raw_num, val_first, reason = _select_best_number_from_line(line_stripped)
        if raw_num is not None:
            name_part = line_stripped[:line_stripped.find(raw_num)].strip(":- \t") if raw_num in line_stripped else line_stripped.split()[0]
            if _looks_like_name(name_part):
                canon_first, score_first = fuzzy_canonical(name_part, threshold=70)
                if canon_first and val_first is not None:
                    # determine confidence: if chosen numeric token was a fallback inside a range span, mark low
                    if reason == "not_in_range_span":
                        value_conf = "medium" if score_first < 0.8 else "high"
                        suspect_reason = None
                    else:
                        value_conf = "low"
                        suspect_reason = "value_may_be_range_bound_or_fallback"
                    out.append({
                        "raw_name": name_part,
                        "raw_value": raw_num,
                        "value": val_first,
                        "unit": None,
                        "canonical": canon_first,
                        "match_confidence": min(1.0, (score_first or 0.0) + 0.05),
                        "source": "line_firstnum",
                        "raw_row_text": raw_row_text,
                        "value_confidence": value_conf,
                        "suspect_reason": suspect_reason
                    })

        # ---------- A) existing special line-start pattern ----------
        m = LINE_START_PATTERN.match(line_stripped)
        if m:
            raw_name = m.group(1).strip()
            raw_value = m.group(2).strip()
            unit = m.group(3) if m.group(3) else None
            if _looks_like_name(raw_name):
                val = normalize_number(raw_value)
                canon, score = fuzzy_canonical(raw_name, threshold=70)
                if canon and val is not None:
                    # check if the value equals a nearby range bound (if a range is present)
                    range_spans = _find_range_spans(line_stripped)
                    suspect = False
                    suspect_reason = None
                    if range_spans:
                        # extract bounds from first range span for sanity check
                        a = range_spans[0][2]
                        b = range_spans[0][3]
                        if val is not None and (a is not None and b is not None):
                            if val == a or val == b:
                                suspect = True
                                suspect_reason = "value_equals_range_bound"
                    out.append({
                        "raw_name": raw_name,
                        "raw_value": raw_value,
                        "value": val,
                        "unit": unit,
                        "canonical": canon,
                        "match_confidence": score,
                        "source": "special_line_start",
                        "raw_row_text": raw_row_text,
                        "value_confidence": "low" if suspect else "high",
                        "suspect_reason": suspect_reason
                    })

        # ---------- B) generic token-based scan ----------
        # require both letters and digits in line (lab-like)
        if not any(c.isalpha() for c in line_stripped) or not any(c.isdigit() for c in line_stripped):
            continue

        # find numbers and attempt to pick a best candidate for 'value'
        nums_all = list(re.finditer(NUM_PATTERN, line_stripped))
        chosen_raw = None
        chosen_val = None
        chosen_reason = None

        if nums_all:
            # prefer first numeric that's not inside explicit range spans
            range_spans = _find_range_spans(line_stripped)
            for nm in nums_all:
                s, e = nm.start(1), nm.end(1)
                if not _num_in_range_spans(s, e, range_spans):
                    chosen_raw = nm.group(1).strip()
                    chosen_val = normalize_number(chosen_raw)
                    chosen_reason = "not_in_range_span"
                    break
            if chosen_raw is None:
                # fallback to first number but mark low confidence
                firstm = nums_all[0]
                chosen_raw = firstm.group(1).strip()
                chosen_val = normalize_number(chosen_raw)
                chosen_reason = "first_fallback"

        tokens = re.split(r'[\t,;|:]', line_stripped)
        for token in tokens:
            token_stripped = token.strip()
            if not _looks_like_name(token_stripped):
                continue
            canon, score = fuzzy_canonical(token_stripped, threshold=85)
            if canon and chosen_val is not None:
                # set confidence based on the chosen_reason from number selection
                if chosen_reason == "not_in_range_span":
                    value_conf = "medium" if score < 0.8 else "high"
                    suspect_reason = None
                elif chosen_reason == "first_fallback":
                    value_conf = "low"
                    suspect_reason = "value_may_be_range_bound_or_fallback"
                else:
                    value_conf = "unknown"
                    suspect_reason = None

                out.append({
                    "raw_name": token_stripped,
                    "raw_value": chosen_raw,
                    "value": chosen_val,
                    "unit": None,
                    "canonical": canon,
                    "match_confidence": score,
                    "source": "line_token",
                    "raw_row_text": raw_row_text,
                    "value_confidence": value_conf,
                    "suspect_reason": suspect_reason
                })

    return out

__all__ = [
    "extract_params_from_text",
    "fallback_line_scan",
    "normalize_number",
    "fuzzy_canonical",
    "PARAM_MAP",
]
