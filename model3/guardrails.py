"""
Guardrails & sanitization utilities for Model-3 outputs.

Functions:
- scan_for_dangerous_recommendations(list_of_strings) -> (bool, offenders)
- redact_recommendations(list_of_strings) -> sanitized_list
- scan_parsed_object_for_danger(parsed) -> (bool, offenders, sanitized)
"""

from typing import List, Tuple, Any, Dict

DANGEROUS_KEYWORDS = [
    "prescribe", "dosage", "dose", "administer", "surgery",
    "immediate hospitalization", "hospitalize", "call 911", "emergency",
    "must take", "take as directed", "inject", "injectable"
]

SAFE_REDACT_PLACEHOLDER = "Recommendation removed for safety; consult a clinician."

def _find_offenders(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    low = text.lower()
    offenders = [k for k in DANGEROUS_KEYWORDS if k in low]
    return offenders

def scan_for_dangerous_recommendations(items: List[str]) -> Tuple[bool, List[str]]:
    """
    items: list of recommendation strings
    returns (has_danger:bool, offender_keywords:list)
    """
    offenders = []
    for it in items:
        if not isinstance(it, str):
            continue
        offs = _find_offenders(it)
        for o in offs:
            if o not in offenders:
                offenders.append(o)
    return (len(offenders) > 0, offenders)

def redact_recommendations(items: List[str]) -> List[str]:
    """
    Replace any item containing dangerous keywords with a safe placeholder.
    """
    out = []
    for it in items:
        if not isinstance(it, str):
            out.append(SAFE_REDACT_PLACEHOLDER)
            continue
        if _find_offenders(it):
            out.append(SAFE_REDACT_PLACEHOLDER)
        else:
            out.append(it)
    return out

def scan_parsed_object_for_danger(parsed: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Inspect parsed model3 object and return:
      (has_danger, offenders, sanitized_parsed)
    It will specifically scan arrays likely to contain recommendations: lifestyle_guidance, possible_explanations, notes, when_to_consult_doctor.
    If danger is detected, replacements are applied to the sanitized copy.
    """
    had = False
    offenders = []
    san = dict(parsed) if isinstance(parsed, dict) else parsed

    # check lists
    for key in ("lifestyle_guidance", "possible_explanations", "recommendations"):
        val = san.get(key)
        if isinstance(val, list):
            has, offs = scan_for_dangerous_recommendations(val)
            if has:
                had = True
                for o in offs:
                    if o not in offenders:
                        offenders.append(o)
                san[key] = redact_recommendations(val)

    # check when_to_consult_doctor and notes (strings)
    for key in ("when_to_consult_doctor", "notes", "summary"):
        val = san.get(key)
        if isinstance(val, str):
            offs = _find_offenders(val)
            if offs:
                had = True
                for o in offs:
                    if o not in offenders:
                        offenders.append(o)
                # replace offending phrase region with placeholder (conservative)
                san[key] = SAFE_REDACT_PLACEHOLDER

    return had, offenders, san
