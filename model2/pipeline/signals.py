"""
Signal extraction: convert patterns/derived/params into neutral physiological signals.

Signals are not diagnoses. They are normalized strengths (0.0-1.0) representing the
degree to which a physiological deviation is present in the report.
Also provides enrichment helpers to mark patterns as 'subclinical' when signals exist.
"""
from typing import Dict, Any
import math

def _clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _score_from_severity_label(label: str) -> float:
    # map severity labels from severity.label to approximate numeric strength
    if not label:
        return 0.0
    l = str(label).lower()
    if "very_severe" in l:
        return 0.98
    if "severe" in l and "very" not in l:
        return 0.9
    if l.startswith("high") or l.startswith("low"):
        return 0.7
    if l.startswith("borderline"):
        return 0.45
    if l == "normal":
        return 0.0
    return 0.2

def extract_signals(patterns: Dict[str, Any], derived: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
    """
    Return a dict of signals -> strength
    Uses patterns (from detect_patterns), derived metrics and raw params to compute normalized signals.
    """
    p = patterns.get("patterns", {}) if isinstance(patterns, dict) else {}

    signals = {
        "platelet_suppression": 0.0,
        "erythrocyte_abnormality": 0.0,
        "lipid_dysregulation": 0.0,
        "systemic_inflammation": 0.0,
        "renal_stress": 0.0,
        "glycemic_instability": 0.0,
        "metabolic_sign": 0.0
    }

    # Platelet suppression
    thromb = p.get("thrombocytopenia", {})
    if thromb.get("present"):
        sev = thromb.get("severity") or ""
        signals["platelet_suppression"] = _clamp01(_score_from_severity_label(sev))

    # Erythrocyte abnormalities (anemia patterns)
    anemia = p.get("anemia", {})
    if anemia.get("present"):
        sev = anemia.get("severity") or ""
        type_bonus = 0.1 if anemia.get("type") in ("microcytic", "macrocytic") else 0.0
        signals["erythrocyte_abnormality"] = _clamp01(_score_from_severity_label(sev) + type_bonus)

    # Inflammation
    infl = p.get("inflammation", {})
    if infl.get("present"):
        sev = infl.get("severity") or ""
        signals["systemic_inflammation"] = _clamp01(_score_from_severity_label(sev))

    # Lipid dysregulation: use dyslipidemia pattern + derived ratios if present
    dys = p.get("dyslipidemia", {})
    lip_strength = 0.0
    if dys.get("present"):
        lip_strength = 0.7
    # incorporate TG_to_HDL_ratio and LDL_estimated if present
    tg_hdl = derived.get("TG_to_HDL_ratio")
    ldl_est = derived.get("LDL_estimated")
    try:
        if tg_hdl is not None:
            lip_strength = max(lip_strength, min(1.0, tg_hdl / 5.0))
        if ldl_est is not None:
            if ldl_est >= 190:
                lip_strength = max(lip_strength, 0.9)
            elif ldl_est >= 160:
                lip_strength = max(lip_strength, 0.7)
            elif ldl_est >= 130:
                lip_strength = max(lip_strength, 0.45)
    except Exception:
        pass
    signals["lipid_dysregulation"] = _clamp01(lip_strength)

    # Metabolic sign: TG/HDL low HDL and TG high or glucose/HbA1c
    meta = p.get("metabolic_syndrome_signals", {})
    meta_strength = 0.0
    if meta.get("present"):
        meta_strength = 0.6
    try:
        a1c = params.get("HbA1c")
        glucose = params.get("Glucose_Fasting")
        if a1c is not None:
            a1c_f = float(a1c)
            if a1c_f >= 6.5:
                meta_strength = max(meta_strength, 0.9)
            elif a1c_f >= 5.7:
                meta_strength = max(meta_strength, 0.5)
        if glucose is not None:
            g = float(glucose)
            if g >= 126:
                meta_strength = max(meta_strength, 0.8)
            elif g >= 100:
                meta_strength = max(meta_strength, 0.45)
    except Exception:
        pass
    signals["metabolic_sign"] = _clamp01(meta_strength)

    # glycemic instability uses a1c/glucose
    gly = 0.0
    try:
        if params.get("HbA1c") is not None:
            v = float(params.get("HbA1c"))
            gly = 0.9 if v >= 6.5 else (0.5 if v >= 5.7 else 0.0)
        elif params.get("Glucose_Fasting") is not None:
            gv = float(params.get("Glucose_Fasting"))
            gly = 0.8 if gv >= 126 else (0.45 if gv >= 100 else 0.0)
    except Exception:
        pass
    signals["glycemic_instability"] = _clamp01(gly)

    # Renal stress from creatinine/urea
    try:
        cr = params.get("Creatinine")
        ub = params.get("Urea_BUN")
        rscore = 0.0
        if cr is not None:
            cr_v = float(cr)
            if cr_v > 1.5:
                rscore = max(rscore, 0.8)
            elif cr_v > 1.2:
                rscore = max(rscore, 0.45)
        if ub is not None:
            ub_v = float(ub)
            if ub_v > 25:
                rscore = max(rscore, 0.6)
        signals["renal_stress"] = _clamp01(rscore)
    except Exception:
        signals["renal_stress"] = 0.0

    # normalize and ensure all keys exist
    for k in list(signals.keys()):
        signals[k] = float(round(_clamp01(signals[k]), 3))

    return signals

# ---------------------------------------------------------------------
# Enrichment helper: convert strong signals into 'subclinical' flags for patterns
# ---------------------------------------------------------------------
def enrich_patterns_with_signals(patterns: Dict[str, Any], signals: Dict[str, float], threshold: float = 0.35) -> None:
    """
    Mutate `patterns` in-place: for patterns that are NOT present but have a corresponding
    signal >= threshold, mark pattern['subclinical'] = True.
    This allows downstream components to treat mild/early physiology separately from 'present' pathology.
    """
    if not isinstance(patterns, dict) or not isinstance(signals, dict):
        return

    MAP_SIGNAL_TO_PATTERN = {
        "platelet_suppression": "thrombocytopenia",
        "erythrocyte_abnormality": "anemia",
        "lipid_dysregulation": "dyslipidemia",
        "systemic_inflammation": "inflammation",
        "renal_stress": "inflammation",  # renal stress often coexists with inflammation; choose conservative mapping
        "glycemic_instability": "metabolic_syndrome_signals",
        "metabolic_sign": "metabolic_syndrome_signals",
    }

    for sig, val in signals.items():
        if val is None:
            continue
        try:
            if float(val) >= threshold:
                patt_name = MAP_SIGNAL_TO_PATTERN.get(sig)
                if not patt_name:
                    continue
                patt = patterns.get(patt_name)
                if isinstance(patt, dict):
                    # only mark subclinical if not already present
                    if not patt.get("present") and not patt.get("subclinical"):
                        patt["subclinical"] = True
                        patt.setdefault("support", []).append(f"{sig.upper()}_SUBCLINICAL")
        except Exception:
            continue
