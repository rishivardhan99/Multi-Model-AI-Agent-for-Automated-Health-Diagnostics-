"""
Signal extraction: convert patterns/derived/params into neutral physiological signals.

Signals are not diagnoses. They are normalized strengths (0.0-1.0) representing the
degree to which a physiological deviation is present in the report.

This variant *gates* signals on pattern presence to avoid signal leakage that
creates spurious themes and downstream hallucinations.
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
    if "very_severe" in l or "very severe" in l:
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

    IMPORTANT: signals are *gated* by the presence of the corresponding pattern to avoid
    spurious signal creation when no pattern was detected.
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

    # Platelet suppression (strict: depends on thrombocytopenia.present)
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

    # Inflammation (only if inflammation pattern present)
    infl = p.get("inflammation", {})
    if infl.get("present"):
        sev = infl.get("severity") or ""
        signals["systemic_inflammation"] = _clamp01(_score_from_severity_label(sev))

    # Lipid dysregulation:
    # STRICT GATE: only compute lipid signal if dyslipidemia pattern present
    dys = p.get("dyslipidemia", {}) or {}
    lip_strength = 0.0
    if dys.get("present"):
        # baseline from pattern
        lip_strength = 0.7
        # derived metrics may increase strength (but only if pattern exists)
        try:
            tg_hdl = derived.get("TG_to_HDL_ratio")
            ldl_est = derived.get("LDL_estimated")
            if tg_hdl is not None:
                # scale TG/HDL ratio gently (expect ratios in sensible range)
                lip_strength = max(lip_strength, min(1.0, float(tg_hdl) / 5.0))
            if ldl_est is not None:
                ldl_val = float(ldl_est)
                if ldl_val >= 190:
                    lip_strength = max(lip_strength, 0.9)
                elif ldl_val >= 160:
                    lip_strength = max(lip_strength, 0.7)
                elif ldl_val >= 130:
                    lip_strength = max(lip_strength, 0.45)
        except Exception:
            pass
    signals["lipid_dysregulation"] = _clamp01(lip_strength)

    # Metabolic sign: gated by metabolic_syndrome_signals pattern
    meta = p.get("metabolic_syndrome_signals", {}) or {}
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

    # Glycemic instability: gated on metabolic_syndrome_signals pattern
    gly = 0.0
    if p.get("metabolic_syndrome_signals", {}).get("present"):
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

    # Renal stress: conservative gating. Only set renal signal if inflammation pattern present
    # (conservative choice to avoid treating minor creatinine blips as systemic renal stress)
    rscore = 0.0
    if p.get("inflammation", {}).get("present"):
        try:
            cr = params.get("Creatinine")
            ub = params.get("Urea_BUN")
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
        except Exception:
            pass
    signals["renal_stress"] = _clamp01(rscore)

    # normalize and ensure all keys exist
    for k in list(signals.keys()):
        signals[k] = float(round(_clamp01(signals[k]), 3))

    return signals

# ---------------------------------------------------------------------
# Enrichment helper: convert strong signals into 'subclinical' flags for patterns
# ---------------------------------------------------------------------
def enrich_patterns_with_signals(patterns: Dict[str, Any], signals: Dict[str, float], threshold: float = 0.6) -> None:
    """
    Mutate `patterns` in-place: for patterns that are NOT present but have a corresponding
    signal >= threshold, mark pattern['subclinical'] = True.
    This allows downstream components to treat mild/early physiology separately from 'present' pathology.

    Default threshold increased to 0.6 to avoid amplifying weak/noisy signals.
    """
    if not isinstance(patterns, dict) or not isinstance(signals, dict):
        return

    MAP_SIGNAL_TO_PATTERN = {
        "platelet_suppression": "thrombocytopenia",
        "erythrocyte_abnormality": "anemia",
        "lipid_dysregulation": "dyslipidemia",
        "systemic_inflammation": "inflammation",
        # do NOT map renal_stress -> inflammation to avoid feedback amplification
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
