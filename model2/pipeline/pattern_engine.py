# model2/pipeline/pattern_engine.py
"""
Pattern detection using deterministic, conservative rules.
Patterns represent clinically meaningful groupings, not single-value noise.
"""

from typing import Dict, Any
from .severity import label_from_range

def _is_high(lbl):
    return lbl and str(lbl.get("label", "")).startswith("high")

def _is_low(lbl):
    return lbl and str(lbl.get("label", "")).startswith("low")

def detect_patterns(params: Dict[str, Any]) -> Dict[str, Any]:
    patterns = {}
    age = params.get("age")
    gender = params.get("gender")

    # ----------------------------
    # Core CBC metrics
    # ----------------------------
    hb = label_from_range("Hemoglobin", params.get("Hemoglobin"), age=age, gender=gender)
    mcv = label_from_range("MCV", params.get("MCV"), age=age, gender=gender)
    rdw = label_from_range("RDW", params.get("RDW"), age=age, gender=gender)

    plate = label_from_range("Platelets", params.get("Platelets"), age=age, gender=gender)
    wbc = label_from_range("WBC", params.get("WBC"), age=age, gender=gender)

    neut = label_from_range("Neutrophils_PERCENT", params.get("Neutrophils"), age=age, gender=gender)
    lymph = label_from_range("Lymphocytes_PERCENT", params.get("Lymphocytes"), age=age, gender=gender)

    # ----------------------------
    # Inflammation (CRP / ESR)
    # ----------------------------
    crp = label_from_range("CRP", params.get("CRP"))
    esr = label_from_range("ESR", params.get("ESR"))

    inflammation_present = False
    inflammation_support = []

    if _is_high(crp):
        inflammation_present = True
        inflammation_support.append("CRP_HIGH")
    if _is_high(esr):
        inflammation_present = True
        inflammation_support.append("ESR_HIGH")

    patterns["inflammation"] = {
        "present": inflammation_present,
        "support": inflammation_support,
        "severity": crp.get("label") if crp else (esr.get("label") if esr else "unknown")
    }

    # ----------------------------
    # Anemia
    # ----------------------------
    anemia_present = False
    anemia_type = None
    anemia_support = []

    if _is_low(hb):
        anemia_present = True
        anemia_support.append("Hemoglobin_LOW")

        if _is_low(mcv):
            anemia_type = "microcytic"
            anemia_support.append("MCV_LOW")
        elif _is_high(mcv):
            anemia_type = "macrocytic"
            anemia_support.append("MCV_HIGH")
        else:
            anemia_type = "normocytic"

        if _is_high(rdw):
            anemia_support.append("RDW_HIGH")

    patterns["anemia"] = {
        "present": anemia_present,
        "type": anemia_type,
        "support": anemia_support,
        "severity": hb.get("label") if hb else "unknown",
        "note": hb.get("note") if hb else ""
    }

    # ----------------------------
    # Thrombocytopenia
    # ----------------------------
    thromb_present = _is_low(plate)
    is_isolated = bool(thromb_present and not anemia_present and not _is_low(wbc))

    patterns["thrombocytopenia"] = {
        "present": bool(thromb_present),
        "isolated": bool(is_isolated),
        "severity": plate.get("label") if plate else "unknown",
        "support": ["Platelets_LOW"] if thromb_present else []
    }

    # ----------------------------
    # Infection indicators (supporting only)
    # ----------------------------
    patterns["neutrophilia"] = {
        "present": _is_high(neut),
        "severity": neut.get("label") if neut else "unknown",
        "support": ["Neutrophils_HIGH"] if _is_high(neut) else []
    }

    patterns["lymphocytosis"] = {
        "present": _is_high(lymph),
        "severity": lymph.get("label") if lymph else "unknown",
        "support": ["Lymphocytes_HIGH"] if _is_high(lymph) else []
    }

    # ----------------------------
    # Dyslipidemia (STRICT)
    # Requires ≥2 abnormal lipid parameters
    # ----------------------------
    tc = label_from_range("Total_Cholesterol", params.get("Total_Cholesterol"))
    ldl = label_from_range("LDL", params.get("LDL"))
    tg = label_from_range("Triglycerides", params.get("Triglycerides"))

    lipid_abnormal_count = sum([
        1 for lbl in (tc, ldl, tg) if _is_high(lbl)
    ])

    dyslipid_present = lipid_abnormal_count >= 2
    dys_support = []

    if dyslipid_present:
        if _is_high(tc): dys_support.append("Total_Cholesterol_HIGH")
        if _is_high(ldl): dys_support.append("LDL_HIGH")
        if _is_high(tg): dys_support.append("Triglycerides_HIGH")

    patterns["dyslipidemia"] = {
        "present": dyslipid_present,
        "support": dys_support,
        "details": {"tc": tc, "ldl": ldl, "tg": tg}
    }

    # ----------------------------
    # Metabolic syndrome (STRICT)
    # Diabetes-range only
    # ----------------------------
    glucose = params.get("Glucose_Fasting")
    hba1c = params.get("HbA1c")

    metabolic_present = False
    metabolic_support = []

    try:
        if (hba1c is not None and float(hba1c) >= 6.5):
            metabolic_present = True
            metabolic_support.append("HbA1c_DIABETES_RANGE")
        elif (glucose is not None and float(glucose) >= 126):
            metabolic_present = True
            metabolic_support.append("Glucose_DIABETES_RANGE")
    except Exception:
        pass

    patterns["metabolic_syndrome_signals"] = {
        "present": metabolic_present,
        "support": metabolic_support
    }

    # ----------------------------
    # Normalize severity strings
    # ----------------------------
    for info in patterns.values():
        if isinstance(info, dict) and isinstance(info.get("severity"), str):
            info["severity"] = info["severity"].lower()

    return {"patterns": patterns}
