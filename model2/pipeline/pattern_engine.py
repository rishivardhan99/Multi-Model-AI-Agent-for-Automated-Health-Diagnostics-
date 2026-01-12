"""
Pattern detection using deterministic rules.
Detects anemia, thrombocytopenia, infection/inflammation signals, dyslipidemia, metabolic signals.
"""
from typing import Dict, Any
from .severity import label_from_range

def detect_patterns(params: Dict[str, Any]) -> Dict[str, Any]:
    patterns = {}
    age = params.get("age")
    gender = params.get("gender")

    # compute labels for core CBC metrics
    hb = label_from_range("Hemoglobin", params.get("Hemoglobin"), age=age, gender=gender)
    mcv = label_from_range("MCV", params.get("MCV"), age=age, gender=gender)
    rdw = label_from_range("RDW", params.get("RDW"), age=age, gender=gender)

    # Platelets and WBC
    plate = label_from_range("Platelets", params.get("Platelets"), age=age, gender=gender)
    wbc = label_from_range("WBC", params.get("WBC"), age=age, gender=gender)

    # Neutrophils and lymphocytes are usually percentages in many lab reports
    neut = label_from_range("Neutrophils_PERCENT", params.get("Neutrophils"), age=age, gender=gender)
    lymph = label_from_range("Lymphocytes_PERCENT", params.get("Lymphocytes"), age=age, gender=gender)

    # Inflammation: CRP or ESR evidence
    crp = label_from_range("CRP", params.get("CRP"))
    esr = label_from_range("ESR", params.get("ESR"))
    inflammation_present = False
    inflammation_support = []
    if crp and str(crp.get("label","")).startswith("high"):
        inflammation_present = True
        inflammation_support.append("CRP_HIGH")
    if esr and str(esr.get("label","")).startswith("high"):
        inflammation_present = True
        inflammation_support.append("ESR_HIGH")

    # Anemia detection
    anemia_present = False
    anemia_type = None
    support = []
    if hb and (str(hb.get("label","")).startswith("low") or "severe" in str(hb.get("label",""))):
        anemia_present = True
        support.append("Hemoglobin_LOW")
        if mcv and str(mcv.get("label","")).startswith("low"):
            anemia_type = "microcytic"
            support.append("MCV_LOW")
        elif mcv and str(mcv.get("label","")).startswith("high"):
            anemia_type = "macrocytic"
            support.append("MCV_HIGH")
        else:
            anemia_type = "normocytic"
        if rdw and str(rdw.get("label","")).startswith("high"):
            support.append("RDW_HIGH")

    patterns["anemia"] = {"present": anemia_present, "type": anemia_type, "support": support, "severity": hb.get("label") if hb else "unknown", "note": hb.get("note") if hb else ""}

    # Thrombocytopenia
    thromb_present = plate and (str(plate.get("label","")).startswith("low") or "severe" in str(plate.get("label","")))
    is_isolated = thromb_present and not ( (wbc and str(wbc.get("label","")).startswith("low")) or anemia_present )
    thromb_support = ["Platelets_LOW"] if thromb_present else []
    patterns["thrombocytopenia"] = {"present": bool(thromb_present), "isolated": bool(is_isolated), "severity": plate.get("label") if plate else "unknown", "support": thromb_support}

    # Neutrophilia / lymphocytosis (infection indicators)
    patterns["neutrophilia"] = {"present": bool(neut and str(neut.get("label","")).startswith("high")), "severity": neut.get("label") if neut else "unknown", "support": ["Neutrophils_HIGH"] if neut and str(neut.get("label","")).startswith("high") else []}
    patterns["lymphocytosis"] = {"present": bool(lymph and str(lymph.get("label","")).startswith("high")), "severity": lymph.get("label") if lymph else "unknown", "support": ["Lymphocytes_HIGH"] if lymph and str(lymph.get("label","")).startswith("high") else []}

    # inflammation
    patterns["inflammation"] = {"present": inflammation_present, "support": inflammation_support, "severity": crp.get("label") if crp else (esr.get("label") if esr else "unknown")}

    # Dyslipidemia: rely on cholesterol and HDL/TG
    tc_label = label_from_range("Total_Cholesterol", params.get("Total_Cholesterol"))
    ldl_label = label_from_range("LDL", params.get("LDL"))
    tg_label = label_from_range("Triglycerides", params.get("Triglycerides"))
    dyslipid = False
    dys_support = []
    if (tc_label and str(tc_label.get("label","")).startswith("high")) or (ldl_label and str(ldl_label.get("label","")).startswith("high")) or (tg_label and str(tg_label.get("label","")).startswith("high")):
        dyslipid = True
        if tc_label and str(tc_label.get("label","")).startswith("high"):
            dys_support.append("Total_Cholesterol_HIGH")
        if ldl_label and str(ldl_label.get("label","")).startswith("high"):
            dys_support.append("LDL_HIGH")
        if tg_label and str(tg_label.get("label","")).startswith("high"):
            dys_support.append("Triglycerides_HIGH")
    patterns["dyslipidemia"] = {"present": dyslipid, "support": dys_support, "details": {"tc": tc_label, "ldl": ldl_label, "tg": tg_label}}

    # Metabolic syndrome signals: high TG, low HDL, high fasting glucose or HbA1c
    glucose = params.get("Glucose_Fasting")
    hba1c = params.get("HbA1c")
    metabolic_signs = False
    metab_support = []
    hdl_lbl = label_from_range("HDL", params.get("HDL"))
    if tg_label and str(tg_label.get("label","")).startswith("high") or (hdl_lbl and str(hdl_lbl.get("label","")).startswith("low")):
        metabolic_signs = True
        metab_support.append("TG_HIGH_or_HDL_LOW")
    try:
        if (hba1c is not None and float(hba1c) >= 5.7) or (glucose is not None and float(glucose) >= 100):
            metabolic_signs = True
            metab_support.append("Glucose_pre-diabetes_or_hyperglycemia")
    except Exception:
        pass
    patterns["metabolic_syndrome_signals"] = {"present": metabolic_signs, "support": metab_support}

    # normalize severity strings
    for pname, pinfo in patterns.items():
        if isinstance(pinfo, dict):
            sev = pinfo.get("severity", "")
            if isinstance(sev, str):
                pinfo["severity"] = sev.lower()

    return {"patterns": patterns}
