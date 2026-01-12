"""
model2/pipeline/risk_engine.py
Derived metric calculations & heuristic cardiovascular risk estimate.

- LDL (Friedewald): LDL = Total_Cholesterol - HDL - Triglycerides/5  (if triglycerides in mg/dL and < 400)
- Non-HDL = Total_Cholesterol - HDL
- TG/HDL ratio
- Total/HDL ratio

Cardio risk estimate is a conservative, heuristic band (low/medium/high).
This is not a clinical score (do not present as Framingham / ASCVD). For research, plug in validated calculators later.
"""

from typing import Dict, Any, Optional

def compute_derived(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    params: normalized parameter dict from loader
    returns dict of derived values and notes
    """
    out: Dict[str, Any] = {}
    tc = _num(params.get("Total_Cholesterol"))
    hdl = _num(params.get("HDL"))
    tg = _num(params.get("Triglycerides"))
    ldl = _num(params.get("LDL"))

    # compute LDL if missing and triglycerides present
    if ldl is None and tc is not None and hdl is not None and tg is not None:
        try:
            if isinstance(tg, (int, float)) and tg < 400:
                ldl_est = tc - hdl - (tg / 5.0)
                out["LDL_estimated"] = round(ldl_est, 2)
            else:
                out["LDL_estimated"] = None
        except Exception:
            out["LDL_estimated"] = None
    else:
        out["LDL_estimated"] = ldl

    # non-HDL
    if tc is not None and hdl is not None:
        try:
            out["Non_HDL"] = round(tc - hdl, 2)
        except Exception:
            out["Non_HDL"] = None
    else:
        out["Non_HDL"] = None

    # ratios
    if tg is not None and hdl is not None and hdl != 0:
        try:
            out["TG_to_HDL_ratio"] = round(tg / hdl, 2)
        except Exception:
            out["TG_to_HDL_ratio"] = None
    else:
        out["TG_to_HDL_ratio"] = None

    if tc is not None and hdl is not None and hdl != 0:
        try:
            out["Total_to_HDL_ratio"] = round(tc / hdl, 2)
        except Exception:
            out["Total_to_HDL_ratio"] = None
    else:
        out["Total_to_HDL_ratio"] = None

    return out

def _num(x: Optional[object]) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def cardio_risk_band(params: Dict[str, Any], derived: Dict[str, Any], themes: Optional[list] = None) -> Dict[str, Any]:
    """
    Heuristic risk band (LOW, MODERATE, HIGH).
    If themes do not indicate lipid/metabolic relevance, return {'applicable': False}.
    """
    # determine if cardio is relevant via themes (if provided)
    relevant_themes = {"lipid_dysregulation", "metabolic_sign", "glycemic_instability"}
    # If themes are missing OR irrelevant → cardio not applicable
    if not themes:
        return {
            "applicable": False,
            "score": None,
            "band": "NOT_APPLICABLE",
            "notes": "Cardiovascular estimate not applicable (no dominant lipid or metabolic signal)."
        }

    top_themes = {t.get("theme") for t in themes if isinstance(t, dict)}
    if not top_themes.intersection(relevant_themes):
        return {
            "applicable": False,
            "score": None,
            "band": "NOT_APPLICABLE",
            "notes": "Cardiovascular estimate not applicable (no dominant lipid or metabolic signal)."
        }

    # existing computation follows (unchanged)
    score = 0.0
    age = params.get("age")
    try:
        age = int(age) if age is not None else None
    except Exception:
        age = None

    ldl = derived.get("LDL_estimated") or _num(params.get("LDL"))
    hdl = _num(params.get("HDL"))
    tg_hdl = derived.get("TG_to_HDL_ratio")
    glucose = _num(params.get("Glucose_Fasting"))
    a1c = _num(params.get("HbA1c"))

    # Age contribution
    if age:
        if age >= 65:
            score += 1.5
        elif age >= 55:
            score += 1.0
        elif age >= 45:
            score += 0.5

    # LDL
    if ldl:
        if ldl >= 190:
            score += 3.0
        elif ldl >= 160:
            score += 2.0
        elif ldl >= 130:
            score += 1.0

    # HDL protective
    if hdl:
        if hdl >= 60:
            score -= 1.0
        elif hdl < 40:
            score += 1.0

    # TG/HDL
    if tg_hdl:
        if tg_hdl >= 4:
            score += 1.5
        elif tg_hdl >= 2:
            score += 0.7

    # glucose / a1c
    if a1c and a1c >= 6.5:
        score += 2.0
    elif glucose and glucose >= 126:
        score += 1.5
    elif a1c and a1c >= 5.7:
        score += 0.5

    # map to band
    if score >= 4.0:
        band = "HIGH"
    elif score >= 1.5:
        band = "MODERATE"
    else:
        band = "LOW"

    return {"applicable": True, "score": round(score, 2), "band": band, "notes": "Heuristic composite score; not a validated clinical risk tool."}
