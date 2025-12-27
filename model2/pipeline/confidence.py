"""
Combine several signals into an overall confidence for the model2 output.

Improvements:
- If top KG candidate lacks corroborating pattern/inflammation evidence, reduce KG contribution.
- Provide clearer explanation text and component breakdown.
"""
from typing import Dict, Any, List

def compute_confidence(
    params: Dict[str, Any],
    patterns: Dict[str, Any],
    probable_causes: Dict[str, Any],
    missing_params: List[str]
) -> Dict[str, Any]:

    presence_score = _parameter_presence_score(params)
    pattern_score = _pattern_strength(patterns)
    kg_score = _kg_top_score(probable_causes)

    # detect contradictory situation: top cause is infection but no inflammation pattern
    kg_penalty = _detect_kg_contradiction(patterns, probable_causes)

    # weighted sum (preserve original weights but apply kg_penalty)
    score = round(0.4 * presence_score + 0.4 * pattern_score + 0.2 * kg_score * kg_penalty, 3)

    return {
        "score": score,
        "components": {
            "presence": presence_score,
            "pattern": pattern_score,
            "kg": round(kg_score * kg_penalty, 3)
        },
        "explanation": build_confidence_explanation(
            presence_score,
            pattern_score,
            missing_params,
            kg_penalty,
            probable_causes
        )
    }

def _parameter_presence_score(params: Dict[str, Any]) -> float:
    IGNORED = {"age", "gender", "patient_id", "filename", "report_date"}

    total = 0
    present = 0

    for k, v in params.items():
        if k in IGNORED:
            continue
        total += 1
        if v is not None:
            present += 1

    if total == 0:
        return 0.0

    return round(min(1.0, present / total), 3)

def _pattern_strength(patterns: Dict[str, Any]) -> float:
    p = patterns.get("patterns", {}) if isinstance(patterns, dict) else {}
    if not p:
        return 0.0

    weight = 0.0
    count = 0

    for _, info in p.items():
        if info.get("present"):
            count += 1
            sev = str(info.get("severity", "") or "")
            if "very_severe" in sev or "severe" in sev:
                weight += 1.0
            else:
                weight += 0.55

    if count == 0:
        return 0.0

    return round(min(1.0, weight / count), 3)

def _kg_top_score(probable_causes: Dict[str, Any]) -> float:
    causes = probable_causes.get("causes", []) if isinstance(probable_causes, dict) else []
    if not causes:
        return 0.0
    top = causes[0]
    try:
        return float(top.get("score", 0.0))
    except Exception:
        return 0.0

def _detect_kg_contradiction(patterns: Dict[str, Any], probable_causes: Dict[str, Any]) -> float:
    """
    If the top probable cause is an infection but inflammation pattern is absent,
    return penalty multiplier < 1. Otherwise return 1.0.
    """
    causes = probable_causes.get("causes", []) if isinstance(probable_causes, dict) else []
    if not causes:
        return 1.0
    top = causes[0].get("cause", "")
    top_lower = top.lower()
    inflammation_present = False
    patt = patterns.get("patterns", {}) if isinstance(patterns, dict) else {}
    infl = patt.get("inflammation", {}) if isinstance(patt.get("inflammation", {}), dict) else {}
    if infl.get("present"):
        inflammation_present = True

    # if top cause is bacterial or viral and there is no inflammation signal, apply penalty
    if ("bacterial" in top_lower or "infection" in top_lower or "viral" in top_lower) and not inflammation_present:
        return 0.35
    return 1.0

def build_confidence_explanation(
    presence: float,
    pattern: float,
    missing_params: List[str],
    kg_penalty: float,
    probable_causes: Dict[str, Any]
) -> str:

    reasons = []

    if presence < 0.5:
        reasons.append(
            "several relevant laboratory parameters are missing, limiting diagnostic certainty"
        )

    if pattern < 0.7:
        reasons.append(
            "some detected patterns overlap or lack confirmatory markers"
        )

    # KG contradiction note
    if kg_penalty < 1.0:
        top = None
        causes = probable_causes.get("causes", []) if isinstance(probable_causes, dict) else []
        if causes:
            top = causes[0].get("cause")
        if top:
            reasons.append(f"top inferred cause ({top}) lacks corroborating inflammatory markers; its influence was reduced")

    if missing_params:
        reasons.append(
            "missing parameters include: "
            + ", ".join(missing_params[:6])
            + ("..." if len(missing_params) > 6 else "")
        )

    if not reasons:
        return "High confidence due to complete and consistent laboratory evidence."

    return "Moderate confidence because " + "; ".join(reasons) + "."
