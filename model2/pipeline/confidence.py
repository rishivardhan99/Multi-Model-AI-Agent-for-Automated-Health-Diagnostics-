# model2/pipeline/confidence.py
from typing import Dict, Any, List, Optional

# ---------------------------
# Low-level scorers (kept/compatible with your earlier logic)
# ---------------------------
def _parameter_presence_score(params: Dict[str, Any]) -> float:
    IGNORED = {"age", "gender", "patient_id", "filename", "report_date"}
    total = 0
    present = 0
    if not isinstance(params, dict):
        return 0.0
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
    if not isinstance(patterns, dict):
        return 0.0
    p = patterns.get("patterns", {}) if isinstance(patterns.get("patterns", {}), dict) else {}
    if not p:
        return 0.0

    weight = 0.0
    count = 0
    for info in p.values():
        if not isinstance(info, dict):
            continue
        if info.get("present"):
            count += 1
            sev = str(info.get("severity", "") or "").lower()
            if "very_severe" in sev or "very severe" in sev or "severe" in sev:
                weight += 1.0
            else:
                weight += 0.55
    if count == 0:
        return 0.0
    return round(min(1.0, weight / count), 3)

def _kg_top_score(probable_causes: Dict[str, Any]) -> float:
    if not isinstance(probable_causes, dict):
        return 0.0
    causes = probable_causes.get("causes", [])
    if not isinstance(causes, list) or not causes:
        return 0.0
    top = causes[0] if isinstance(causes[0], dict) else {"score": 0.0}
    try:
        return float(top.get("score", 0.0))
    except Exception:
        return 0.0

def _theme_dominance_score(themes: Optional[List[Dict[str, Any]]]) -> float:
    if not themes or not isinstance(themes, list):
        return 0.0
    strengths = [float(t.get("strength", 0.0) or 0.0) for t in themes if isinstance(t, dict)]
    strengths = sorted([s for s in strengths if s is not None], reverse=True)
    if not strengths:
        return 0.0
    top = strengths[0]
    if len(strengths) >= 3 and strengths[2] > 0.3:
        return round(min(1.0, top * 0.75), 3)
    return round(min(1.0, top), 3)

def _detect_kg_contradiction(patterns: Dict[str, Any], probable_causes: Dict[str, Any]) -> float:
    if not isinstance(probable_causes, dict):
        return 1.0
    causes = probable_causes.get("causes", [])
    if not isinstance(causes, list) or not causes:
        return 1.0
    top = causes[0]
    top_cause = ""
    if isinstance(top, dict):
        top_cause = str(top.get("cause", "")).lower()
    elif isinstance(top, str):
        top_cause = top.lower()

    infl_present = False
    if isinstance(patterns, dict):
        patt = patterns.get("patterns", {}) if isinstance(patterns.get("patterns", {}), dict) else {}
        infl = patt.get("inflammation", {}) if isinstance(patt.get("inflammation", {}), dict) else {}
        infl_present = bool(infl.get("present", False))

    if any(k in top_cause for k in ("infection", "viral", "bacterial")) and not infl_present:
        return 0.35
    return 1.0

# ---------------------------
# Combination & explanation
# ---------------------------
def _compute_weighted_score(components: Dict[str, float]) -> float:
    weights = {"presence": 0.30, "pattern": 0.30, "kg": 0.20, "themes": 0.20}
    score = 0.0
    for k, w in weights.items():
        v = float(components.get(k, 0.0) or 0.0)
        score += v * w
    return round(max(0.0, min(1.0, score)), 3)

def _build_confidence_explanation(
    presence: float,
    pattern: float,
    patterns_obj: Dict[str, Any],
    kg_adjusted: float,
    missing_params: List[str],
    probable_causes: Dict[str, Any],
    themes_score: float
) -> str:
    reasons = []
    active_patterns = []
    isolated_patterns = []
    if isinstance(patterns_obj, dict):
        for name, info in (patterns_obj.get("patterns", {}) or {}).items():
            if not isinstance(info, dict):
                continue
            if bool(info.get("present", False)):
                active_patterns.append(name)
                if bool(info.get("isolated", False)):
                    isolated_patterns.append(name)

    num_patterns = len(active_patterns)

    if num_patterns == 0 and presence >= 0.9:
        return "High confidence due to normal laboratory values and absence of pathological patterns."

    if num_patterns == 1 and len(isolated_patterns) == 1 and pattern >= 0.5 and kg_adjusted >= 0.6:
        p = isolated_patterns[0].replace("_", " ")
        return (
            f"High confidence in an isolated {p} finding, supported by the corresponding laboratory value. "
            "No evidence of systemic involvement was detected."
        )

    if num_patterns >= 2 and pattern >= 0.6:
        return (
            "Moderate to high confidence based on multiple corroborating laboratory patterns and consistent signal aggregation. "
            "Clinical correlation is advised."
        )

    if pattern < 0.6 and (kg_adjusted > 0.0 or themes_score > 0.0):
        return (
            "Moderate confidence because detected patterns are not strongly corroborated, "
            "but contextual signals or KG associations provide supplementary evidence."
        )

    if pattern < 0.4 and kg_adjusted < 0.3 and presence < 0.6:
        if missing_params:
            return (
                "Low confidence due to missing or incomplete laboratory parameters; repeat testing or clinician review recommended."
            )
        return "Low confidence due to limited corroborating data."

    return (
        "Moderate confidence based on available laboratory data and deterministic reasoning, "
        "with some uncertainty due to limited corroboration."
    )

def compute_confidence(
    params: Dict[str, Any],
    patterns: Dict[str, Any],
    probable_causes: Dict[str, Any],
    missing_params: List[str],
    themes: Optional[List[Dict[str, Any]]] = None,
    quality_flags: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Public function used by Model-2 runner.
    Keeps backwards compatibility: returns {'score': <report_confidence>, ...}
    and adds 'analysis_score' and 'report_confidence' explicitly.
    """
    presence = _parameter_presence_score(params)
    pattern = _pattern_strength(patterns)
    kg_raw = _kg_top_score(probable_causes)
    theme_score = _theme_dominance_score(themes)

    kg_penalty = _detect_kg_contradiction(patterns, probable_causes)
    kg_adjusted = round(min(1.0, kg_raw * kg_penalty), 3)

    comps = {"presence": round(presence, 3), "pattern": round(pattern, 3), "kg": kg_adjusted, "themes": round(theme_score, 3)}
    analysis_score = _compute_weighted_score(comps)

    # Report-level confidence rules (account for data quality)
    low_quality = False
    quality_reason = ""
    if isinstance(quality_flags, dict):
        low_quality = bool(quality_flags.get("low_data_quality", False))
        quality_reason = quality_flags.get("reason", "")

    # Base report confidence = analysis_score unless special override
    report_confidence = float(analysis_score)

    # If there are NO abnormal patterns but parameter coverage is high -> high report confidence
    no_abnormal_patterns = (isinstance(patterns, dict) and not any([v.get("present") for v in (patterns.get("patterns", {}) or {}).values()]))

    if no_abnormal_patterns and presence >= 0.9:
        report_confidence = 0.85

    # If low quality data detected -> apply conservative penalty
    if low_quality:
        # set a floor but reduce confidence strongly
        report_confidence = round(max(0.15, min(0.4, analysis_score * 0.4)), 3)

    # If plenty missing params -> lower confidence
    try:
        if missing_params and len(missing_params) > 0:
            # scale down by fraction of missing params; if many missing, strong drop
            missing_frac = min(1.0, len(missing_params) / max(1, (len(params) + len(missing_params))))
            # reduce proportional to missing_frac
            report_confidence = round(report_confidence * (1.0 - 0.5 * missing_frac), 3)
    except Exception:
        pass

    # Clamp
    report_confidence = round(max(0.0, min(1.0, report_confidence)), 3)

    explanation = _build_confidence_explanation(
        presence=presence,
        pattern=pattern,
        patterns_obj=patterns or {},
        kg_adjusted=kg_adjusted,
        missing_params=missing_params or [],
        probable_causes=probable_causes or {},
        themes_score=theme_score
    )

    missing_fraction = 0.0
    try:
        total = len(params) + len(missing_params)
        if total > 0:
            missing_fraction = round(len(missing_params) / total, 3)
    except Exception:
        pass

    return {
    "score": report_confidence,                  # legacy (DO NOT display)
    "confidence_band": confidence_band(report_confidence),
    "analysis_score": analysis_score,
    "report_confidence": report_confidence,      # internal use only
    "components": comps,
    "explanation": explanation,
    "quality_reason": quality_reason,
    "missing_param_fraction": missing_fraction

}

def confidence_band(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "Unknown"

    if s >= 0.75:
        return "Moderate–High"
    elif s >= 0.45:
        return "Moderate"
    elif s > 0.0:
        return "Low"
    return "Unknown"
