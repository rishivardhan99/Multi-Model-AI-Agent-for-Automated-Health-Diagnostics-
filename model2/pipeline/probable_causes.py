"""
Combine deterministic observations and KG inference to produce ranked probable causes.

Improvements:
- Penalize causes when key corroborating evidence is absent (negative evidence rules).
- Boost causes when specific pattern_details indicate specific contexts (e.g., isolated thrombocytopenia -> ITP).
- Return 'adjustments' explaining boosts/penalties (useful for UI & model3).
"""
from typing import Dict, Any, List, Optional
from .knowledge_graph import KnowledgeGraph, build_medical_kg
from .priors import BASE_PRIORS

# mapping of cause -> required supporting observations (at least one preferred)
CAUSE_SUPPORT_RULES = {
    "Bacterial_Infection": ["Neutrophils_HIGH", "CRP_HIGH"],
    "Viral_Infection": ["Lymphocytes_HIGH", "CRP_HIGH"],
    "Inflammation": ["CRP_HIGH", "ESR_HIGH"],
    # thrombocytopenia causes: none mandatory, but isolated thrombocytopenia boosts ITP
}

def infer_probable_causes(observations: List[str], pattern_details: Dict[str,Any], priors: Dict[str,float]=None, themes: Optional[List[Dict[str, Any]]]=None) -> Dict[str,Any]:
    """
    observations: list of observation node names e.g., ["Hemoglobin_LOW","MCV_LOW","Platelets_LOW"]
    pattern_details: patterns detected by pattern_engine (used for boosts/penalties)
    priors: base prior weights for causes (optional)
    themes: optional list of theme dicts (from build_themes) to condition priors conservatively
    returns:
      {
        "causes": [
          {"cause": "Iron_Deficiency", "score":0.78, "support": ["MCV_LOW->possible_cause->Iron_Deficiency"], "source":"kg"}
        ],
        "raw_scores": {...},
        "adjustments": {cause: "reason string"}
      }
    """
    kg = build_medical_kg()
    # get base kg scores (max weight per cause from observations)
    kg_scores = kg.infer_causes(observations)

    # initialize combined using evidence accumulation with damping
    combined: Dict[str, float] = {}
    support: Dict[str, List[str]] = {}
    for obs in observations:
        edges = kg.query(obs)
        for e in edges:
            targ = e["target"]
            w = e["weight"]
            existing = combined.get(targ, 0.0)
            combined[targ] = min(1.0, existing + (w * 0.75))
            support.setdefault(targ, []).append(f"{obs}->{e['relation']}->{targ}")

    # incorporate provided KG-only inference (ensure we included any leftover)
    for k, v in kg_scores.items():
        if k not in combined:
            combined[k] = v
            support.setdefault(k, [])

    # incorporate priors (multiplicative but gentle)
    if priors is None:
        priors = BASE_PRIORS

    # theme-aware prior adjustment (conservative)
    theme_boost: Dict[str, float] = {}
    if themes:
        top = themes[0] if len(themes) else None
        if top:
            tname = (top.get("theme") or "").lower()
            if "lipid" in tname:
                theme_boost["Dyslipidemia"] = 0.03
                theme_boost["Metabolic_Syndrome"] = 0.02
            if "renal" in tname or "renal_stress" in tname:
                theme_boost["Kidney_Disease"] = 0.03
            if "platelet" in tname:
                theme_boost["ITP"] = 0.02

    for c in list(combined.keys()):
        prior_obj = priors.get(c, {})
        p = float(prior_obj.get("base", 0.0))

        p = p + float(theme_boost.get(c, 0.0))
        combined[c] = min(1.0, combined[c] * (1.0 + 0.5 * p))  # small prior influence

    # ADJUSTMENTS: apply domain-specific calibration rules (penalize/boost)
    adjustments: Dict[str, str] = {}
    obs_set = set(observations)

    # 1) Penalize infection claims lacking inflammatory support
    for cause in list(combined.keys()):
        reqs = CAUSE_SUPPORT_RULES.get(cause)
        if reqs:
            has_support = any(r in obs_set for r in reqs)
            if not has_support:
                old = combined[cause]
                new = round(old * 0.28, 3)  # strong downweight when no corroborating evidence
                combined[cause] = new
                adjustments[cause] = "reduced (key inflammatory markers missing; weak evidence)"

    # 2) Boost causes when pattern_details indicate specific contexts
    patt = (pattern_details or {}).get("patterns", {})
    # ------------------------------------------------------------------
    # HARD GATE: Do NOT assert etiologies if parent pattern is absent
    # ------------------------------------------------------------------
    anemia_patt = patt.get("anemia", {}) if isinstance(patt.get("anemia", {}), dict) else {}

    if not anemia_patt.get("present", False):
        # CBC shows no anemia → block iron deficiency / B12 deficiency
        for c in ("Iron_Deficiency", "Vitamin_B12_Deficiency"):
            if c in combined:
                combined[c] = round(combined[c] * 0.05, 3)
                adjustments[c] = "suppressed (no anemia pattern present; etiology cannot be inferred)"


    thromb = patt.get("thrombocytopenia", {}) if isinstance(patt.get("thrombocytopenia", {}), dict) else {}
    if thromb.get("present") and thromb.get("isolated"):
        for boost_c in ("ITP", "Viral_Infection"):
            old = combined.get(boost_c, 0.0)
            new = min(1.0, round(old * 1.25 + 0.05, 3))
            combined[boost_c] = new
            adjustments[boost_c] = adjustments.get(boost_c, "") + (" boosted due to isolated thrombocytopenia;")

    # 3) If a cause is suggested only by very weak KG evidence (low weight) and contradicting strong negative markers, mark weak
    for c in list(combined.keys()):
        if combined[c] < 0.15:
            adjustments[c] = adjustments.get(c, "") + " weak_signal"

    # normalize relative to max so scores are comparable
    RISK_CAUSES = {"Cardiovascular_Risk", "Metabolic_Risk"}

    # Cap risk-type causes BEFORE normalization
    for c in combined:
        if c in RISK_CAUSES:
            combined[c] = min(combined[c], 0.85)

    
    mx = max(combined.values()) if combined else 0.0
    # prevent single weak causes from normalizing to 1.0
    if mx < 0.4:
        mx = 1.0

    if mx > 0:
        for k in combined:
            combined[k] = round(combined[k] / mx, 3)

    # prepare sorted list and detailed support entries
    sorted_causes = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    causes_out: List[Dict[str,Any]] = []
    for cause, score in sorted_causes:
        risk_type = "preventive" if cause in {"Cardiovascular_Risk", "Metabolic_Risk"} else "diagnostic"

        # downgrade diagnostic claims when evidence is weak
        if adjustments.get(cause, "").startswith("suppressed") or score < 0.25:
            risk_type = "suggestive"



        causes_out.append({
            "cause": cause,
            "score": score,
            "risk_type": risk_type,
            "support": support.get(cause, [])[:6],
            "adjustment": adjustments.get(cause, ""),
            "source": "kg"
        })


    return {"causes": causes_out, "raw_scores": combined, "adjustments": adjustments}
