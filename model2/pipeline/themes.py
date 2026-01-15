# model2/pipeline/themes.py
"""
Create ranked themes from signals. Themes represent what the report is mainly about.
Produces a list sorted by strength. Each theme includes evidence and source patterns.
"""
from typing import Dict, Any, List
import math

# map signals -> theme names (keeps stable canonical names)
_SIGNAL_TO_THEME = {
    "platelet_suppression": "platelet_suppression",
    "erythrocyte_abnormality": "erythrocyte_abnormality",
    "lipid_dysregulation": "lipid_dysregulation",
    "systemic_inflammation": "systemic_inflammation",
    "renal_stress": "renal_stress",
    "glycemic_instability": "glycemic_instability",
    "metabolic_sign": "metabolic_sign"
}

def build_themes(signals: Dict[str, float], top_k: int = 3, min_strength: float = 0.30) -> List[Dict[str, Any]]:
    """
    Convert signals dict -> top themes list.
    Only includes themes above min_strength; returns at most top_k themes.
    Each theme:
      { "theme": <str>, "strength": <0-1>, "evidence": [signal names], "source_signals": {signal:strength} }
    """
    if not isinstance(signals, dict):
        return []

    # Build candidate list
    candidates = []
    for sname, svalue in signals.items():
        if svalue is None:
            continue
        sv = float(svalue)
        if sv >= min_strength:
            theme_name = _SIGNAL_TO_THEME.get(sname, sname)
            candidates.append((theme_name, sv, sname))

    if not candidates:
        return []

    # sort by strength desc
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:top_k]
    max_v = top[0][1]

    filtered = []
    for tname, strength, sname in top:
        if strength >= max_v * 0.6:
            filtered.append((tname, strength, sname))


    # take top_k and normalize strengths relative to top
    top = candidates[:top_k]
    max_v = top[0][1] if top else 1.0
    themes = []
    for tname, strength, sname in top:
        norm_strength = float(round(strength / max_v, 3)) if max_v > 0 else float(round(strength,3))
        themes.append({
            "theme": tname,
            "strength": float(round(strength,3)),
            "relative_strength": norm_strength,
            "evidence": [sname],
            "source_signals": {sname: float(round(strength,3))}
        })
    return themes
