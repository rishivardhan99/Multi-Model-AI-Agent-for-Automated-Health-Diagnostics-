# model2/pipeline/priors.py
"""
Reference ranges and conditional priors.

Priors are small, conditional weights and include the 'requires' key to specify
which themes or signals must be present for the prior to apply.
"""
from typing import Dict, Tuple, Optional, Any

REFERENCE_RANGES = {
    "Hemoglobin": {"male": (13.5, 17.5, "g/dL"), "female": (12.0, 15.5, "g/dL"), "default": (12.0, 17.5, "g/dL")},
    "Hematocrit": {"male": (41, 53, "%"), "female": (36, 46, "%"), "default": (36, 53, "%")},
    "WBC": {"default": (4.0, 11.0, "x10^3/uL")},
    "Platelets": {"default": (150, 450, "x10^3/uL")},
    "MCV": {"default": (80, 100, "fL")},
    "MCH": {"default": (27, 33, "pg")},
    "MCHC": {"default": (32, 36, "g/dL")},
    "RDW": {"default": (11.5, 14.5, "%")},
    "Total_Cholesterol": {"default": (0, 200, "mg/dL")},
    "HDL": {"default": (40, 60, "mg/dL")},
    "Triglycerides": {"default": (0, 150, "mg/dL")},
    "LDL": {"default": (0, 100, "mg/dL")},
    "Urea_BUN": {"default": (7, 20, "mg/dL")},
    "Creatinine": {"male": (0.74, 1.35, "mg/dL"), "female": (0.59, 1.04, "mg/dL"), "default": (0.6, 1.35, "mg/dL")},
    "ALT": {"default": (7, 56, "U/L")},
    "AST": {"default": (10, 40, "U/L")},
    "CRP": {"default": (0, 5, "mg/L")},
    "HbA1c": {"default": (4.0, 5.6, "%")},
    "Glucose_Fasting": {"default": (70, 100, "mg/dL")},
    "Neutrophils_PERCENT": {"default": (40, 75, "%")},
    "Lymphocytes_PERCENT": {"default": (20, 45, "%")},
    "Monocytes_PERCENT": {"default": (2, 8, "%")},
    "Eosinophils_PERCENT": {"default": (0, 6, "%")},
    "Basophils_PERCENT": {"default": (0, 2, "%")},
}

# Conditional priors: cause -> { base: float, requires: [themes/signals], description: str }
BASE_PRIORS = {
    "Iron_Deficiency": {"base": 0.05, "requires": ["erythrocyte_abnormality"], "description": "Iron deficiency prior applies when RBC/MCV pattern is present"},
    "Vitamin_B12_Deficiency": {"base": 0.02, "requires": ["erythrocyte_abnormality"], "description": "Macrocytic signals increase this prior"},
    "ITP": {"base": 0.01, "requires": ["platelet_suppression"], "description": "Isolated thrombocytopenia increases ITP prior"},
    "Viral_Infection": {"base": 0.02, "requires": ["systemic_inflammation", "platelet_suppression"], "description": "Viral infections may cause thrombocytopenia/inflammation"},
    "Hypersplenism": {"base": 0.005, "requires": ["platelet_suppression"], "description": "Splenic causes for low platelets"},
    "Metabolic_Syndrome": {"base": 0.03, "requires": ["metabolic_sign", "lipid_dysregulation"], "description": "Metabolic prior requires metabolic/lipid signals"},
    "Dyslipidemia": {"base": 0.04, "requires": ["lipid_dysregulation"], "description": "Lipid prior requires lipid dysregulation signal"},
    "Kidney_Disease": {"base": 0.02, "requires": ["renal_stress"], "description": "Renal prior needs renal stress signal"},
    "Inflammation": {"base": 0.06, "requires": ["systemic_inflammation"], "description": "Inflammation prior requires CRP/ESR elevation"},
}

def get_reference_range(param: str, age: Optional[int]=None, gender: Optional[str]=None):
    info = REFERENCE_RANGES.get(param)
    if not info:
        return None
    if gender:
        g = gender.lower()
        if g.startswith("m") and "male" in info:
            return info["male"]
        if g.startswith("f") and "female" in info:
            return info["female"]
    return info.get("default") if isinstance(info, dict) else info

def get_prior(cause: str, themes: Optional[list] = None) -> float:
    """
    Return prior weight for a cause (0.0 if unknown or preconditions not met).
    `themes` is the list returned from themes.build_themes (list of dicts).
    """
    if cause not in BASE_PRIORS:
        return 0.0
    entry = BASE_PRIORS[cause]
    base = float(entry.get("base", 0.0))
    requires = entry.get("requires", []) or []
    if not requires:
        return base
    if not themes:
        return 0.0
    # simple check: if any required theme in detected themes, allow prior
    detected = {t.get("theme") for t in (themes or [])}
    for r in requires:
        # allow match if requirement is present among detected themes
        for t in themes or []:
            if t.get("theme") == r and t.get("strength", 0) >= 0.6:
                return base

    # If no direct match but signal present (fallback), return reduced prior
    # e.g., if theme detection missed but a signal stronger than 0.6 exists, allow small prior
    theme_map = {t.get("theme"): t.get("strength", 0.0) for t in (themes or [])}
    return 0.0
