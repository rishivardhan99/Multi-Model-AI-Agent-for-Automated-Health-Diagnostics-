"""
Reference ranges and priors.
"""
from typing import Dict, Tuple, Optional

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

BASE_PRIORS = {
    "Iron_Deficiency": 0.05,
    "Vitamin_B12_Deficiency": 0.02,
    "ITP": 0.01,
    "Viral_Infection": 0.02,
    "Hypersplenism": 0.005,
    "Metabolic_Syndrome": 0.03,
    "Dyslipidemia": 0.04,
    "Kidney_Disease": 0.02,
    "Inflammation": 0.06,
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

def get_prior(cause: str) -> float:
    """Return prior weight for a cause (0.0 if unknown)."""
    return float(BASE_PRIORS.get(cause, 0.0))
