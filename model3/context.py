"""
Context utilities for Model-3 pipeline.

Responsibilities:
- Load and sanitize user context
- Interpret context deterministically into narrative modifiers
- Provide deterministic_consult_advice(...) for actionable 'when_to_consult_doctor'
"""
from typing import Any, Dict, Optional, Tuple, List
import json
from pathlib import Path

_DEFAULT_CONTEXT = {
    "age": None,
    "gender": None,
    "lifestyle": {
        "smoking": None,
        "alcohol": None,
        "activity": None
    },
    "medical_notes": None,
    "current_symptoms": None
}


def load_user_context(path_or_obj: Optional[Any]) -> Dict:
    """
    Load user context from:
     - a dict (returned as-is after sanitization)
     - a path-like string / Path pointing to a JSON file
     - inline JSON string (will attempt to parse)
     - None -> returns {}
    """
    if path_or_obj is None:
        return {}
    if isinstance(path_or_obj, dict):
        return _sanitize_context(path_or_obj)
    if isinstance(path_or_obj, str):
        p = Path(path_or_obj)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return _sanitize_context(data)
            except Exception:
                pass
        # attempt parse inline JSON
        try:
            data = json.loads(path_or_obj)
            if isinstance(data, dict):
                return _sanitize_context(data)
        except Exception:
            pass
    return {}


def _sanitize_context(raw: Dict) -> Dict:
    """
    Ensure minimal shape and safe types while preserving provided fields.
    DO NOT introduce keys with None values that would overwrite original meaningful values.
    """
    ctx = {
        "age": None,
        "gender": None,
        "lifestyle": {"smoking": None, "alcohol": None, "activity": None},
        "medical_notes": None,
        "current_symptoms": None
    }
    if not isinstance(raw, dict):
        return ctx

    # Age (safe int)
    ctx["age"] = _safe_int(raw.get("age"))

    # Gender: preserve only if non-empty string. Do NOT set to None if missing.
    gender = raw.get("gender")
    if gender is not None:
        try:
            g = str(gender).strip()
            if g != "":
                ctx["gender"] = g
        except Exception:
            pass

    # Lifestyle keys: accept nested or flat forms; preserve non-empty strings
    lifestyle = raw.get("lifestyle") or raw.get("life_style") or {}
    if not isinstance(lifestyle, dict):
        lifestyle = {}
    ctx["lifestyle"] = {
        "smoking": _safe_str(lifestyle.get("smoking") or raw.get("smoking")),
        "alcohol": _safe_str(lifestyle.get("alcohol") or raw.get("alcohol")),
        "activity": _safe_str(lifestyle.get("activity") or raw.get("activity"))
    }

    ctx["medical_notes"] = _safe_str(raw.get("medical_notes") or raw.get("notes"))
    ctx["current_symptoms"] = _safe_str(raw.get("current_symptoms") or raw.get("symptoms"))

    return ctx


def _safe_int(v):
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _safe_str(v):
    if v is None:
        return None
    try:
        s = str(v).strip()
        return s if s != "" else None
    except Exception:
        return None


# ----- deterministic interpretation rules -----
def interpret_context(ctx: Dict) -> Dict:
    """
    Convert sanitized user context into deterministic narrative modifiers.

    Returns:
      - context_summary: short 1-3 line summary suitable for prompts.
      - urgency_level: one of ("low","routine","moderate","elevated")
      - lifestyle_focus: list of strings describing focus
      - notes: explanatory note string
    """
    if not isinstance(ctx, dict) or not ctx:
        return {
            "context_summary": "",
            "urgency_level": "routine",
            "lifestyle_focus": [],
            "notes": ""
        }

    age = ctx.get("age")
    gender = ctx.get("gender")
    lifestyle = ctx.get("lifestyle", {}) or {}
    symptoms = (ctx.get("current_symptoms") or "")
    medical_notes = ctx.get("medical_notes")

    # determine urgency by age + symptoms heuristics (deterministic)
    urgency = "routine"
    if symptoms:
        # symptoms should raise baseline urgency; some keywords increase to elevated
        urgency = "moderate"
        s_low = symptoms.lower()
        if any(k in s_low for k in ("fever", "bleeding", "bruis", "shortness", "breath", "chest", "faint", "collapse")):
            urgency = "elevated"
    else:
        if age is not None:
            try:
                if age >= 65:
                    urgency = "elevated"
                elif age >= 50:
                    urgency = "moderate"
                elif age < 30:
                    urgency = "low"
            except Exception:
                urgency = "routine"

    # lifestyle focus bullets
    focus = []
    smoking = (lifestyle.get("smoking") or "").lower()
    alcohol = (lifestyle.get("alcohol") or "").lower()
    activity = (lifestyle.get("activity") or "").lower()

    if smoking and any(tok in smoking for tok in ("no", "never", "none", "n")):
        focus.append("maintain non-smoking habit")
    elif smoking and smoking not in ("no","never","n","none"):
        focus.append("discuss smoking cessation if applicable")

    if activity and any(tok in activity for tok in ("moderate", "regular", "yes")):
        focus.append("continue regular physical activity")
    elif activity and activity not in ("moderate","regular","yes","no","none"):
        focus.append("consider increasing habitual physical activity")

    if alcohol and any(tok in alcohol for tok in ("occasional","no","rare")):
        focus.append("keep alcohol intake moderate")
    elif alcohol and alcohol not in ("no","none","occasional","rare"):
        focus.append("discuss alcohol reduction if relevant")

    # build context summary (1-3 short parts)
    parts = []
    if age is not None:
        parts.append(f"Age: {age}")
    if gender:
        parts.append(f"Gender: {gender}")
    if symptoms:
        parts.append(f"Symptoms: {symptoms}")
    else:
        parts.append("Asymptomatic at time of report")
    if medical_notes:
        parts.append(f"Medical history note: {medical_notes}")

    context_summary = "; ".join(parts[:4])

    notes = ""
    if urgency == "low":
        notes = "Younger asymptomatic individual; baseline urgency is low."
    elif urgency == "elevated":
        notes = "Older age or red-flag symptoms increase baseline urgency for follow-up."
    elif urgency == "moderate":
        notes = "Some factors suggest moderate urgency; interpret with clinical correlation."

    return {
        "context_summary": context_summary,
        "urgency_level": urgency,
        "lifestyle_focus": focus,
        "notes": notes
    }


# ----- deterministic consult rules -----
def deterministic_consult_advice(model2_compact: Dict, user_ctx: Dict, interpreted_ctx: Dict) -> str:
    """
    Produce a conservative 'when_to_consult_doctor' string based on:
      - model2_compact: compacted Model-2 facts (patterns, severity, cardio, confidence)
      - user_ctx: sanitized user context (age, symptoms)
      - interpreted_ctx: output of interpret_context(user_ctx)
    """
    adv = []
    urgency = interpreted_ctx.get("urgency_level", "routine")

    patterns = model2_compact.get("patterns", {}) or {}
    severity = model2_compact.get("severity", {}) or {}
    cardio = model2_compact.get("cardio", {}) or {}
    conf = model2_compact.get("confidence", {}) or {}

    # Severe numeric flags -> immediate
    if isinstance(severity, dict):
        for p, info in severity.items():
            lab = info.get("label") if isinstance(info, dict) else None
            if isinstance(lab, str) and ("very_severe" in lab or "severe" in lab):
                adv.append(f"Immediate clinical review recommended due to severe abnormality in {p}.")

    # Platelet-specific safety check
    plate_info = severity.get("Platelets") if isinstance(severity.get("Platelets"), dict) else None
    try:
        if plate_info and isinstance(plate_info, dict):
            if plate_info.get("label") and ("very_severe" in plate_info.get("label") or "severe" in plate_info.get("label")):
                adv.append("Platelet count is severely low — seek urgent clinical evaluation (possible bleeding risk).")
    except Exception:
        pass

    # Infection flags
    if isinstance(patterns, dict):
        inf_pat = patterns.get("neutrophilia") or {}
        if inf_pat.get("present") and (inf_pat.get("severity") and "severe" in str(inf_pat.get("severity"))):
            adv.append("Marked neutrophilia detected — consider urgent evaluation for possible bacterial infection.")

    # Cardio risk red flags
    if isinstance(cardio, dict):
        band = str(cardio.get("band", "")).upper()
        score = 0.0
        try:
            score = float(cardio.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if band == "HIGH" or score >= 4.0:
            adv.append("Cardiovascular risk estimate is high — arrange expedited clinician review and risk factor management.")

    # User-reported symptoms escalate
    if user_ctx and user_ctx.get("current_symptoms"):
        adv.append("Reported symptoms present — consult clinician sooner than routine for correlation with symptoms.")

    # Age-based fallback
    age = user_ctx.get("age")
    if age and isinstance(age, int) and age >= 65:
        adv.append("Age > 65: consider earlier follow-up due to higher baseline risk.")

    # Low confidence -> suggest repeat/review
    conf_score = conf.get("score") if isinstance(conf, dict) else None
    try:
        if conf_score is not None and float(conf_score) < 0.4:
            adv.append("Low confidence due to missing or inconsistent parameters; consider repeat testing or clinician review.")
    except Exception:
        pass

    # Build output
    if adv:
        full = "Seek clinical review in these circumstances:\n"
        for line in adv:
            full += f"- {line}\n"
        full += "If none of the above apply, discuss results with your clinician during routine follow-up or earlier if new symptoms develop."
        return full

    if urgency == "elevated":
        return "Consult a clinician promptly (within 48–72 hours) for review due to age/clinical context or if any new symptoms develop."
    if urgency == "moderate":
        return "Consider clinician review within 1–2 weeks, or sooner if you develop symptoms (e.g., fever, unexplained bleeding, new shortness of breath)."
    if urgency == "low":
        return "No immediate clinical action required based on the lab values alone; discuss during your next routine visit or earlier if new symptoms develop."

    return "Discuss these results with a clinician to confirm findings; seek earlier review if symptoms develop or values change."
