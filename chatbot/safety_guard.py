import re

# ==========================================================
# USER INTENT SAFETY (INPUT)
# ==========================================================

USER_HARD_REFUSE_PATTERNS = [
    r"\bdiagnos(e|is|ing)\b",
    r"\bwhat (medicine|medication|drug)s?\b",
    r"\bprescrib(e|ing)?\b",
    r"\bshould I take\b",
    r"\bstop\b (medicine|medication|drug)",
    r"\bemergency\b",
    r"\bimmediate\b (treatment|surgery|hospital)",
]

USER_COMPILED = [re.compile(p, re.I) for p in USER_HARD_REFUSE_PATTERNS]


def check_and_refuse(user_text: str):
    """
    USER input intent check.
    Returns:
      - "HARD_REFUSE" if blocked
      - None otherwise
    """
    if not user_text:
        return None

    for pat in USER_COMPILED:
        if pat.search(user_text):
            return "HARD_REFUSE"

    return None


def refuse_response() -> str:
    return (
        "I can’t provide medical diagnoses or prescribe treatments. "
        "I can help explain the report, discuss risk levels, and suggest "
        "general lifestyle or dietary guidance based on the findings. "
        "Please consult a qualified healthcare professional for decisions."
    )


# ==========================================================
# LLM OUTPUT SAFETY (POST-GENERATION)
# ==========================================================

LLM_HARD_REFUSE_PATTERNS = [
    r"\byou have\b",
    r"\bthis confirms\b",
    r"\bdiagnos(e|is)\b",
    r"\bprescrib(e|ed|ing)?\b",
    r"\btake\b \d+\s?(mg|ml)",
    r"\bstart\b (medication|drug)",
    r"\bstop\b (medication|drug)",
    r"\bdosage\b",
    r"\btablet\b",
    r"\bcapsule\b",
]

LLM_COMPILED = [re.compile(p, re.I) for p in LLM_HARD_REFUSE_PATTERNS]


def check_llm_output_and_refuse(text: str) -> bool:
    """
    LLM output safety check.
    Returns True ONLY for real medical violations.
    Diet, habits, reassurance are allowed.
    """
    if not text:
        return False

    for pat in LLM_COMPILED:
        if pat.search(text):
            return True

    return False
