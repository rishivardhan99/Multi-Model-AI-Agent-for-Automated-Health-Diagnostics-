# chatbot/safety_guard.py
import re

REFUSAL_PATTERNS = [
    r"\bdiagnos(e|is|ing)\b",
    r"\bwhat (medication|medicine|drug)s?\b",
    r"\bprescrib(e|ing)?\b",
    r"\bshould I take\b",
    r"\bshould I stop\b",
    r"\bemergency\b",
    r"\bimmediate\b (treatment|surgery|hospital)"
]

compiled = [re.compile(p, flags=re.I) for p in REFUSAL_PATTERNS]

def check_and_refuse(user_question: str) -> bool:
    if not user_question:
        return False
    for c in compiled:
        if c.search(user_question):
            return True
    return False

def refuse_response() -> str:
    return (
        "I cannot provide diagnoses or medical prescriptions. "
        "This report should be reviewed by a qualified healthcare professional. "
        "If this is an emergency, contact emergency services immediately."
    )
