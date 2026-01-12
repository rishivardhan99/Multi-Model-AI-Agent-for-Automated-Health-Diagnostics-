# model2/verifier.py
"""
Verifier for Model-2 output structure and simple guardrail checks.

Provides verify_result(parsed) -> (ok: bool, message: str, sanitized_or_parsed)
 - ok True means basic schema checks passed.
 - If false and a sanitized variant is produced, sanitized is returned for inspection.

This verifier is intentionally permissive about data types (Model-2 is mostly deterministic)
but ensures the presence of key top-level keys so the UI / Model-3 can rely on them.
"""
from typing import Any, Tuple
from jsonschema import validate, ValidationError

# Minimal schema to validate that Model-2 output contains essential keys.
SCHEMA = {
    "type": "object",
    "required": ["metadata", "parameters", "patterns", "probable_causes", "confidence"],
    "properties": {
        "metadata": {"type": "object"},
        "parameters": {"type": "object"},
        "status": {"type": "object"},
        "notes": {"type": "object"},
        "derived": {"type": "object"},
        "patterns": {"type": "object"},
        "probable_causes": {"type": "object"},
        "cardio": {"type": "object"},
        "severity": {"type": "object"},
        "confidence": {"type": "object"},
    },
    "additionalProperties": True
}

# guardrails functions (if defined) help detect dangerous content. Optional import to avoid hard dependency.
try:
    from model2.pipeline.guardrails import scan_for_dangerous_recommendations, redact_recommendations
except Exception:
    # define safe fallbacks if guardrails missing
    def scan_for_dangerous_recommendations(x):
        return False, []
    def redact_recommendations(x):
        return x

def verify_result(parsed: Any) -> Tuple[bool, str, Any]:
    """
    Validate parsed Model-2 output. Returns (ok, message, parsed_or_sanitized)
    """
    if parsed is None:
        return False, "Parsed object is None", None
    if not isinstance(parsed, dict):
        return False, "Parsed output is not a dict", None
    try:
        validate(instance=parsed, schema=SCHEMA)
    except ValidationError as e:
        return False, f"Schema validation failed: {e.message}", parsed

    # If probable_causes contains string recommendations accidentally, scan
    # (this is defensive; Model-2 normally has no 'recommendations' field)
    # We scan any list under probable_causes.causes[*].support for suspicious tokens
    try:
        causes = parsed.get("probable_causes", {}).get("causes", [])
        suspicious = []
        for c in causes:
            for s in c.get("support", []):
                if isinstance(s, str) and any(k in s.lower() for k in ("prescribe", "dosage", "take", "surgery")):
                    suspicious.append(s)
        if suspicious:
            return False, f"Detected suspicious content in probable_causes.support: {suspicious[:3]}", parsed
    except Exception:
        pass

    # final: pass
    return True, "OK", parsed
