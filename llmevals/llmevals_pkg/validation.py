# llmevals_pkg/validation.py
import os
from typing import List, Dict, Any

# Penalty weights (points subtracted from raw overall_score)
DEFAULT_WEIGHTS = {
    # hallucinations are the worst
    ("hallucination", "high"): 20,
    ("hallucination", "medium"): 10,
    ("hallucination", "low"): 5,
    # missing information penalized but less
    ("missing", "high"): 10,
    ("missing", "medium"): 4,
    ("missing", "low"): 2,
}

# threshold for system to consider a validated (non-abstained) decision successful
SYSTEM_PASS_THRESHOLD = float(os.getenv('SYSTEM_PASS_THRESHOLD', '90.0'))

# If any high severity hallucination appears, we consider abstention as required/correct
# (this means the evaluation layer will treat an abstention/rejection as correct behaviour)


def critical_missing_count(issues):
    """
    Count missing issues that represent high-confidence omissions,
    not benign clinical judgment.
    """
    if not issues:
        return 0
    return sum(
        1 for it in issues
        if it.get('type') == 'missing'
        and it.get('severity') == 'high'
    )


def _get_weights() -> Dict:
    # Allows overriding weights with env var if needed (not required)
    return DEFAULT_WEIGHTS


def compute_penalty(issues: List[Dict[str, Any]]) -> float:
    """Sum penalty points for a list of issues."""
    if not issues:
        return 0.0
    weights = _get_weights()
    total = 0.0
    for it in issues:
        t = it.get('type')
        sev = it.get('severity')
        if not t or not sev:
            continue
        total += float(weights.get((t, sev), 0))
    return total


def should_abstain(eval_obj: Dict[str, Any]) -> bool:
    """Decide whether the correct system action is to abstain for this eval.

    Logic (conservative, audit-safe):
    - If eval contains an explicit error -> abstain
    - If any issue with (type='hallucination' and severity='high') -> abstain
    - Optionally: if overall_score is missing -> abstain
    """
    if not isinstance(eval_obj, dict):
        return True
    if eval_obj.get('error'):
        return True
    issues = eval_obj.get('issues') or []
    for it in issues:
        if it.get('type') == 'hallucination' and it.get('severity') == 'high':
            return True
    # if overall_score is missing or invalid, abstain
    try:
        _ = float(eval_obj.get('overall_score'))
    except Exception:
        return True
    return False


def compute_validated_score(raw_score: float, issues: List[Dict[str, Any]]) -> float:
    """Apply penalties to raw_score to produce a validated score in [0,100]."""
    if raw_score is None:
        raw_score = 0.0
    penalty = compute_penalty(issues or [])
    result = float(raw_score) - float(penalty)
    if result < 0:
        result = 0.0
    if result > 100:
        result = 100.0
    return result


def abstention_type(eval_obj: Dict[str, Any]) -> str | None:
    if not isinstance(eval_obj, dict):
        return 'technical'
    if eval_obj.get('error'):
        return 'technical'

    issues = eval_obj.get('issues', []) or []

    # Rule 1: high-risk hallucination → clinical abstention
    for it in issues:
        if it.get('type') == 'hallucination' and it.get('severity') == 'high':
            return 'clinical'

    # Rule 2 (NEW): coverage failure → clinical abstention
    if critical_missing_count(issues) >= 2:
        return 'clinical'


    return None



def assess(eval_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return validation summary with explicit abstention typing.

    clinical_abstention  -> counts as system-correct (safety behavior)
    technical_abstention -> excluded from denominator (infrastructure failure)
    """
    reasons = []
    raw_score = None
    try:
        raw_score = float(eval_obj.get('overall_score')) if eval_obj and 'overall_score' in eval_obj else None
    except Exception:
        raw_score = None

    issues = eval_obj.get('issues') if isinstance(eval_obj, dict) else []
    penalty = compute_penalty(issues)
    validated_score = compute_validated_score(raw_score if raw_score is not None else 0.0, issues)

    abst_type = abstention_type(eval_obj)
    is_abstain = abst_type is not None

    # system pass rules
    if abst_type == 'clinical':
        system_pass = True
        reasons.append('clinical_abstention_due_to_high_risk_output')

    elif abst_type == 'technical':
        system_pass = None  # excluded from denominator
        reasons.append('technical_abstention_llm_or_parse_error')
    else:
        # soft threshold for acceptable aligned reasoning
        SOFT_PASS_THRESHOLD = float(os.getenv('SOFT_PASS_THRESHOLD', '60.0'))
        has_high_hallucination = False
        for it in issues or []:
            if it.get('type') == 'hallucination' and it.get('severity') == 'high':
                has_high_hallucination = True
                break
        if has_high_hallucination:
            system_pass = False
            reasons.append('high_severity_hallucination')
        else:
            system_pass = validated_score >= SOFT_PASS_THRESHOLD
            if system_pass:
                reasons.append('validated_score_meets_soft_threshold')
            else:
                reasons.append('validated_score_below_soft_threshold')

    return {
        'raw_score': raw_score,
        'penalty': penalty,
        'validated_score': validated_score,
        'is_abstain': is_abstain,
        'abstention_type': abst_type,
        'system_pass': system_pass,
        'reasons': reasons,
    }
