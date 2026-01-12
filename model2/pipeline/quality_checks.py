# model2/pipeline/quality_checks.py
"""
Quality checks for Model-2 inputs.

Functions:
 - detect_low_data_quality(parameters: Dict[str,Any], min_params:int=4, identical_threshold:int=3) -> Dict
    Returns flags:
      {
        "low_data_quality": bool,
        "reason": str,
        "distinct_values": int,
        "total_numeric_values": int,
        "identical_value": <value> | None,
        "identical_count": int,
        "missing_fraction": float
      }

This module is intentionally conservative and deterministic.
"""

from typing import Dict, Any, Optional, List
import math
import statistics

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))

def detect_low_data_quality(
    parameters: Dict[str, Any],
    expected_params: Optional[List[str]] = None,
    min_params: int = 4,
    identical_threshold: int = 3,
    distinct_threshold_ratio: float = 0.3
) -> Dict[str, Any]:
    """
    Inspect parameter dict for signals of corrupted or low-quality input.

    Heuristics used:
      - If total numeric parameters >= min_params and the number of distinct numeric
        values <= identical_threshold -> likely broadcast/placeholder values.
      - If many params present but distinct_values / total_numeric_values < distinct_threshold_ratio -> suspicious.
      - If large missing_fraction (based on expected_params) -> low quality.

    Returns a dictionary of flags and metrics (non-exceptional).
    """
    out = {
        "low_data_quality": False,
        "reason": "",
        "distinct_values": 0,
        "total_numeric_values": 0,
        "identical_value": None,
        "identical_count": 0,
        "missing_fraction": 0.0,
    }

    if not isinstance(parameters, dict):
        out["low_data_quality"] = True
        out["reason"] = "invalid_parameters_type"
        return out

    numeric_values = []
    for k, v in parameters.items():
        if v is None:
            continue
        if _is_number(v):
            # round to 3 decimals to collapse OCR noise (0.0001->0)
            try:
                numeric_values.append(round(float(v), 3))
            except Exception:
                continue

    total_numeric = len(numeric_values)
    out["total_numeric_values"] = total_numeric

    if total_numeric == 0:
        # No numeric data at all -> low quality
        out["low_data_quality"] = True
        out["reason"] = "no_numeric_values"
        out["distinct_values"] = 0
        out["missing_fraction"] = 1.0 if expected_params else 0.0
        return out

    distinct = {}
    for val in numeric_values:
        distinct[val] = distinct.get(val, 0) + 1

    distinct_count = len(distinct)
    out["distinct_values"] = distinct_count

    # find the most common numeric value and its count
    most_common_val = max(distinct.items(), key=lambda x: x[1])
    out["identical_value"] = most_common_val[0]
    out["identical_count"] = most_common_val[1]

    # compute missing fraction if expected params provided
    missing_frac = 0.0
    if expected_params:
        expected_set = set(expected_params)
        present = 0
        for p in expected_params:
            v = parameters.get(p)
            if v is not None:
                present += 1
        missing_frac = 1.0 - (present / len(expected_params)) if len(expected_params) else 0.0
    out["missing_fraction"] = round(missing_frac, 3)

    # Heuristic rules for low data quality:
    # 1) If we have >= min_params numeric values and identical_count >= identical_threshold AND identical_count/total_numeric >= 0.5 -> suspicious
    if total_numeric >= min_params and out["identical_count"] >= identical_threshold and (out["identical_count"] / total_numeric) >= 0.5:
        out["low_data_quality"] = True
        out["reason"] = "broadcast_identical_value"
        return out

    # 2) If too few distinct values relative to total (e.g., < distinct_threshold_ratio)
    if total_numeric >= min_params and (distinct_count / max(1, total_numeric)) <= distinct_threshold_ratio:
        out["low_data_quality"] = True
        out["reason"] = "low_distinct_value_diversity"
        return out

    # 3) If missing fraction is large (>50%) and there are at least some numeric values -> low quality
    if expected_params and missing_frac >= 0.5 and total_numeric < len(expected_params) * 0.4:
        out["low_data_quality"] = True
        out["reason"] = "many_missing_params"
        return out

    # otherwise consider quality acceptable
    out["low_data_quality"] = False
    out["reason"] = "ok"
    return out
