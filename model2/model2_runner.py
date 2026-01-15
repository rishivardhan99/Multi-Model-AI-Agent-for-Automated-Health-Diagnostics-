#!/usr/bin/env python3
# model2/model2_runner.py
"""
Orchestrator for Model-2 pipeline.

Usage:
    python model2_runner.py --input path/to/clean_00001.model1_final.csv --output_dir ./outputs

Produces:
  - outputs/model2_outputs/<basename>.model2.json
  - outputs/model2_outputs/<basename>.model2.txt
Returns: the final safe_output dict (and raises/logs on fatal errors)
"""
import argparse
import os
import json
import logging
import shutil
import time
from typing import Dict, Any, List, Optional

from .pipeline.loader import load_input
from .pipeline.pattern_engine import detect_patterns
from .pipeline.risk_engine import compute_derived, cardio_risk_band
from .pipeline.severity import label_from_range
from .pipeline.probable_causes import infer_probable_causes
from .pipeline.confidence import compute_confidence
from .pipeline.priors import BASE_PRIORS
from .pipeline.guardrails import sanitize_output

# NEW imports
from .pipeline.signals import extract_signals, enrich_patterns_with_signals
from .pipeline.themes import build_themes

# serializer helpers in same package
from .serializer import save_json, save_text, save_raw_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model2_runner")

META_IGNORE = {"age", "gender", "patient_id", "filename", "report_date"}


def run(input_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Execute the Model-2 pipeline end-to-end and write outputs.
    Returns the safe_output dict.
    """
    start_ts = int(time.time())
    try:
        os.makedirs(output_dir, exist_ok=True)
        model2_dir = os.path.join(output_dir, "model2_outputs")
        os.makedirs(model2_dir, exist_ok=True)

        base = os.path.basename(input_path)
        base_noext = os.path.splitext(base)[0]

        # copy raw input into model2_outputs for traceability (best-effort)
        try:
            save_raw_input(input_path, model2_dir, f"raw_input_{base}")
        except Exception:
            logger.debug("Could not copy raw input (maybe input is inside container). Continuing.")

        # 0. Load Model-1 structured dict
        model1_struct = load_input(input_path)
        logger.info(f"Loaded model1 structured keys: {list(model1_struct.keys())}")

        # Build flat params (numerics expected by downstream)
        flat_params: Dict[str, Any] = {}
        flat_params.update(model1_struct.get("parameters", {}) or {})
        # attach age/gender for reference-range lookups
        if "age" in model1_struct:
            flat_params["age"] = model1_struct.get("age")
        if "gender" in model1_struct:
            flat_params["gender"] = model1_struct.get("gender")

        # compute missing params (only expected numeric param keys)
        missing_params = [k for k, v in (model1_struct.get("parameters") or {}).items() if v is None]

        # 1. Derived metrics
        derived = compute_derived(flat_params)

        # 2. Pattern detection
        patterns = detect_patterns(flat_params)

        # 2.5 Signal extraction (neutral physiological signals)
        signals = extract_signals(patterns, derived, flat_params)

        # 2.7 Enrich patterns with signals (add subclinical flags where appropriate)
        try:
            enrich_patterns_with_signals(patterns, signals, threshold=0.6)
        except Exception:
            logger.debug("Could not enrich patterns with signals (enrich function failed). Continuing.")

        # 3. Theme discovery (determine what the report is actually about)
        themes = build_themes(signals)

        # 4. Build observations for KG/probable causes
        observations: List[str] = []
        # from detected patterns' support lists
        for pname, pinfo in (patterns.get("patterns", {}) or {}).items():
            if isinstance(pinfo, dict) and pinfo.get("present"):
                observations.extend(pinfo.get("support", []) or [])

        # prefer Model-1 status flags when available; fallback on severity labels
        status_map = model1_struct.get("status", {}) or {}
        key_params = (
            "Hemoglobin", "Platelets", "MCV", "RDW", "LDL", "Total_Cholesterol",
            "Triglycerides", "HDL", "Creatinine", "CRP", "Glucose_Fasting", "HbA1c",
            "Neutrophils", "Lymphocytes", "WBC"
        )

        for param in key_params:
            st = status_map.get(param)
            if st:
                st_up = str(st).upper()
                if "LOW" in st_up:
                    observations.append(f"{param}_LOW")
                elif "HIGH" in st_up:
                    observations.append(f"{param}_HIGH")
            else:
                # fallback to label_from_range only if numeric present
                val = flat_params.get(param)
                if isinstance(val, (int, float)):
                    lab_label = label_from_range(param, val, age=flat_params.get("age"), gender=flat_params.get("gender"))
                    lbl = lab_label.get("label", "")
                    if isinstance(lbl, str) and (lbl.startswith("low") or lbl.startswith("high") or "severe" in lbl):
                        node = f"{param}_LOW" if lbl.startswith("low") else f"{param}_HIGH"
                        observations.append(node)

        # 5. Probable causes (KG + heuristics) — now receives themes so priors may be conditional
        probable = infer_probable_causes(observations, patterns, priors=BASE_PRIORS, themes=themes)

        # 6. Cardio risk — compute only if themes suggest lipids/metabolic relevance
        cardio = cardio_risk_band(flat_params, derived, themes=themes)

        # 7. Severity map per numeric parameter (for display)
        severity_map: Dict[str, Any] = {}
        for param, val in (model1_struct.get("parameters") or {}).items():
            if isinstance(val, (int, float)):
                severity_map[param] = label_from_range(param, val, age=flat_params.get("age"), gender=flat_params.get("gender"))

        # 8. Confidence (now includes theme signal)
        # compute_confidence now returns a dict {"score", "components", "explanation"}
        confidence = compute_confidence(flat_params, patterns, probable, missing_params, themes)

        # 9. Assemble final structured output
        output: Dict[str, Any] = {
            "metadata": {
                "input_file": os.path.abspath(input_path),
                "base": base_noext,
                "generated_at": start_ts,
            },
            "parameters": model1_struct.get("parameters", {}),
            "status": model1_struct.get("status", {}),
            "notes": model1_struct.get("notes", {}),
            "derived": derived,
            "patterns": patterns,
            "signals": signals,
            "themes": themes,
            "probable_causes": probable,
            "contextual_risks": {
                # cardio may be {"applicable": False} or have band/score
                "cardiovascular": cardio
            },
            "severity": severity_map,
            "confidence": confidence,
            "notes_summary": "Model-2 deterministic + KG reasoning. This output is not a diagnosis. Refer to clinician.",
        }

        # 10. Sanitize (guardrails)
        safe_output = sanitize_output(output)

        # 11. Persist outputs inside model2_outputs
        out_json_path = os.path.join(model2_dir, base_noext + ".model2.json")
        out_txt_path = os.path.join(model2_dir, base_noext + ".model2.txt")
        save_json(os.path.basename(out_json_path), safe_output, folder=model2_dir)
        save_text(os.path.basename(out_txt_path), human_summary(safe_output), folder=model2_dir)

        logger.info(f"Model-2 outputs written: {out_json_path}, {out_txt_path}")
        return safe_output

    except Exception as e:
        logger.exception("Model-2 run failed")
        # write a small error log into model2_outputs if possible
        try:
            model2_dir = os.path.join(output_dir, "model2_outputs")
            os.makedirs(model2_dir, exist_ok=True)
            with open(os.path.join(model2_dir, "error_log.txt"), "a", encoding="utf-8") as ef:
                ef.write(f"TS:{int(time.time())}\nINPUT: {input_path}\nERROR: {str(e)}\n\n")
        except Exception:
            pass
        raise


def human_summary(out: dict) -> str:
    """
    Produce a compact human-readable summary that the UI can display or provide as a download.
    """
    lines: List[str] = []
    md = out.get("metadata", {})
    lines.append(f"Report: {md.get('base', '-')}")
    lines.append("")
    # Derived metrics
    if out.get("derived"):
        lines.append("Derived metrics:")
        for k, v in out.get("derived", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    # Patterns
    lines.append("Detected patterns:")
    pats = out.get("patterns", {}).get("patterns", {})
    found_any = False
    for pname, pinfo in pats.items():
        present = pinfo.get("present")
        subclinical = pinfo.get("subclinical")
        if present:
            found_any = True
            typ = pinfo.get("type") or pinfo.get("severity") or ""
            support = ", ".join(pinfo.get("support", [])[:4])
            lines.append(f"- {pname}: present ({typ}) support: {support}")
        elif subclinical:
            found_any = True
            typ = pinfo.get("type") or pinfo.get("severity") or ""
            support = ", ".join(pinfo.get("support", [])[:4])
            lines.append(f"- {pname}: subclinical signal present ({typ}) support: {support}")
    if not found_any:
        lines.append("- None of the tracked patterns were strongly present.")
    lines.append("")
    # Themes (new)
    lines.append("Dominant themes (what this report is mainly about):")
    for t in out.get("themes", [])[:3]:
        try:
            lines.append(f"- {t.get('theme')}: strength={t.get('strength'):.2f}; evidence: {', '.join(t.get('evidence', [])[:4])}")
        except Exception:
            lines.append(f"- {t}")
    lines.append("")
    # Probable causes (top)
    lines.append("Top probable causes:")
    for c in out.get("probable_causes", {}).get("causes", [])[:6]:
        adj = c.get("adjustment", "")
        support = ", ".join(c.get("support", [])[:3])
        lines.append(f"- {c.get('cause')} (score: {c.get('score')}) support: {support} {adj}")
    lines.append("")
    # Cardio (only if applicable)
    cardio_obj = out.get("contextual_risks", {}).get("cardiovascular", {})
    if cardio_obj.get("applicable") is False:
        lines.append("Cardiovascular risk: Not applicable for this report (no dominant lipid/metabolic signal).")
    else:
        lines.append(f"Cardiovascular risk band: {cardio_obj.get('band')} (score {cardio_obj.get('score')})")
    lines.append("")
    # Confidence
    conf = out.get("confidence", {})
    # prefer a friendlier high-confidence message when appropriate
    try:
        patterns_present = any(v.get("present") for v in out.get("patterns", {}).get("patterns", {}).values())
    except Exception:
        patterns_present = False
    if conf.get("score") is not None:
        if conf.get("score") >= 0.65 and not patterns_present:
            lines.append("Confidence: High — laboratory parameters are present and no strong pathological patterns were detected.")
        else:
            lines.append(f"Overall confidence: {conf.get('score')}")
            if conf.get("explanation"):
                lines.append(f"Confidence explanation: {conf.get('explanation')}")
    lines.append("")
    lines.append("Notes:")
    lines.append(out.get("notes_summary", ""))
    lines.append("")
    lines.append("Disclaimer: This is an automated, non-diagnostic summary based on Model-1 and Model-2 outputs.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model-2 pipeline on Model-1 outputs")
    parser.add_argument("--input", "-i", required=True, help="Path to Model-1 output (csv/json)")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory to write Model-2 outputs (will create outputs/model2_outputs/)")
    args = parser.parse_args()
    run(args.input, args.output_dir)
