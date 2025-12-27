#!/usr/bin/env python3
# model3/model3_runner.py
"""
Model-3 runner that:
 - Uses LLMs (primary + fallback_models) to produce a guarded JSON narrative.
 - Validates schema (schema_model3.json).
 - Applies guardrails (detect & redact dangerous recommendations).
 - Enforces deterministic 'when_to_consult_doctor' priority (replaces LLM if too weak).
 - Writes exactly 3 files for audit: json, txt, prompt.
"""

import os
import sys
import argparse
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

from gemini_client import call_gemini
from prompts import build_prompt_from_row
from context import load_user_context, interpret_context, deterministic_consult_advice
from guardrails import scan_parsed_object_for_danger

load_dotenv()

SCHEMA_PATH = Path(__file__).parent / "schema_model3.json"
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|```$", flags=re.MULTILINE)

def strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("```json", "").replace("```", "")
    return CODE_FENCE_RE.sub("", text).strip()

def parse_json_text(text: str):
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

def load_model2_input(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    for key in ("rows", "data", "items", "results"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    if isinstance(raw, dict):
        vals = [v for v in raw.values() if isinstance(v, dict)]
        if vals:
            return vals
    raise RuntimeError("Unrecognized Model-2 JSON structure.")

def call_gemini_with_retry(prompt, model_name, max_retries=3):
    last_status = None
    last_resp = None
    last_text = None
    for attempt in range(1, max_retries + 1):
        status_code, resp_json, raw_text = call_gemini(prompt, model_name=model_name)
        last_status = status_code
        last_resp = resp_json
        last_text = raw_text
        if status_code == 200:
            return status_code, resp_json, raw_text
        if status_code in (429, 500, 502, 503):
            retry_after = 2 ** attempt
            try:
                retry_after = int(last_resp.get("error", {}).get("details", [{}])[-1].get("retryDelay", f"{retry_after}s").replace("s", ""))
            except Exception:
                pass
            print(f"[WARN] Gemini error {status_code}. Retrying in {retry_after}s (attempt {attempt}/{max_retries})")
            time.sleep(retry_after)
            continue
        break
    return last_status, last_resp, last_text

def deterministic_fallback_summary(model2_rows, user_context_interpreted=None):
    summary = (
        "This blood report was analyzed using validated numerical checks "
        "and pattern-based reasoning from earlier pipeline stages. "
        "Some laboratory parameters were outside reference ranges, "
        "but no definitive diagnosis can be made from this data alone."
    )
    explanations = []
    guidance = []
    for r in model2_rows:
        if isinstance(r, dict) and "patterns" in r:
            for k, v in r["patterns"].items():
                if isinstance(v, dict) and v.get("present"):
                    explanations.append(
                        f"Indicators related to {k.replace('_', ' ')} were observed; this can have multiple benign or transient causes."
                    )
    if not explanations:
        explanations.append("No strong pathological patterns were detected based on the available laboratory data.")
    guidance.extend([
        "Maintain a balanced diet and adequate hydration.",
        "Ensure sufficient rest and manage stress levels.",
        "Continue routine health monitoring as advised by a clinician."
    ])
    if user_context_interpreted and isinstance(user_context_interpreted, dict):
        for f in user_context_interpreted.get("lifestyle_focus", []):
            if f not in guidance:
                guidance.append(f)
    when_to_consult = "Consult a healthcare professional if symptoms develop, values worsen over time, or for routine clinical correlation."
    notes = "Generated without an LLM due to temporary service unavailability."
    if user_context_interpreted and user_context_interpreted.get("notes"):
        notes = user_context_interpreted.get("notes") + " " + notes
    return {
        "summary": summary,
        "possible_explanations": explanations[:3],
        "lifestyle_guidance": guidance[:4],
        "when_to_consult_doctor": when_to_consult,
        "notes": notes
    }

def compact_merged_context(rows):
    merged = {}
    if isinstance(rows, dict):
        src = rows
        if "patterns" in src:
            merged["patterns"] = src["patterns"]
        for k in ("causes", "causes_suspected", "causes_suspected_list", "causes_suspected"):
            if k in src:
                merged["causes"] = src[k]
                break
        for key in ("Platelets", "Hemoglobin", "MCHC", "MCH", "MCV", "Total_Cholesterol", "HDL", "LDL_estimated", "Triglycerides", "Urea_BUN", "CRP"):
            if key in src:
                merged[key] = src[key]
        if "score" in src:
            merged["score"] = src["score"]
        if "band" in src:
            merged["band"] = src["band"]
        if "severity" in src:
            merged["severity"] = src["severity"]
        if "confidence" in src:
            merged["confidence"] = src["confidence"]
        if "cardio" in src:
            merged["cardio"] = src["cardio"]
        return merged

    merged = {}
    for section in rows:
        if not isinstance(section, dict):
            continue
        if "patterns" in section and isinstance(section["patterns"], dict):
            merged.setdefault("patterns", {}).update({
                k: {"present": bool(v.get("present", False))}
                for k, v in section["patterns"].items() if isinstance(v, dict)
            })
        for k in ("causes", "causes_suspected", "causes_suspected_list"):
            if k in section:
                merged.setdefault("causes", [])
                val = section[k]
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "cause" in item:
                            merged["causes"].append(item["cause"])
                        elif isinstance(item, str):
                            merged["causes"].append(item)
                elif isinstance(val, str):
                    merged["causes"].append(val)
        for key in ("Platelets", "Hemoglobin", "MCHC", "MCH", "MCV", "Total_Cholesterol", "HDL", "LDL_estimated", "Triglycerides", "Urea_BUN", "CRP"):
            if key in section and key not in merged:
                merged[key] = section[key]
        if "score" in section and "score" not in merged:
            merged["score"] = section["score"]
        if "band" in section and "band" not in merged:
            merged["band"] = section["band"]
        if "severity" in section and "severity" not in merged:
            merged["severity"] = section["severity"]
        if "confidence" in section and "confidence" not in merged:
            merged["confidence"] = section["confidence"]
        if "cardio" in section and "cardio" not in merged:
            merged["cardio"] = section["cardio"]
    return merged

def enforce_consult_priority(parsed_obj, compact_context, user_context):
    """
    Ensure 'when_to_consult_doctor' is detailed and actionable. If LLM field is short/generic,
    replace with deterministic_consult_advice() and record deterministic advice in meta.
    Returns (patched_obj, meta_patch)
    """
    meta_patch = {}
    interpreted = interpret_context(user_context or {})
    deterministic = deterministic_consult_advice(compact_context, user_context or {}, interpreted)
    meta_patch["when_to_consult_deterministic"] = deterministic

    when_field = None
    if isinstance(parsed_obj, dict):
        when_field = parsed_obj.get("when_to_consult_doctor") or parsed_obj.get("when_to_consult")

    accept = False
    if isinstance(when_field, str):
        wf = when_field.strip()
        lower = wf.lower()
        trigger_keywords = ["if", "within", "hours", "days", "seek", "urgent", "immediate", "bleeding", "fever", "hospital", "symptom", "shortness", "breath", "pain", "consult", "repeat testing", "within 48", "within 72"]
        has_trigger = any(k in lower for k in trigger_keywords)
        if len(wf) >= 60 and has_trigger:
            accept = True

    if not accept:
        if isinstance(parsed_obj, dict):
            parsed_obj["when_to_consult_doctor"] = deterministic
            parsed_obj.setdefault("notes", "")
            if "replacement_of_when_to_consult" not in parsed_obj["notes"]:
                parsed_obj["notes"] = (parsed_obj["notes"] + " ").strip() + "replacement_of_when_to_consult: deterministic advice applied."
        else:
            parsed_obj = {"when_to_consult_doctor": deterministic}
    return parsed_obj, meta_patch

def main():
    p = argparse.ArgumentParser(description="Model-3 runner (Gemini) – single report synthesis with multi-model fallback")
    p.add_argument("--input", "-i", required=True, help="Path to Model-2 JSON file")
    p.add_argument("--out_dir", "-o", default="./outputs/model3", help="Output directory")
    p.add_argument("--model", "-m", default=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"), help="Primary Gemini model")
    p.add_argument("--fallback_models", "-f", default=os.getenv("FALLBACK_MODELS", ""), help="Comma-separated fallback models")
    p.add_argument("--user_context", "-u", default=None, help="Optional path to user_context JSON file (or an inline JSON string)")
    p.add_argument("--max_retries", type=int, default=3, help="Max retries per model on transient errors")
    p.add_argument("--force_fallback", action="store_true", help="Force deterministic fallback (for testing)")
    args = p.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print("Input file not found:", input_path, file=sys.stderr)
        sys.exit(2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_path.stem

    try:
        rows = load_model2_input(input_path)
    except Exception as e:
        print("Error loading Model-2 input:", e, file=sys.stderr)
        sys.exit(3)

    # load & sanitize user_context and interpret it
    user_context_sanitized = {}
    user_context_interpreted = {}
    if args.user_context:
        try:
            user_context_sanitized = load_user_context(args.user_context)
            user_context_interpreted = interpret_context(user_context_sanitized)
            # append interpretive summary to sanitized for prompts
            user_context_sanitized.update(user_context_interpreted)
        except Exception as e:
            print(f"[WARN] Could not load/interpret user_context {args.user_context}: {e}", file=sys.stderr)
            user_context_sanitized = {}
            user_context_interpreted = {}

    # merge model2 rows to compact context
    merged_context_raw = {}
    if isinstance(rows, dict):
        merged_context_raw = rows
    else:
        for section in rows:
            if isinstance(section, dict):
                for k, v in section.items():
                    if k not in merged_context_raw:
                        merged_context_raw[k] = v
                    else:
                        if isinstance(merged_context_raw[k], dict) and isinstance(v, dict):
                            merged_context_raw[k].update(v)
                        else:
                            merged_context_raw[f"{k}_extra"] = v

    compact_context = compact_merged_context(merged_context_raw)
    prompt = build_prompt_from_row(compact_context, user_context=user_context_sanitized)

    # build models list
    models_to_try = []
    primary_model = args.model or os.getenv("GEMINI_MODEL")
    if primary_model:
        models_to_try.append(primary_model)
    if args.fallback_models:
        for m in args.fallback_models.split(","):
            mstrip = m.strip()
            if mstrip and mstrip not in models_to_try:
                models_to_try.append(mstrip)
    if not args.fallback_models:
        env_fb = os.getenv("FALLBACK_MODELS", "")
        if env_fb:
            for m in env_fb.split(","):
                mstrip = m.strip()
                if mstrip and mstrip not in models_to_try:
                    models_to_try.append(mstrip)

    if args.force_fallback:
        print("[INFO] --force_fallback set; skipping LLM calls and using deterministic fallback.")
        result_record = {
            "input_file": input_path.name,
            "model": "deterministic_fallback",
            "status": "ok",
            "result": deterministic_fallback_summary(rows, user_context_interpreted),
            "meta": {"fallback": True}
        }
    else:
        model_attempts = []
        final_result = None
        final_model_used = None

        for model_name in models_to_try:
            if not model_name:
                continue
            print(f"[INFO] Trying model: {model_name}")
            status_code, resp_json, raw_text = call_gemini_with_retry(prompt, model_name=model_name, max_retries=args.max_retries)
            parsed = parse_json_text(raw_text)
            attempt = {"model": model_name, "status_code": status_code, "parsed_present": parsed is not None, "raw_text_snippet": (raw_text or "")[:800]}

            if isinstance(parsed, dict) and "error" in parsed:
                attempt["outcome"] = "llm_api_error"
                attempt["parsed"] = parsed
                model_attempts.append(attempt)
                print(f"[WARN] LLM API error from model {model_name}. Trying next model.")
                continue

            if parsed is None:
                attempt["outcome"] = "no_parsable_json"
                model_attempts.append(attempt)
                print(f"[WARN] No parsable JSON from model {model_name}. Trying next model.")
                continue

            # check & redact dangerous items
            had_danger, offenders, sanitized = scan_parsed_object_for_danger(parsed)
            if had_danger:
                attempt["outcome"] = "dangerous_content_redacted"
                attempt["parsed"] = sanitized
                attempt["danger_offenders"] = offenders
                model_attempts.append(attempt)
                # still attempt schema validation with sanitized output
                parsed = sanitized
                print(f"[WARN] Dangerous content detected and redacted ({offenders}). Proceeding with sanitized output.")

            try:
                validate(instance=parsed, schema=SCHEMA)
                attempt["outcome"] = "schema_valid"
                attempt["parsed"] = parsed
                model_attempts.append(attempt)

                # enforce deterministic consult priority
                patched_parsed, meta_patch = enforce_consult_priority(parsed, compact_context, user_context_sanitized)

                final_result = patched_parsed
                final_model_used = model_name

                result_record = {
                    "input_file": input_path.name,
                    "model": final_model_used,
                    "status": "ok",
                    "result": final_result,
                    "meta": {
                        "model_attempts": model_attempts,
                        **meta_patch,
                        "user_context_interpreted": user_context_interpreted
                    }
                }
                print(f"[INFO] Model {model_name} produced a schema-valid output (after guardrails & consult enforcement).")
                break

            except ValidationError as e:
                attempt["outcome"] = "schema_validation_failed"
                attempt["schema_error"] = str(e)
                attempt["parsed"] = parsed
                model_attempts.append(attempt)
                print(f"[WARN] Schema validation failed for model {model_name}: {e}. Trying next model.")
                continue

        if final_result is None:
            print("[INFO] All LLM attempts failed; using deterministic fallback summary.")
            result_record = {
                "input_file": input_path.name,
                "model": "deterministic_fallback",
                "status": "ok",
                "result": deterministic_fallback_summary(rows, user_context_interpreted),
                "meta": {
                    "fallback": True,
                    "model_attempts": model_attempts,
                    "user_context_interpreted": user_context_interpreted
                }
            }

    # write outputs (json, txt, prompt)
    json_path = out_dir / f"{base_name}.model3.json"
    json_path.write_text(json.dumps(result_record, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_path = out_dir / f"{base_name}.model3.txt"
    r = result_record["result"]
    if result_record.get("status") == "ok" and isinstance(r, dict):
        txt_content = (f"Report: {base_name}\n\n"
                       f"Summary:\n{r.get('summary','')}\n\n"
                       f"Possible explanations:\n")
        for item in r.get("possible_explanations", []):
            txt_content += f"- {item}\n"
        txt_content += "\nLifestyle guidance:\n"
        for item in r.get("lifestyle_guidance", []):
            txt_content += f"- {item}\n"
        txt_content += f"\nWhen to consult a doctor:\n{r.get('when_to_consult_doctor','')}\n\n"
        txt_content += f"Notes:\n{r.get('notes','')}\n\n"
        txt_content += "Disclaimer:\nThis is an automated, non-diagnostic summary based on Model-1 and Model-2 outputs.\n"
        if user_context_interpreted and user_context_interpreted.get("context_summary"):
            txt_content += f"\nContext summary: {user_context_interpreted.get('context_summary')}\n"
    else:
        txt_content = "Model-3 failed to generate a valid narrative. See JSON for details."
    txt_path.write_text(txt_content, encoding="utf-8")

    prompt_path = out_dir / f"{base_name}.model3.prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    print("Done. Output files (3):")
    print("-", json_path.name)
    print("-", txt_path.name)
    print("-", prompt_path.name)


if __name__ == "__main__":
    main()
