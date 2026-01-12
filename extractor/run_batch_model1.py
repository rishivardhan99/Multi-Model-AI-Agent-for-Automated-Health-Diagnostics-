#!/usr/bin/env python3
"""
extractor/run_batch_model1.py
Batch runner that produces per-report final CSVs identical in shape to Streamlit UI:
 - outputs/structured_per_report/<stem>.structured.csv  (wide numeric + metadata)
 - outputs/model1_per_report/<stem>.model1_final.csv    (numeric + <param>_status + <param>_note)

Behavior:
 - Reuses extractor modules from extractor/*
 - Reuses standardization & interpretation from root modules (validation_and_standardization, model1_interpretation)
 - Applies sanity checks (PARAM_MAP ranges + global rules) to drop implausible values before interpretation
 - Applies the same final pivot & note-generation logic as Streamlit (no re-classification)
"""
import importlib.util
import json
import re
from PIL import Image
import csv
import sys
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # /app
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "extractor"
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"
STRUCTURED_DIR = OUT / "structured_per_report"
FINAL_DIR = OUT / "model1_per_report"

STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# dynamic loader
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

step1_pdf_utils = load_mod("step1_pdf_utils", EX / "step1_pdf_utils.py")
step1_ocr_utils = load_mod("step1_ocr_utils", EX / "step1_ocr_utils.py")
param_extractor = load_mod("param_extractor", EX / "param_extractor.py")
postprocess = load_mod("postprocess", EX / "postprocess.py")
json_utils = load_mod("json_utils", EX / "json_utils.py")

# model1 & standardization
from validation_and_standardization import read_structured_csv, standardize_dataframe
from model1_interpretation import interpret_dataframe

# helpers from loaded modules
is_pdf_digital = getattr(step1_pdf_utils, "is_pdf_digital", lambda p: False)
extract_text_from_pdf = getattr(step1_pdf_utils, "extract_text_from_pdf", lambda p: ([], {}))
pdf_to_images = getattr(step1_pdf_utils, "pdf_to_images", lambda p, dpi=300: [])
ocr_image_to_text = getattr(step1_ocr_utils, "ocr_image_to_text", lambda img, try_multiple=True: "")
extract_params_from_text = getattr(param_extractor, "extract_params_from_text", lambda txt: [])
fallback_line_scan = getattr(param_extractor, "fallback_line_scan", lambda txt: [])
postprocess_row = getattr(postprocess, "postprocess_row", lambda row, units_map=None: row)
PARAM_MAP = getattr(param_extractor, "PARAM_MAP", {})
try:
    PARAM_MAP_JSON = json.load(open(EX / "param_map.json", "r", encoding="utf-8"))
except Exception:
    PARAM_MAP_JSON = PARAM_MAP or {}

# patient extraction regex (same as Streamlit)
PID_RE = re.compile(r'Patient\s*ID[:\s\-]*([\w\-\./]+)', re.IGNORECASE)
AGE_RE = re.compile(r'\bAge[:\s\-]*(\d{1,3})', re.IGNORECASE)
GENDER_RE = re.compile(r'\bGender[:\s\-]*(Male|Female|M|F|Other)', re.IGNORECASE)

def _safe_gender_norm(g):
    if g is None:
        return ""
    try:
        if isinstance(g, float) and pd.isna(g):
            return ""
    except Exception:
        pass
    s = str(g).strip()
    if not s:
        return ""
    s_low = s.lower()
    if s_low.startswith("m"):
        return "Male"
    if s_low.startswith("f"):
        return "Female"
    return s

def extract_patient_info(text: str):
    if not text:
        return "", "", ""
    pid = ""
    age = ""
    gender = ""
    m = PID_RE.search(text)
    if m:
        pid = m.group(1).strip()
    m = AGE_RE.search(text)
    if m:
        age = m.group(1).strip()
    m = GENDER_RE.search(text)
    if m:
        gender = _safe_gender_norm(m.group(1))
    return pid, age, gender

def downscale_if_needed(pil_img: Image.Image, max_area: int = 40_000_000):
    w, h = pil_img.size
    area = w*h
    if area <= max_area:
        return pil_img
    scale = (max_area/area)**0.5
    new_w = max(1, int(w*scale))
    new_h = max(1, int(h*scale))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)

def text_for_file(path: Path, dpi=300):
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            if is_pdf_digital(str(path)):
                pages, _ = extract_text_from_pdf(str(path))
                return "\n".join(pages)
            imgs = pdf_to_images(str(path), dpi=dpi)
            out = []
            for im in imgs:
                if not isinstance(im, Image.Image):
                    im = Image.fromarray(im)
                im = downscale_if_needed(im)
                out.append(ocr_image_to_text(im, try_multiple=True))
            return "\n".join(out)
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            img = Image.open(str(path))
            img = downscale_if_needed(img)
            return ocr_image_to_text(img, try_multiple=True)
        if ext == ".json" and hasattr(json_utils, "load_json") and hasattr(json_utils, "flatten_json_text"):
            j = json_utils.load_json(str(path))
            return json_utils.flatten_json_text(j)
        # fallback read
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

# selection logic copied from Streamlit (to select best candidate per canonical)
def _get_param_range(canon: str):
    info = PARAM_MAP.get(canon, PARAM_MAP_JSON.get(canon, {}))
    if isinstance(info, dict):
        rng = info.get("range")
    else:
        rng = None
    if not rng or not isinstance(rng, dict):
        return None, None
    lo = rng.get("min", None)
    hi = rng.get("max", None)
    try:
        lo = float(lo) if lo is not None else None
        hi = float(hi) if hi is not None else None
    except Exception:
        lo, hi = None, None
    return lo, hi

def _select_best_candidate(canon: str, cands: list):
    if not cands:
        return None, None, None

    lo, hi = _get_param_range(canon)

    def in_range(v):
        if v is None:
            return False
        if lo is not None and v < lo:
            return False
        if hi is not None and v > hi:
            return False
        return True

    def score(c):
        return float(c.get("match_confidence") or 0.0)

    # ---------- NEW: de-prioritize range-boundary artifacts ----------
    def is_range_bound(c):
        v = c.get("value")
        if v is None:
            return False
        # exact boundary match
        if lo is not None and abs(v - lo) < 1e-9:
            return True
        if hi is not None and abs(v - hi) < 1e-9:
            return True
        # extractor already flags these
        if c.get("suspect_reason") in (
            "value_equals_range_bound",
            "value_may_be_range_bound_or_fallback",
        ):
            return True
        return False

    clean = [c for c in cands if not is_range_bound(c)]
    pool = clean if clean else cands
    # ---------------------------------------------------------------

    # ORIGINAL LOGIC (unchanged, but applied to pool)
    in_rng = [c for c in pool if in_range(c.get("value"))]
    if in_rng:
        best = max(in_rng, key=score)
        return best["value"], score(best), best.get("raw_name")

    relaxed = []
    if lo is not None and hi is not None:
        span = hi - lo
        lo2 = lo - 0.3 * span
        hi2 = hi + 0.3 * span
        for c in pool:
            v = c.get("value")
            if v is None:
                continue
            if lo2 <= v <= hi2:
                relaxed.append(c)
    if relaxed:
        best = max(relaxed, key=score)
        return best["value"], score(best), best.get("raw_name")

    plausible = [c for c in pool if isinstance(c.get("value"), (int, float)) and abs(c["value"]) < 1000]
    best = max(plausible or pool, key=score)
    return best["value"], score(best), best.get("raw_name")

def hard_drop_implausible(df):
    """
    Remove rows flagged as implausible by validation layer.
    This is the single enforcement gate for Model-1 sanitation.
    Returns a copy with implausible rows removed.
    """
    try:
        if "valid" not in df.columns or "invalid_reason" not in df.columns:
            return df
        before = len(df)
        df = df[
            ~(
                (df["valid"] == False) &
                (df["invalid_reason"] == "implausible_value")
            )
        ].copy()
        after = len(df)
        print(f"[Model-1] HARD DROP removed {before - after} implausible rows")
        return df
    except Exception as e:
        # defensive: do not crash pipeline
        print(f"[Model-1] hard_drop_implausible error: {e}")
        return df

# Sanity-check helper: uses PARAM_MAP if available, else global rules
def is_plausible_value(param, v):
    """
    Decide whether a numeric value is biologically plausible.

    This function MUST NOT reject valid abnormal values.
    It should only reject values that are physically / biologically impossible
    or clear OCR garbage.
    """
    try:
        fv = float(v)
    except Exception:
        return False

    # reject NaN / inf
    if not np.isfinite(fv):
        return False

    # reject zeros / negatives for labs that cannot be zero or negative
    # (keep conservative — do NOT use reference ranges)
    if fv <= 0:
        return False

    # absolute hard ceiling: OCR garbage protection
    # (covers cases like 99999, 1234567, etc.)
    if abs(fv) > 1e5:
        return False

    # OPTIONAL: very loose sanity bounds for known parameters (SAFE)
    info = PARAM_MAP.get(param) or PARAM_MAP_JSON.get(param)
    if isinstance(info, dict):
        hard = info.get("hard_limits")
        if isinstance(hard, dict):
            lo = hard.get("min")
            hi = hard.get("max")
            try:
                if lo is not None and fv < float(lo):
                    return False
                if hi is not None and fv > float(hi):
                    return False
            except Exception:
                pass

    # IMPORTANT:
    # DO NOT reject based on reference ranges
    # Abnormal values MUST survive
    return True

# Final CSV shaping helpers copied / adapted from Streamlit (note generation)
def _ref_range_str_for_param(canon):
    info = PARAM_MAP.get(canon) or PARAM_MAP_JSON.get(canon)
    if not info:
        return None
    rng = info.get("range") if isinstance(info, dict) else None
    unit = info.get("unit") if isinstance(info, dict) else None
    if rng and isinstance(rng, dict):
        lo = rng.get("min"); hi = rng.get("max")
        if lo is not None and hi is not None:
            try:
                lo_f = float(lo); hi_f = float(hi)
                if unit:
                    return f"{lo_f:g}–{hi_f:g} {unit}"
                return f"{lo_f:g}–{hi_f:g}"
            except Exception:
                pass
        if lo is not None or hi is not None:
            parts = []
            if lo is not None:
                parts.append(f"min {lo}")
            if hi is not None:
                parts.append(f"max {hi}")
            if unit:
                return " ".join(parts) + f" {unit}"
            return " ".join(parts)
    txt = info.get("ref") if isinstance(info, dict) else None
    if txt:
        return str(txt)
    return None

def _build_note_for_status(canon, value, status):
    if status is None:
        return None
    s = str(status).strip().lower()
    ref = _ref_range_str_for_param(canon)
    val_str = None
    try:
        val_f = float(value)
        val_str = f"{val_f:g}"
    except Exception:
        val_str = None
    if "normal" in s:
        if ref:
            return f"{canon} is within NORMAL range ({ref})"
        if val_str:
            return f"{canon} is within NORMAL range (value: {val_str})"
        return f"{canon} is within NORMAL range"
    if "low" in s and "border" not in s:
        if ref:
            return f"{canon} is LOW (< {ref})"
        if val_str:
            return f"{canon} is LOW (value: {val_str})"
        return f"{canon} is LOW"
    if "high" in s and "border" not in s:
        if ref:
            return f"{canon} is HIGH (> {ref})"
        if val_str:
            return f"{canon} is HIGH (value: {val_str})"
        return f"{canon} is HIGH"
    if "border" in s or "borderline" in s:
        if ref:
            return f"{canon} is BORDERLINE ({ref})"
        return f"{canon} is BORDERLINE"
    if val_str:
        return f"{canon}: {val_str} ({status})"
    return f"{canon}: {status}"

# Heuristic column pick (same as Streamlit)
def _pick_column(df, keys):
    cols = [c for c in df.columns]
    for k in keys:
        for c in cols:
            if k in c.lower():
                return c
    return None

# Main per-file processing
def process_file(path: Path):
    print("Processing:", path.name)
    txt = text_for_file(path)
    pid, age, gender = extract_patient_info(txt)

    primary = extract_params_from_text(txt) or []
    secondary = fallback_line_scan(txt) or []
    all_cands = primary + secondary
    # keep only candidates that have canonical & value
    all_cands = [c for c in all_cands if c.get("canonical") and c.get("value") is not None]

    # group by canonical
    grouped = {}
    for c in all_cands:
        grouped.setdefault(c["canonical"], []).append(c)

    # build wide row using selection logic (same as Streamlit)
    param_keys = list(PARAM_MAP.keys()) if PARAM_MAP else list(PARAM_MAP_JSON.keys())
    row = {"filename": path.name, "patient_id": pid or "", "age": age or "", "gender": gender or ""}
    for canon in param_keys:
        val, conf, raw = _select_best_candidate(canon, grouped.get(canon, []))
        row[canon] = val if val is not None else ""

    # SANITY CHECK: clear implausible values (and keep an audit note)
    # row_notes = {}
    # for k in list(row.keys()):
    #     if k in ("filename", "patient_id", "age", "gender"):
    #         continue
    #     v = row.get(k)
    #     if v in ("", None):
    #         continue
    #     try:
    #         if not is_plausible_value(k, v):
    #             row_notes[k] = f"Cleared by sanity-check (value: {v})"
    #             row[k] = ""
    #     except Exception:
    #         pass
    # if row_notes:
    #     # put into _notes key so postprocess can see if desired
    #     row["_notes"] = row_notes

    # postprocess salvage rules (uses your postprocess.postprocess_row)
    try:
        row = postprocess_row(row, units_map=None)
    except TypeError:
        # fallback if postprocess.postprocess_row signature takes only 1 arg
        try:
            row = postprocess_row(row)
        except Exception:
            pass
    except Exception:
        pass

    # write structured CSV for this report
    out_struct = STRUCTURED_DIR / f"{path.stem}.structured.csv"
    fieldnames = ["filename", "patient_id", "age", "gender"] + param_keys
    # ensure keys exist
    for k in fieldnames:
        if k not in row:
            row[k] = ""
    with open(out_struct, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    print("Wrote structured:", out_struct.name)
    

    # Now produce final Model-1 CSV (Streamlit-identical shaping)
    try:
        df_long = read_structured_csv(out_struct)
        df_std = standardize_dataframe(df_long)
        df_std = hard_drop_implausible(df_std)

    except Exception as e:
        print("Standardization error:", e)
        df_long = pd.DataFrame()
        df_std = pd.DataFrame()

    try:
        df_interp, units_used = interpret_dataframe(
            df_std,
            save_outputs=True,
            out_dir=OUT,
            basename=path.stem,
            save_json=True,
            border_frac=0.10,
            units_tracking=True,
        )
    except Exception as e:
        print("Interpretation error:", e)
        df_interp = pd.DataFrame()
        units_used = {}

    # Start final_df from df_wide (structured)
    final_df = pd.read_csv(out_struct)

    # compute numeric_cols present in this report (non-empty after postprocess)
    keep_core = {'filename', 'patient_id', 'age', 'gender'}
    numeric_cols = set()
    for c in final_df.columns:
        if c in keep_core:
            continue
        ser = final_df[c].dropna().astype(str).map(lambda x: x.strip()).replace('', pd.NA)
        if ser.notna().any():
            numeric_cols.add(c)

    # Enrich from interpretation (only for numeric_cols)
    if not df_interp.empty and numeric_cols:
        interp = df_interp.copy()
        canonical_keys = ['canonical', 'canonical_name', 'parameter', 'param', 'test_name', 'test', 'name']
        value_keys     = ['value', 'result', 'measured', 'numeric', 'measured_value']
        status_keys    = ['status', 'interpretation', 'classification', 'flag', 'interpret']
        note_keys      = ['note', 'explain', 'message', 'interpretation_text', 'comment']

        param_col = _pick_column(interp, canonical_keys) or "canonical"
        value_col = _pick_column(interp, value_keys)
        status_col = _pick_column(interp, status_keys)
        note_col = _pick_column(interp, note_keys)

        index_col = 'filename' if 'filename' in interp.columns else ('patient_id' if 'patient_id' in interp.columns else None)
        if index_col is None or param_col is None:
            print("Interp lacks filename/patient_id or canonical; skipping enrichment for", path.name)
        else:
            # filter to only params that are present in numeric_cols
            if param_col not in interp.columns:
                print("Param column not found in interp; skipping")
            else:
                df_interp_filtered = interp[interp[param_col].isin(numeric_cols)].copy()
                params_seen = set(df_interp_filtered[param_col].astype(str).unique().tolist())

                # build maps
                status_map = {}
                note_map = {}
                for _, r in df_interp_filtered.iterrows():
                    idx = r.get(index_col)
                    if pd.isna(idx):
                        continue
                    idx = str(idx).strip()
                    if not idx:
                        continue
                    canon = str(r.get(param_col)).strip()
                    if not canon:
                        continue
                    status_map.setdefault(idx, {})
                    note_map.setdefault(idx, {})
                    # status
                    if status_col and pd.notna(r.get(status_col)):
                        raw_status = r.get(status_col)
                        if isinstance(raw_status, (list, tuple)):
                            raw_status = '; '.join(str(x).strip() for x in raw_status if str(x).strip())
                        else:
                            raw_status = str(raw_status).strip()
                        existing = status_map[idx].get(canon)
                        if existing:
                            parts = set([p.strip() for p in existing.split(';') if p.strip()])
                            parts.update(p.strip() for p in raw_status.split(';') if p.strip())
                            status_map[idx][canon] = '; '.join(sorted(parts))
                        else:
                            status_map[idx][canon] = raw_status
                    # note
                    if note_col and pd.notna(r.get(note_col)):
                        raw_note = r.get(note_col)
                        if isinstance(raw_note, (list, tuple)):
                            raw_note = '; '.join(str(x).strip() for x in raw_note if str(x).strip())
                        else:
                            raw_note = str(raw_note).strip()
                        existing = note_map[idx].get(canon)
                        if existing:
                            parts = set([p.strip() for p in existing.split(';') if p.strip()])
                            parts.update(p.strip() for p in raw_note.split(';') if p.strip())
                            note_map[idx][canon] = '; '.join(sorted(parts))
                        else:
                            note_map[idx][canon] = raw_note

                filenames = final_df['filename'].astype(str).tolist()
                all_canons = sorted(params_seen)
                for canon in all_canons:
                    s_col = f"{canon}_status"
                    n_col = f"{canon}_note"
                    status_vals = []
                    note_vals = []
                    for fn in filenames:
                        s = status_map.get(fn, {}).get(canon)
                        n = note_map.get(fn, {}).get(canon)
                        if s is not None:
                            s = str(s).strip()
                        if n is not None:
                            n = str(n).strip()
                        if (not n or n == "") and s:
                            numeric_val = None
                            base_numeric_col = canon if canon in final_df.columns else None
                            if base_numeric_col:
                                try:
                                    numeric_val = final_df.loc[final_df['filename'].astype(str) == fn, base_numeric_col].iloc[0]
                                except Exception:
                                    numeric_val = None
                            n_generated = _build_note_for_status(canon, numeric_val, s)
                            n = n_generated
                        status_vals.append(s if s is not None else "")
                        note_vals.append(n if n is not None else "")
                    final_df[s_col] = status_vals
                    final_df[n_col] = note_vals

    # trim status whitespace
    for c in list(final_df.columns):
        if c.endswith("_status"):
            final_df[c] = final_df[c].astype(str).map(lambda x: x.strip() if isinstance(x, str) else x)

    # drop parameters that are INVALID across all rows (remove numeric + status + note)
    dropped_invalid = []
    status_cols = [c for c in final_df.columns if c.endswith("_status")]
    for sc in status_cols:
        svals = final_df[sc].astype(str).map(lambda x: x.strip().lower() if isinstance(x, str) else "")
        if len(svals) > 0 and (svals == "invalid").all():
            base_name = sc[:-7]
            cols_to_remove = [base_name, sc, f"{base_name}_note"]
            removed = []
            for col in cols_to_remove:
                if col in final_df.columns:
                    final_df.drop(columns=col, inplace=True)
                    removed.append(col)
            if removed:
                dropped_invalid.append(base_name)
    if dropped_invalid:
        print("Dropped globally INVALID params:", ", ".join(dropped_invalid))

    # drop empty columns (except metadata)
    cols_to_drop = []
    keep_core = {'filename', 'patient_id', 'age', 'gender'}
    for c in list(final_df.columns):
        if c in keep_core:
            continue
        series = final_df[c]
        non_null = series.dropna().astype(str).map(lambda x: x.strip()).replace('', np.nan).dropna()
        if non_null.empty:
            cols_to_drop.append(c)
    if cols_to_drop:
        final_df.drop(columns=cols_to_drop, inplace=True)
        print("Dropped empty columns:", ", ".join(cols_to_drop))

    # Persist final CSV per report
    out_final = FINAL_DIR / f"{path.stem}.model1_final.csv"
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_csv(out_final, index=False)
    print("Wrote final:", out_final.name)

def main():
    files = sorted(SAMPLES.glob("*.*"))
    if not files:
        print("No files found in samples/")
        return
    for p in files:
        try:
            process_file(p)
        except Exception as e:
            print("Error processing", p.name, ":", e)

if __name__ == "__main__":
    main()
