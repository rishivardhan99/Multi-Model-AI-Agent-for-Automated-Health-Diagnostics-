"""
streamlit_app.py just for model1
FULL PIPELINE APP (OPTION 1) - UPDATED
-----------------------------------------
PDF / Image / JSON  → OCR / Extraction → structured.csv →
wide→long → standardization → interpretation → pivot tables

Improvements:
 - Ensures outputs are saved automatically to OUT_DIR (/app/outputs).
 - Calls interpret_dataframe(..., save_outputs=True) to write CSV/JSON/units files.
 - Hardened patient gender extraction to avoid float/NaN .lower() issues.
 - Displays units_used JSON and saved file paths in the UI.
 - Adds: Final CSV shaping that produces ONE CSV containing numeric values + status + note columns
         (removes empty parameters and drops globally INVALID params). This is implemented WITHOUT changing existing logic.
"""

# ensure project root is on Python path (helps inside Docker)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util
import json
import re
import time
import io
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------
EXTRACTOR = ROOT / "extractor"
OUT_DIR = Path("/app/outputs")   # always write outputs here (inside container)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = ROOT / "samples" / "uploaded"
SAMPLES.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Dynamic loader for extractor helpers
# ------------------------------------------------------------
def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, str(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------
# Load extractor modules (from extractor/)
# ------------------------------------------------------------
step1_pdf_utils = load_module("step1_pdf_utils", EXTRACTOR / "step1_pdf_utils.py")
step1_ocr_utils = load_module("step1_ocr_utils", EXTRACTOR / "step1_ocr_utils.py")
json_utils = load_module("json_utils", EXTRACTOR / "json_utils.py")
param_extractor = load_module("param_extractor", EXTRACTOR / "param_extractor.py")

is_pdf_digital = getattr(step1_pdf_utils, "is_pdf_digital")
extract_text_from_pdf = getattr(step1_pdf_utils, "extract_text_from_pdf")
pdf_to_images = getattr(step1_pdf_utils, "pdf_to_images")
ocr_image_to_text = getattr(step1_ocr_utils, "ocr_image_to_text")

extract_params_from_text = getattr(param_extractor, "extract_params_from_text")
fallback_line_scan = getattr(param_extractor, "fallback_line_scan")

try:
    PARAM_MAP = json.load(open(EXTRACTOR / "param_map.json", "r", encoding="utf-8"))
except Exception:
    PARAM_MAP = {}

PARAM_MAP_INTERNAL = getattr(param_extractor, "PARAM_MAP", PARAM_MAP)



# Load Model-1 modules (root)

from validation_and_standardization import read_structured_csv, standardize_dataframe
from model1_interpretation import interpret_dataframe
from table_utils import pivot_params_to_wide, summary_counts_by_interpretation, patient_level_flag


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Blood Report AI – Full Pipeline", layout="wide")
st.title("🩺 Blood Report Extraction → Standardization → Interpretation (Model-1)")

st.markdown(
    "Upload a **PDF / Image / JSON report**, or upload a **structured.csv**.\n\n"
    "- Digital PDF → direct text\n"
    "- Scanned → OCR (Tesseract)\n"
    "- JSON → flattened\n\n"
    "Structured CSVs and Model-1 outputs (CSV/JSON/units) are saved automatically into `/app/outputs`."
)



# Upload UI

uploaded = st.file_uploader("Upload file (pdf/png/jpg/jpeg/json/csv):",
                            type=["pdf", "png", "jpg", "jpeg", "json", "csv"])

existing = list(OUT_DIR.glob("*.structured.csv"))
selected_existing = None
if existing:
    selected_existing = st.selectbox("Or select existing structured.csv",
                                     options=["-- select --"] + [x.name for x in existing])



# Patient info extraction helpers (robust)

PID_RE = re.compile(r'Patient\s*ID[:\s\-]*([\w\-\./]+)', re.IGNORECASE)
AGE_RE = re.compile(r'\bAge[:\s\-]*(\d{1,3})', re.IGNORECASE)
GENDER_RE = re.compile(r'\bGender[:\s\-]*(Male|Female|M|F|Other)', re.IGNORECASE)


def _safe_gender_norm(g):
    """Return normalized gender string or None. Accepts floats/NaN."""
    if g is None:
        return None
    # pandas NaN is float('nan')
    try:
        if isinstance(g, float) and pd.isna(g):
            return None
    except Exception:
        pass
    s = str(g).strip()
    if not s:
        return None
    s_low = s.lower()
    if s_low.startswith("m"):
        return "Male"
    if s_low.startswith("f"):
        return "Female"
    return s  # fallback: return original string


def extract_patient_info(text: str):
    pid = None
    age = None
    gender = None
    if not text:
        return pid, age, gender

    m = PID_RE.search(text)
    if m:
        pid = m.group(1).strip()

    m = AGE_RE.search(text)
    if m:
        age = m.group(1).strip()

    m = GENDER_RE.search(text)
    if m:
        g = m.group(1)
        gender = _safe_gender_norm(g)

    return pid, age, gender


# ------------------------------------------------------------
# Downscale large images before OCR
# ------------------------------------------------------------
def downscale_if_needed(pil_img: Image.Image, max_area: int = 40_000_000):
    w, h = pil_img.size
    area = w * h
    if area <= max_area:
        return pil_img
    scale = (max_area / area) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


# ------------------------------------------------------------
# Convert file → text
# ------------------------------------------------------------
def text_for_file(path: Path, dpi=300):
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            if is_pdf_digital(str(path)):
                pages, _ = extract_text_from_pdf(str(path))
                return "\n".join(pages)
            else:
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

        if ext == ".json":
            if hasattr(json_utils, "load_json") and hasattr(json_utils, "flatten_json_text"):
                j = json_utils.load_json(str(path))
                return json_utils.flatten_json_text(j)
            else:
                return path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


# ------------------------------------------------------------
# Best-candidate selection (same as before)
# ------------------------------------------------------------
def _get_param_range(canon: str):
    info = PARAM_MAP_INTERNAL.get(canon, PARAM_MAP.get(canon, {}))
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

# ------------------------------------------------------------
# Extract parameters from text blob
# ------------------------------------------------------------
def extract_params_from_text_blob(text_blob: str):
    pid, age, gender = extract_patient_info(text_blob)

    primary = extract_params_from_text(text_blob) or []
    secondary = fallback_line_scan(text_blob) or []
    all_cands = [c for c in (primary + secondary) if c.get("canonical") and c.get("value") is not None]

    # group by canonical
    grouped = {}
    for c in all_cands:
        grouped.setdefault(c["canonical"], []).append(c)

    results = {}
    confs = {}
    raw_src = {}

    for canon in list(PARAM_MAP_INTERNAL.keys()):
        val, conf, raw = _select_best_candidate(canon, grouped.get(canon, []))
        results[canon] = val if val is not None else ""
        confs[canon] = conf or 0.0
        raw_src[canon] = raw

    return pid, age, gender, results, confs, raw_src


# ------------------------------------------------------------
# Parse file / main pipeline run
# ------------------------------------------------------------
structured_path = None
raw_text = ""
confs = {}
raw_sources = {}
CANONICALS = []

if uploaded and uploaded.name.endswith(".csv"):
    # User provided structured.csv directly
    local = SAMPLES / uploaded.name
    local.write_bytes(uploaded.getbuffer())
    # ensure saved into OUT_DIR as well
    structured_path = OUT_DIR / local.name
    # copy uploaded to outputs
    local_out = OUT_DIR / local.name
    local_out.write_bytes(local.read_bytes())
    st.success(f"Loaded structured CSV and saved to outputs: {local_out.name}")

elif uploaded:
    # Full extraction pipeline
    local = SAMPLES / uploaded.name
    local.write_bytes(uploaded.getbuffer())

    st.info(f"Extracting text from {uploaded.name} …")
    raw_text = text_for_file(local)

    st.info("Extracting parameters …")
    pid, age, gender, params, confs, raw_sources = extract_params_from_text_blob(raw_text)

    # Build structured wide row
    row = {"filename": uploaded.name, "patient_id": pid or "", "age": age or "", "gender": gender or ""}
    for c in list(PARAM_MAP_INTERNAL.keys()):
        row[c] = params.get(c, "")

    dfw = pd.DataFrame([row])
    structured_path = OUT_DIR / f"{local.stem}.structured.csv"
    dfw.to_csv(structured_path, index=False)
    st.success(f"structured.csv created → {structured_path.name}")

elif selected_existing and selected_existing != "-- select --":
    structured_path = OUT_DIR / selected_existing
    st.success(f"Using existing structured CSV: {structured_path.name}")

else:
    st.stop()


# ------------------------------------------------------------
# DISPLAY RAW EXTRACTION RESULTS
# ------------------------------------------------------------
st.subheader("Structured CSV (Wide Format)")
df_wide = pd.read_csv(structured_path)
st.dataframe(df_wide)

st.download_button("Download structured.csv",
                   structured_path.read_bytes(),
                   file_name=structured_path.name)

if raw_text:
    st.subheader("Raw Extracted Text")
    st.code(raw_text[:10000])

if confs:
    low = [
        {"parameter": k, "value": df_wide[k].iloc[0], "confidence": v, "raw_source": raw_sources.get(k)}
        for k, v in confs.items()
        if v < 0.7 and df_wide[k].iloc[0] != ""
    ]
    with st.expander("Low-confidence fields (< 0.7)"):
        st.dataframe(pd.DataFrame(low) if low else "None")


# ------------------------------------------------------------
# MODEL-1 PROCESSING
# ------------------------------------------------------------
st.header("Model-1 Processing")

st.subheader("Wide → Long")
df_long = read_structured_csv(structured_path)
st.dataframe(df_long)

st.subheader("Standardization / Validation")
df_std = standardize_dataframe(df_long)
st.dataframe(df_std)

st.subheader("Interpretation")
# call interpret_dataframe with auto-save into OUT_DIR and basename from structured file
try:
    df_interp, units_used = interpret_dataframe(
        df_std,
        save_outputs=True,
        out_dir=OUT_DIR,
        basename=Path(structured_path).stem,
        save_json=True,
        border_frac=0.10,
        units_tracking=True,
    )
    st.success(f"Saved Model-1 outputs to: {OUT_DIR}")
    # Show saved file names
    st.write("Saved files:")
    st.write(f"- {OUT_DIR / f'{Path(structured_path).stem}.model1_interpreted.csv'}")
    st.write(f"- {OUT_DIR / f'{Path(structured_path).stem}.model1_interpreted.json'}")
    st.write(f"- {OUT_DIR / f'{Path(structured_path).stem}.units_used.json'}")

except Exception as e:
    st.error("Error during interpretation:")
    st.exception(e)
    # Still attempt to continue with a safe fallback
    df_interp = pd.DataFrame()
    units_used = {}

# show interpreted dataframe if available
if not df_interp.empty:
    st.dataframe(df_interp)
else:
    st.write("No interpreted rows to display.")


st.subheader("Summary by Interpretation")
if not df_interp.empty:
    st.dataframe(summary_counts_by_interpretation(df_interp))
else:
    st.write("No summary available.")


st.subheader("Patient-Level Wide Table (Pivot)")
if not df_interp.empty:
    st.dataframe(pivot_params_to_wide(df_interp))
else:
    st.write("No pivot available.")


st.subheader("Patient-Level Flags")
if not df_interp.empty:
    st.dataframe(patient_level_flag(df_interp))
else:
    st.write("No patient-level flags available.")


# ------------------------------------------------------------
# Display units used and offer downloads for Model-1 outputs
# ------------------------------------------------------------
st.markdown("### Units observed during standardization")
st.json(units_used or {})

if (OUT_DIR / f"{Path(structured_path).stem}.model1_interpreted.csv").exists():
    with open(OUT_DIR / f"{Path(structured_path).stem}.model1_interpreted.csv", "rb") as fh:
        st.download_button("Download Model-1 CSV", fh.read(), file_name=f"{Path(structured_path).stem}.model1_interpreted.csv", mime="text/csv")

if (OUT_DIR / f"{Path(structured_path).stem}.model1_interpreted.json").exists():
    with open(OUT_DIR / f"{Path(structured_path).stem}.model1_interpreted.json", "rb") as fh:
        st.download_button("Download Model-1 JSON", fh.read(), file_name=f"{Path(structured_path).stem}.model1_interpreted.json", mime="application/json")

st.success("Pipeline completed successfully!")


# ------------------------------------------------------------
# FINAL CSV SHAPING — create a single CSV with numeric + status + note columns
# (This is an add-on and does not change any existing logic above.)
# Robust manual pivoting is used to avoid pivot_table failures.
# Additional fixes:
#  - trim statuses
#  - generate notes when missing (without re-classifying status)
#  - drop parameters that are INVALID across all rows
# ------------------------------------------------------------

final_csv_path = OUT_DIR / f"{Path(structured_path).stem}.model1_final.csv"

def _pick_column(df, keys):
    cols = [c for c in df.columns]
    for k in keys:
        for c in cols:
            if k in c.lower():
                return c
    return None

def _ref_range_str_for_param(canon):
    """Try to build a human-friendly reference range string from PARAM_MAP_INTERNAL."""
    info = PARAM_MAP_INTERNAL.get(canon) or PARAM_MAP.get(canon)
    if not info:
        return None
    rng = info.get("range") if isinstance(info, dict) else None
    unit = info.get("unit") if isinstance(info, dict) else None
    if rng and isinstance(rng, dict):
        lo = rng.get("min")
        hi = rng.get("max")
        if lo is not None and hi is not None:
            try:
                lo_f = float(lo)
                hi_f = float(hi)
                if unit:
                    return f"{lo_f:g}–{hi_f:g} {unit}"
                return f"{lo_f:g}–{hi_f:g}"
            except Exception:
                pass
        # fallback to textual min/max if present
        if lo is not None or hi is not None:
            parts = []
            if lo is not None:
                parts.append(f"min {lo}")
            if hi is not None:
                parts.append(f"max {hi}")
            if unit:
                return " ".join(parts) + f" {unit}"
            return " ".join(parts)
    # maybe there's a textual 'ref' field
    txt = info.get("ref") if isinstance(info, dict) else None
    if txt:
        return str(txt)
    return None

def _build_note_for_status(canon, value, status):
    """Create a short human-readable note when model1 hasn't provided one.
    Does NOT change status; only generates text for the _note column.
    """
    if status is None:
        return None
    s = str(status).strip().lower()
    ref = _ref_range_str_for_param(canon)
    # include value if numeric
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
    # fallback generic
    if val_str:
        return f"{canon}: {val_str} ({status})"
    return f"{canon}: {status}"

try:
    # Start from the wide numeric df we already displayed (df_wide)
    final_df = df_wide.copy()

    # Ensure filename exists as the primary key column
    if 'filename' not in final_df.columns:
        final_df.reset_index(inplace=True)
        if 'filename' not in final_df.columns and 'index' in final_df.columns:
            final_df.rename(columns={'index': 'filename'}, inplace=True)

    # Enrich from interpretation if available
    if not df_interp.empty:
        interp = df_interp.copy()

        # detect columns heuristically
        canonical_keys = ['canonical', 'canonical_name', 'parameter', 'param', 'test_name', 'test', 'name']
        value_keys     = ['value', 'result', 'measured', 'numeric', 'measured_value']
        status_keys    = ['status', 'interpretation', 'classification', 'flag', 'interpret']
        note_keys      = ['note', 'explain', 'message', 'interpretation_text', 'comment']

        param_col = _pick_column(interp, canonical_keys)
        value_col = _pick_column(interp, value_keys)
        status_col = _pick_column(interp, status_keys)
        note_col = _pick_column(interp, note_keys)

        st.info(f"Detected interp columns — param: {param_col}, value: {value_col}, status: {status_col}, note: {note_col}")

        # index for interp (prefer filename then patient_id)
        index_col = 'filename' if 'filename' in interp.columns else ('patient_id' if 'patient_id' in interp.columns else None)
        if index_col is None or param_col is None:
            st.warning("Interpretation output lacks filename/patient_id or canonical column — final CSV enrichment will be limited.")
        else:
            # build maps filename -> canonical -> status/note
            status_map = {}
            note_map = {}
            params_seen = set()

            # iterate rows carefully
            for _, row in interp.iterrows():
                idx = row.get(index_col)
                if pd.isna(idx):
                    continue
                idx = str(idx).strip()
                if not idx:
                    continue
                canon = row.get(param_col)
                if pd.isna(canon):
                    continue
                canon = str(canon).strip()
                if not canon:
                    continue
                params_seen.add(canon)
                status_map.setdefault(idx, {})
                note_map.setdefault(idx, {})

                # status
                if status_col and pd.notna(row.get(status_col)):
                    try:
                        raw_status = row.get(status_col)
                        # normalize to string and strip whitespace
                        if isinstance(raw_status, (list, tuple)):
                            raw_status = '; '.join(str(x).strip() for x in raw_status if str(x).strip())
                        else:
                            raw_status = str(raw_status).strip()
                        # trim and store
                        existing = status_map[idx].get(canon)
                        if existing:
                            parts = set([p.strip() for p in existing.split(';') if p.strip()])
                            parts.update(p.strip() for p in raw_status.split(';') if p.strip())
                            status_map[idx][canon] = '; '.join(sorted(parts))
                        else:
                            status_map[idx][canon] = raw_status
                    except Exception:
                        status_map[idx][canon] = str(row.get(status_col)).strip()

                # note
                if note_col and pd.notna(row.get(note_col)):
                    try:
                        raw_note = row.get(note_col)
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
                    except Exception:
                        note_map[idx][canon] = str(row.get(note_col)).strip()

            # Now create columns for each canonical seen
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
                    # trim status strings if present
                    if s is not None:
                        s = str(s).strip()
                    if n is not None:
                        n = str(n).strip()
                    # if note missing but status present, build a note
                    if (not n or n == "") and s:
                        # try to use numeric value from final_df if available
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

            st.success(f"Enriched final CSV with {len(all_canons)} interpreted parameters.")

    else:
        st.info("No interpretation available — final CSV will contain numeric values only.")

    # trim whitespace in all status columns (defensive)
    for c in list(final_df.columns):
        if c.endswith("_status"):
            final_df[c] = final_df[c].astype(str).map(lambda x: x.strip() if isinstance(x, str) else x)

    # DROP parameters that are INVALID across all rows (remove numeric, status, note)
    dropped_invalid = []
    status_cols = [c for c in final_df.columns if c.endswith("_status")]
    for sc in status_cols:
        # consider case-insensitive "INVALID"
        svals = final_df[sc].astype(str).map(lambda x: x.strip().lower() if isinstance(x, str) else "")
        if len(svals) > 0 and (svals == "invalid").all():
            base = sc[:-7]  # remove '_status'
            cols_to_remove = [base, sc, f"{base}_note"]
            removed = []
            for col in cols_to_remove:
                if col in final_df.columns:
                    final_df.drop(columns=col, inplace=True)
                    removed.append(col)
            if removed:
                dropped_invalid.append(base)

    if dropped_invalid:
        st.info(f"Dropped {len(dropped_invalid)} globally INVALID parameters: {', '.join(dropped_invalid)}")

    # Remove columns that are empty across all rows (treat '' and whitespace as empty)
    cols_to_drop = []
    for c in list(final_df.columns):
        series = final_df[c]
        non_null = series.dropna().astype(str).map(lambda x: x.strip()).replace('', np.nan).dropna()
        if non_null.empty:
            cols_to_drop.append(c)

    # But keep core metadata columns
    keep_core = {'filename', 'patient_id', 'age', 'gender'}
    cols_to_drop = [c for c in cols_to_drop if c not in keep_core]

    if cols_to_drop:
        final_df.drop(columns=cols_to_drop, inplace=True)
        st.info(f"Dropped empty columns: {', '.join(cols_to_drop)}")

    # Reset index (keep filename as column)
    final_df.reset_index(drop=True, inplace=True)

    # Save final CSV
    final_df.to_csv(final_csv_path, index=False)
    st.success(f"Final combined CSV created → {final_csv_path.name}")

    # Offer download
    with open(final_csv_path, 'rb') as fh:
        st.download_button("Download Final Model-1 CSV (values + status + note)", fh.read(), file_name=final_csv_path.name, mime='text/csv')

except Exception as e:
    st.error("Failed to create final combined CSV:")
    st.exception(e)

# End of updated streamlit_app.py
