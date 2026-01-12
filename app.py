# app.py — Medicube (final, with persistent UI rendering + Chatbot integration)
# Place this file in your project root (same layout as before): extractor/* , model2/* , model3/* and outputs/

import streamlit as st
from pathlib import Path
import sys
import subprocess
import time
import json
import traceback
import os
import shutil
import zipfile
from typing import Optional, Tuple, Any, Dict, List
import csv

import pandas as pd
from dotenv import load_dotenv

# ------------------------- session-state defaults -------------------------
if "last_manifest" not in st.session_state:
    st.session_state.last_manifest = None

if "last_canonical_base" not in st.session_state:
    st.session_state.last_canonical_base = None

if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = False

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"
STRUCTURED_DIR = OUT / "structured_per_report"
MODEL1_DIR = OUT / "model1_per_report"
MODEL2_DIR = OUT / "model2_outputs"
MODEL3_DIR = OUT / "model3"
BATCH_OUT = OUT / "batch"
MANIFEST_DIR = OUT / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# ensure directories exist
for d in (SAMPLES, OUT, STRUCTURED_DIR, MODEL1_DIR, MODEL2_DIR, MODEL3_DIR, BATCH_OUT):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Medicube — Multi-model AI Medical Diagnosis", layout="wide", initial_sidebar_state="expanded")

# ------------------------- Polished CSS -------------------------
st.markdown(
    """
    <style>
      .big-title { font-size:28px; font-weight:800; margin-bottom:6px; color:#03293a; }
      .muted { color:#4b6b7a; margin-bottom:8px; }
      .card { background: linear-gradient(180deg,#ffffff,#f7fbfe); border-radius:10px; padding:16px; margin-bottom:14px; color:#04293a; box-shadow: 0 6px 18px rgba(2,12,27,0.08); }
      .small { font-size:13px; color:#5b6f79; }
      .progress-badge { font-weight:700; padding:6px 10px; border-radius:10px; color:#fff; font-size:13px; }
      .running { background:#f59e0b; }
      .success { background:#16a34a; }
      .failed { background:#ef4444; }
      .pending { background:#6b7280; }
      .uploader-box { border: 2px dashed #cfe9ff; border-radius:8px; padding:12px; background:#fcfeff; }
      .samples-list { max-height:220px; overflow:auto; }
      .section-title { font-size:16px; font-weight:700; color:#083b4a; margin-bottom:8px; }
      .mono { font-family: monospace; background:#f3f6f8; padding:6px; border-radius:6px; display:block; }
    </style>
    """, unsafe_allow_html=True
)

# ------------------------- Helpers (kept and extended) -------------------------
def qualitative_strength(score: Optional[float]) -> str:
    try:
        s = float(score)
    except Exception:
        return "weak"
    if s >= 0.75:
        return "strong"
    if s >= 0.45:
        return "moderate"
    return "weak"

def run_subprocess(cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None, env: Optional[Dict[str,str]] = None) -> Tuple[int, str, str]:
    if not cmd:
        return 1, "", "empty command"
    exec_cmd = cmd
    first = Path(cmd[0])
    if first.suffix == ".py":
        exec_cmd = [sys.executable] + cmd
    env_final = os.environ.copy()
    if env:
        env_final.update(env)
    try:
        proc = subprocess.Popen(exec_cmd, cwd=str(cwd) if cwd else None,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_final)
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", err or ""
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            out, err = proc.communicate()
        except Exception:
            out, err = "", "timeout and kill failed"
        return 124, out or "", err or "timeout"
    except Exception as e:
        return 1, "", f"subprocess invocation failed: {e}"

def latest_file_in_dir(d: Path, pattern: str):
    files = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def safe_json_load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def render_progress_card(title: str, status: str, detail: str = ""):
    cls = {"running": "running", "success": "success", "failed": "failed", "pending": "pending"}.get(status, "pending")
    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:700; font-size:15px;">{title}</div>
            <div class="progress-badge {cls}">{status.capitalize()}</div>
          </div>
          <div style="margin-top:8px;" class="small">{detail}</div>
        </div>
        """, unsafe_allow_html=True
    )

# ------------------------- ZIP extraction utility -------------------------
ALLOWED_EXT = {'.pdf', '.png', '.jpg', '.jpeg', '.json', '.tiff', '.bmp', '.tif'}

def extract_zip_to_samples(zip_path: Path, dest_dir: Path = SAMPLES) -> List[Path]:
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.endswith('/') or member.startswith('__MACOSX'):
                    continue
                ext = Path(member).suffix.lower()
                if ext not in ALLOWED_EXT:
                    continue
                base_name = Path(member).name
                target = dest_dir / base_name
                if target.exists():
                    stem = target.stem
                    suff = target.suffix
                    ts = int(time.time()*1000)
                    target = dest_dir / f"{stem}_{ts}{suff}"
                with zf.open(member) as src, open(target, "wb") as out_f:
                    shutil.copyfileobj(src, out_f)
                extracted.append(target)
    except Exception as e:
        raise RuntimeError(f"Failed to extract zip: {e}")
    return extracted

# ------------------------- Pipeline helpers (kept) -------------------------

def run_extractor_batch(input_sample_path: Path, timeout: int = 600) -> Tuple[bool, str]:
    extractor_candidates = [
        ROOT / "extractor" / "process_file.py",
        ROOT / "extractor" / "run_batch_model1.py",
        ROOT / "extractor" / "process_samples.py",
        ROOT / "extractor" / "run_extractor.py",
        ROOT / "extractor" / "batch_runner.py",
        ROOT / "extractor" / "step1_ingest.py",
        ROOT / "run_extractor.py",
        ROOT / "process_samples.py",
        ROOT / "process_file.py",
    ]
    found = None
    for cand in extractor_candidates:
        if cand.exists():
            found = cand
            break
    if not found:
        return False, f"No extractor script found. Looked for: {', '.join(str(p.name) for p in extractor_candidates)}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    rc, out, err = run_subprocess([str(found), str(input_sample_path)], cwd=ROOT, timeout=timeout, env=env)
    if rc != 0:
        return False, f"Extractor script {found.name} failed (rc={rc}). STDOUT:\n{out}\nSTDERR:\n{err}"
    return True, f"Extractor script {found.name} ran (with arg)."

def run_model1_adapter_for(base_stem: str, timeout: int = 300) -> Tuple[Optional[Path], str]:
    adapter_candidates = [
        ROOT / "extractor" / "run_model1_on_csv.py",
        ROOT / "run_model1_on_csv.py",
        ROOT / "extractor" / "run_batch_model1.py",
    ]
    adapter = None
    for p in adapter_candidates:
        if p.exists():
            adapter = p
            break
    if adapter is None:
        return None, "Model-1 adapter not found (expected extractor/run_model1_on_csv.py)"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    rc, out, err = run_subprocess([str(adapter)], cwd=ROOT, timeout=timeout, env=env)
    if rc != 0:
        return None, f"Model-1 adapter failed (rc={rc}). STDOUT:\n{out}\nSTDERR:\n{err}"
    candidate = MODEL1_DIR / f"{base_stem}.model1_final.csv"
    if candidate.exists():
        return candidate, "Model-1 final CSV produced"
    newest = latest_file_in_dir(MODEL1_DIR, "*.model1_final.csv")
    if newest:
        return newest, f"Adapter finished; using newest model1 file {newest.name}"
    return None, "Adapter finished but could not find any model1_final.csv in outputs/model1_per_report"

def run_model2(model1_csv: Path, timeout: int = 300):
    cmd = [
        sys.executable,
        "-m",
        "model2.model2_runner",
        "--input", str(model1_csv),
        "--output_dir", str(OUT)
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    rc, out, err = run_subprocess(cmd, cwd=ROOT, timeout=timeout, env=env)
    if rc != 0:
        return None, f"Model-2 subprocess failed (rc={rc}). STDOUT:\n{out}\nSTDERR:\n{err}"
    jpath = OUT / "model2_outputs" / (model1_csv.stem + ".model2.json")
    if not jpath.exists():
        newest = latest_file_in_dir(OUT / "model2_outputs", "*.model2.json")
        if newest:
            jpath = newest
    if jpath and jpath.exists():
        data = safe_json_load(jpath)
        return data, f"model2 output loaded ({jpath.name})"
    return None, "Model-2 completed but output JSON not found"

def run_model3(model2_json_path: Path, model_name: Optional[str] = None, timeout: int = 900, extra_env: Optional[Dict[str,str]] = None, user_context_path: Optional[Path] = None) -> Tuple[Optional[dict], str]:
    runner = ROOT / "model3" / "model3_runner.py"
    if not runner.exists():
        return None, f"Model-3 runner not found at {runner}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if extra_env:
        env.update(extra_env)
    cmd = [str(runner), "--input", str(model2_json_path), "--out_dir", str(MODEL3_DIR)]
    if model_name:
        cmd += ["--model", model_name]
    if user_context_path:
        cmd += ["--user_context", str(user_context_path)]
    rc, out, err = run_subprocess(cmd, cwd=ROOT, timeout=timeout, env=env)
    if rc != 0:
        return None, f"Model-3 subprocess failed (rc={rc}). STDOUT:\n{out}\nSTDERR:\n{err}"
    jpath = MODEL3_DIR / (model2_json_path.stem + ".model3.json")
    if not jpath.exists():
        newest = latest_file_in_dir(MODEL3_DIR, "*.model3.json")
        if newest:
            jpath = newest
    if jpath and jpath.exists():
        data = safe_json_load(jpath)
        return data, f"Model-3 output loaded ({jpath.name})"
    return None, "Model-3 completed but model3.json not found"

# ------------------------- Load model3/.env if present -------------------------
model3_env_file = ROOT / "model3" / ".env"
if model3_env_file.exists():
    try:
        load_dotenv(dotenv_path=str(model3_env_file))
    except Exception:
        pass

# ------------------------- Load chatbot/.env if present (important for Chatbot LLM) -------------------------
chatbot_env_file = ROOT / "chatbot" / ".env"
if chatbot_env_file.exists():
    try:
        load_dotenv(dotenv_path=str(chatbot_env_file))
    except Exception:
        pass

# ------------------------- Utility: ensure Model-3 returns required structure -------------------------
def resolve_model3_artifacts(canonical_base: str) -> dict:
    """
    Resolve actual Model-3 artifact paths even if runner used extended stems.
    Looks for files like:
      <canonical_base>*.model3.txt
      <canonical_base>*.model3.json
    Returns dict with Path or None for 'txt' and 'json'.
    """
    candidates = list(MODEL3_DIR.glob(f"{canonical_base}*.model3.txt"))
    txt = candidates[0] if candidates else None
    candidates_json = list(MODEL3_DIR.glob(f"{canonical_base}*.model3.json"))
    js = candidates_json[0] if candidates_json else None
    return {"txt": txt, "json": js}

def ensure_model3_keys(result: dict) -> dict:
    r = dict(result) if isinstance(result, dict) else {}
    r.setdefault("summary", "")
    r.setdefault("possible_explanations", [])
    r.setdefault("lifestyle_guidance", [])
    r.setdefault("notes", "")
    w = r.get("when_to_consult_doctor")
    if not w or not isinstance(w, str) or len(w.strip()) == 0:
        combined_text = " ".join([json.dumps(r.get("possible_explanations","")), str(r.get("notes","")), r.get("summary","")]).lower()
        severity_flag = any(k in combined_text for k in ("severe", "very_severe", "very severe", "urgent", "bleeding", "hospital"))
        if severity_flag:
            r["when_to_consult_doctor"] = "Consider contacting a clinician promptly (within 24–48 hours) for clinical correlation."
        else:
            r["when_to_consult_doctor"] = ("Discuss results with a clinician for interpretation and follow-up if you have symptoms, "
                                           "if values persist on repeat testing, or if you are concerned.")
    return r

# ------------------------- Sidebar: API, model override, debug -------------------------
st.sidebar.title("Settings & Model-3 API")
session_key = st.sidebar.text_input("Paste GEMINI_API_KEY (session only)", value="", type="password")
if session_key:
    st.sidebar.success("Using session GEMINI_API_KEY (won't be saved).")
else:
    if os.getenv("GEMINI_API_KEY"):
        st.sidebar.info("Using GEMINI_API_KEY from model3/.env or environment")
    else:
        st.sidebar.warning("No GEMINI_API_KEY found. Paste into field or create model3/.env")

model_override = st.sidebar.text_input("Model-3 model override (optional)", value=os.getenv("GEMINI_MODEL", "")).strip() or None
show_debug = st.sidebar.checkbox("Show debug info", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("You can set GEMINI_API_KEY in model3/.env (read automatically) or paste for session use.")

# ------------------------- Main layout (uploader left, context right) -------------------------
st.markdown("<div class='big-title'>Multi-model AI Medical Diagnosis — Medicube</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Extractor → Model-1 → Model-2 → Model-3. Upload single file or a ZIP of reports for batch testing.</div>", unsafe_allow_html=True)

col_file, col_ctx = st.columns([1,1])  # equal halves; uploader left, context right

# LEFT: File upload + samples
with col_file:
    st.markdown("<div class='card uploader-box'>", unsafe_allow_html=True)
    st.subheader("Input — single file or ZIP")
    st.markdown("Upload a **single** report (PDF / image / JSON) or a **ZIP** containing multiple supported files (pdf/png/jpg/json). ZIP extraction will place files in `samples/`.")
    uploaded = st.file_uploader("Upload file", type=["pdf","png","jpg","jpeg","json","tiff","bmp","zip"], help="Upload a report or a zip of many reports. ZIP is extracted into samples/.", accept_multiple_files=False)
    st.caption("If you upload a ZIP it will be extracted to samples/ for you to choose a sample to run.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    show_all_samples = st.checkbox("Show all files in samples/ (may be large)", value=False)
    samples = sorted([p.name for p in SAMPLES.glob("*") if p.is_file()])
    sample_choice = ""
    if samples:
        if show_all_samples:
            sample_choice = st.selectbox("Choose sample (from samples/)", options=[""] + samples, index=0)
        else:
            recent_samples = sorted([p for p in SAMPLES.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)[:20]
            recent_names = [p.name for p in recent_samples]
            sample_choice = st.selectbox("Choose sample (recent)", options=[""] + recent_names, index=0)
    else:
        st.info("No sample files in samples/. Upload a file or ZIP to populate samples/.")
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT: Context form & run control
context_disabled_for_batch = False
if uploaded and uploaded.name.lower().endswith(".zip"):
    context_disabled_for_batch = True

with col_ctx:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Patient & Lifestyle (optional — improves narrative)")

    if context_disabled_for_batch:
        st.info("Context form is disabled for batch (ZIP) uploads. For batch runs, context is not applied per-file.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if "ctx_saved" not in st.session_state:
            st.session_state.ctx_saved = False
        age_def = int(st.session_state.get("age", 0)) if st.session_state.get("age", None) not in (None, 0) else 0
        gender_def = st.session_state.get("gender", "")
        smoking_def = st.session_state.get("smoking", "")
        alcohol_def = st.session_state.get("alcohol", "")
        activity_def = st.session_state.get("activity", "")
        symptoms_def = st.session_state.get("current_symptoms", "")
        med_notes_def = st.session_state.get("medical_notes", "")

        with st.form("context_form", clear_on_submit=False):
            colA, colB = st.columns(2)
            with colA:
                ctx_age = st.number_input("Age", min_value=0, max_value=120, value=age_def)
                ctx_gender = st.selectbox("Gender", options=["", "Female", "Male", "Other"], index=["", "Female", "Male", "Other"].index(gender_def) if gender_def in ["", "Female", "Male", "Other"] else 0)
                ctx_smoking = st.selectbox("Smoking", options=["", "Never", "Former", "Current", "Occasional"], index=["", "Never", "Former", "Current", "Occasional"].index(smoking_def) if smoking_def in ["", "Never", "Former", "Current", "Occasional"] else 0)
            with colB:
                ctx_alcohol = st.selectbox("Alcohol", options=["", "None", "Occasional", "Moderate", "Regular"], index=["", "None", "Occasional", "Moderate", "Regular"].index(alcohol_def) if alcohol_def in ["", "None", "Occasional", "Moderate", "Regular"] else 0)
                ctx_activity = st.selectbox("Physical activity", options=["", "None", "Low", "Moderate", "High"], index=["", "None", "Low", "Moderate", "High"].index(activity_def) if activity_def in ["", "None", "Low", "Moderate", "High"] else 0)
                ctx_symptoms = st.text_input("Current symptoms (short)", value=symptoms_def)
            ctx_med_notes = st.text_area("Medical notes / history (short)", placeholder="e.g., hypothyroidism, on meds, etc.", value=med_notes_def)
            submitted_ctx = st.form_submit_button("Save context (session)")
            if submitted_ctx:
                st.session_state.ctx_saved = True
                st.session_state.age = int(ctx_age) if ctx_age else 0
                st.session_state.gender = ctx_gender or ""
                st.session_state.smoking = ctx_smoking or ""
                st.session_state.alcohol = ctx_alcohol or ""
                st.session_state.activity = ctx_activity or ""
                st.session_state.current_symptoms = ctx_symptoms or ""
                st.session_state.medical_notes = ctx_med_notes or ""
                st.success("Context saved for this session.")
        st.markdown("---")

        include_details = st.checkbox("Include detailed model outputs in final report (Model-1 / Model-2)", value=False)

        run_btn = st.button("Run full pipeline (Extractor → M1 → M2 → M3)", use_container_width=True)
        st.markdown("### Recent model-3 results")
        recent = sorted(MODEL3_DIR.glob("*.model3.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        recent_names = [p.name for p in recent]
        recent_choice = st.selectbox("Open recent result", options=[""] + recent_names, key="recent_choice_select")
        if st.button("Open selected recent"):
            if recent_choice:
                try:
                    rpath = MODEL3_DIR / recent_choice
                    base = Path(recent_choice).stem  # canonical-ish
                    # resolve actual txt using resolver
                    m3_files = resolve_model3_artifacts(base)
                    if m3_files["txt"] and m3_files["txt"].exists():
                        st.text(m3_files["txt"].read_text(encoding="utf-8"))
                    else:
                        st.warning("TXT report not found.")

                    if show_debug:
                        rec = safe_json_load(rpath)
                        if rec:
                            st.json(rec)

                except Exception as e:
                    st.error("Could not open selected result: " + str(e))
        st.markdown("</div>", unsafe_allow_html=True)

# If context is disabled for batch, show Run button below (single place)
if context_disabled_for_batch:
    include_details = st.checkbox("Include detailed model outputs in final report (Model-1 / Model-2)", value=False, key="include_details_batch")
    run_btn = st.button("Run full pipeline (Extractor → M1 → M2 → Model-3) (Batch mode)", use_container_width=True)
    st.markdown("### Recent model-3 results")
    recent = sorted(MODEL3_DIR.glob("*.model3.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    recent_names = [p.name for p in recent]
    recent_choice = st.selectbox("Open recent result", options=[""] + recent_names, key="recent_choice_2")
    if st.button("Open selected recent (batch)"):
        if recent_choice:
            try:
                rpath = MODEL3_DIR / recent_choice
                rec = safe_json_load(rpath)
                if rec:
                    st.json(rec)
            except Exception as e:
                st.error("Could not open selected result: " + str(e))

# Logs & progress area (full width below)
st.markdown("<div style='height:10px'></div>")
logs_expander = st.expander("Logs & debug output", expanded=False)

# ------------------------- Report generation utility (summarized final report) -------------------------
def summarize_model1_abnormal(model1_path: Optional[Path]) -> List[str]:
    if not model1_path or not model1_path.exists():
        return []
    try:
        df = pd.read_csv(model1_path)
    except Exception:
        return []
    suspect_cols = [c for c in df.columns if c.lower() in ("flag","status","interpretation","abnormal","result")]
    abnormal = []
    if suspect_cols:
        for _, row in df.iterrows():
            for c in suspect_cols:
                val = str(row.get(c, "")).strip().lower()
                if val and val not in ("normal","within range","within_range","-","ok","") and not val.startswith("normal"):
                    pname = row.get('parameter') or row.get('Parameter') or row.get('test') or row.get('name') or row.get(df.columns[0])
                    try:
                        abnormal.append(f"{pname}: {row.get(c)}")
                    except Exception:
                        abnormal.append(f"{pname}: {val}")
                    break
    return abnormal[:12]

def make_markdown_report(base_name: str, model1_path: Optional[Path], model2_obj: Optional[dict], model3_obj: Optional[dict], user_ctx: Optional[dict], include_details: bool = False) -> str:
    lines = []
    lines.append(f"# Summary Report — {base_name}")
    lines.append("")
    if user_ctx:
        lines.append("## Context Summary")
        lines.append(f"{user_ctx.get('context_summary','')}")
        lines.append("")

    lines.append("## Clinician-style Summary")
    res = model3_obj.get('result') if isinstance(model3_obj, dict) and 'result' in model3_obj else (model3_obj or {})
    summary_text = (res.get('summary') if isinstance(res, dict) else "") or "No narrative summary available."
    lines.append(summary_text)
    lines.append("")

    if isinstance(res, dict) and res.get('when_to_consult_doctor'):
        lines.append("**When to consult a clinician:**")
        lines.append(res.get('when_to_consult_doctor'))
        lines.append("")

    if isinstance(res, dict) and res.get('possible_explanations'):
        lines.append("### Possible explanations")
        for e in res.get('possible_explanations', [])[:8]:
            lines.append(f"- {e}")
        lines.append("")
    if isinstance(res, dict) and res.get('lifestyle_guidance'):
        lines.append("### Lifestyle & self-care guidance")
        for e in res.get('lifestyle_guidance', [])[:8]:
            lines.append(f"- {e}")
        lines.append("")

    if model2_obj:
        md = model2_obj
        derived = md.get('derived', {}) if isinstance(md, dict) else {}
        pats = md.get('patterns', {}).get('patterns', {}) if isinstance(md, dict) else {}
        causes = md.get('probable_causes', {}).get('causes', []) if isinstance(md, dict) else []

        if derived:
            lines.append("## Key derived metrics (from lab values)")
            for k, v in list(derived.items())[:8]:
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        if pats:
            lines.append("## Detected pattern signals")
            present = [p for p, info in pats.items() if info.get('present')]
            if present:
                for p in present[:10]:
                    lines.append(f"- {p.replace('_',' ').title()}")
            else:
                lines.append("- No strong pathological pattern detected based on available laboratory data.")
            lines.append("")

        if causes:
            lines.append("## Top probable causes (algorithmic)")
            for c in causes[:6]:
                strength = qualitative_strength(c.get("score"))
                lines.append(
                    f"- {c.get('cause','?')} — {strength} signal "
                    f"({', '.join(c.get('support',[])[:3])})"
                )

            lines.append("")

    abnormal = summarize_model1_abnormal(model1_path)
    lines.append("## Notable abnormal parameters (Model-1 quick view)")
    if abnormal:
        for a in abnormal:
            lines.append(f"- {a}")
    else:
        lines.append(
            "- One or more parameters may be outside optimal range but not severely abnormal; "
            "see detailed parameter table for context."
        )


    if isinstance(res, dict) and res.get('notes'):
        lines.append("## Notes")
        lines.append(res.get('notes'))
        lines.append("")

    lines.append("---")
    lines.append("_Disclaimer: This is an automated, non-diagnostic summary intended for educational/demo use. Always consult a clinician for decisions._")

    if include_details:
        lines.append("")
        lines.append("---")
        lines.append("## Detailed artifacts (compact)")
        lines.append("")
        if model1_path and model1_path.exists():
            try:
                df = pd.read_csv(model1_path)
                drop_cols = [c for c in df.columns if c.lower() in ("filename", "report_id", "source", "unit_note")]
                if drop_cols:
                    df = df.drop(columns=drop_cols, errors="ignore")
                lines.append("### Model-1 — first rows (extracted parameters)")
                lines.append(df.head(6).to_markdown(index=False))
                lines.append("")
            except Exception:
                lines.append("_Could not read Model-1 CSV for details._")
        if model2_obj:
            try:
                lines.append("### Model-2 (compact JSON preview)")
                mini = {
                    'derived': model2_obj.get('derived', {}),
                    'patterns_present': [p for p,info in model2_obj.get('patterns',{}).get('patterns',{}).items() if info.get('present')] if isinstance(model2_obj.get('patterns',{}), dict) else []
                }
                lines.append("```")
                lines.append(json.dumps(mini, indent=2))
                lines.append("```")
                lines.append("")
            except Exception:
                lines.append("_Could not render Model-2 compact preview._")
    return "\n".join(lines)

# ------------------------- Pipeline orchestration (unchanged) -------------------------
if 'run_btn' not in locals():
    run_btn = False

if st.button and 'run_btn' in locals():
    pass

if 'run_btn' in locals() and run_btn:
    pass

# The actual run logic (this matches your previous implementation exactly)
try:
    if run_btn:
        # determine input files to run (single or batch)
        input_candidates: List[Path] = []

        if uploaded:
            dest = SAMPLES / uploaded.name
            if dest.exists():
                stem = dest.stem
                suff = dest.suffix
                ts = int(time.time()*1000)
                dest = SAMPLES / f"{stem}_{ts}{suff}"
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())

            if dest.suffix.lower() == ".zip":
                extracted = extract_zip_to_samples(dest, dest_dir=SAMPLES)
                if not extracted:
                    st.error("ZIP uploaded but no supported files found inside.")
                    raise RuntimeError("zip empty")
                input_candidates.extend(extracted)
                zip_batch_name = dest.stem
                batch_output_dir = BATCH_OUT / zip_batch_name

                batch_output_dir.mkdir(parents=True, exist_ok=True)
                batch_m2_dir = batch_output_dir / "model2"
                batch_m3_dir = batch_output_dir / "model3"
                batch_reports_dir = batch_output_dir / "reports"

                batch_m2_dir.mkdir(parents=True, exist_ok=True)
                batch_m3_dir.mkdir(parents=True, exist_ok=True)
                batch_reports_dir.mkdir(parents=True, exist_ok=True)

                st.success(f"Extracted {len(extracted)} files to samples/ (batch: outputs/batch/{zip_batch_name}/)")
                batch_mode = True
            else:
                input_candidates.append(dest)
                batch_mode = False
        else:
            batch_mode = False

        if sample_choice and sample_choice != "":
            samp = SAMPLES / sample_choice
            if samp.exists():
                if samp not in input_candidates:
                    input_candidates.append(samp)

        if not input_candidates:
            st.error("Please upload a file (or ZIP) or select a sample from samples/.")
            raise RuntimeError("no input")

        user_context = {}
        ctx = {}
        if (not context_disabled_for_batch) and st.session_state.get("ctx_saved"):
            if st.session_state.get("age") not in (None, 0):
                ctx["age"] = int(st.session_state.get("age"))
            if st.session_state.get("gender"):
                ctx["gender"] = st.session_state.get("gender")
            lifestyle = {}
            if st.session_state.get("smoking"):
                lifestyle["smoking"] = st.session_state.get("smoking")
            if st.session_state.get("alcohol"):
                lifestyle["alcohol"] = st.session_state.get("alcohol")
            if st.session_state.get("activity"):
                lifestyle["activity"] = st.session_state.get("activity")
            if lifestyle:
                ctx["lifestyle"] = lifestyle
            if st.session_state.get("current_symptoms"):
                ctx["current_symptoms"] = st.session_state.get("current_symptoms")
            if st.session_state.get("medical_notes"):
                ctx["medical_notes"] = st.session_state.get("medical_notes")

        try:
            if not context_disabled_for_batch:
                if 'ctx_age' in locals() and ctx_age:
                    ctx["age"] = int(ctx_age)
                if 'ctx_gender' in locals() and ctx_gender:
                    ctx["gender"] = ctx_gender
                lifestyle = ctx.get("lifestyle", {})
                if 'ctx_smoking' in locals() and ctx_smoking:
                    lifestyle["smoking"] = ctx_smoking
                if 'ctx_alcohol' in locals() and ctx_alcohol:
                    lifestyle["alcohol"] = ctx_alcohol
                if 'ctx_activity' in locals() and ctx_activity:
                    lifestyle["activity"] = ctx_activity
                if lifestyle:
                    ctx["lifestyle"] = lifestyle
                if 'ctx_symptoms' in locals() and ctx_symptoms:
                    ctx["current_symptoms"] = ctx_symptoms
                if 'ctx_med_notes' in locals() and ctx_med_notes:
                    ctx["medical_notes"] = ctx_med_notes
        except Exception:
            pass

        if ctx:
            user_context = ctx

        def build_context_summary(uctx: Dict[str,Any]) -> str:
            parts = []
            if uctx.get("age") not in (None, 0):
                parts.append(f"Age:{uctx.get('age')}")
            if uctx.get("gender"):
                parts.append(f"Gender:{uctx.get('gender')}")
            life = uctx.get("lifestyle") or {}
            if life.get("smoking"):
                parts.append(f"Smoking:{life.get('smoking')}")
            if uctx.get("current_symptoms"):
                parts.append(f"Symptoms:{uctx.get('current_symptoms')}")
            if uctx.get("medical_notes"):
                parts.append(f"Medical:{uctx.get('medical_notes')}")
            return "; ".join(parts)

        if user_context:
            user_context["context_summary"] = build_context_summary(user_context)
            urg = "routine"
            if user_context.get("current_symptoms"):
                urg = "moderate"
            agev = user_context.get("age")
            if isinstance(agev, int):
                if agev >= 65:
                    urg = "elevated"
                elif agev >= 50 and urg == "routine":
                    urg = "moderate"
            user_context["urgency_hint"] = urg

        extra_env = {}
        if session_key:
            extra_env["GEMINI_API_KEY"] = session_key
        elif os.getenv("GEMINI_API_KEY"):
            extra_env["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

        total = len(input_candidates)
        progress = st.progress(0.0)
        file_status_rows = []
        produced_reports = []

        for idx, file_path in enumerate(input_candidates):
            fname = file_path.name
            status_msg = ""
            m2obj = None
            m3obj = None
            model1_path = None
            try:
                render_progress_card("Step 1 — Extractor (OCR & param extraction)", "running", f"Processing {fname} ...")
                ok, msg = run_extractor_batch(file_path, timeout=600)
                if not ok:
                    render_progress_card("Step 1 — Extractor (OCR & param extraction)", "failed", msg)
                    raise RuntimeError(msg)
                render_progress_card("Step 1 — Extractor (OCR & param extraction)", "success", msg)

                base = file_path.stem
                struct_path = STRUCTURED_DIR / f"{base}.structured.csv"
                waited = 0.0
                while waited < 12 and not struct_path.exists():
                    time.sleep(0.5)
                    waited += 0.5
                if not struct_path.exists():
                    alt = latest_file_in_dir(STRUCTURED_DIR, f"{base}*.structured.csv")
                    if alt:
                        struct_path = alt
                if not struct_path.exists():
                    raise RuntimeError(f"Structured CSV for {fname} not found in outputs/structured_per_report")

                render_progress_card("Step 2 — Model-1 (interpretation & status)", "running", "Converting structured CSV to final per-report model1 CSV (status + notes)")
                model1_path, m1msg = run_model1_adapter_for(base, timeout=300)
                if not model1_path:
                    render_progress_card("Step 2 — Model-1 (interpretation & status)", "failed", m1msg)
                    raise RuntimeError("Model-1 adapter failed: " + m1msg)
                render_progress_card("Step 2 — Model-1 (interpretation & status)", "success", f"{model1_path.name}")

                if not batch_mode:
                    try:
                        with st.expander(f"Model-1 output — {model1_path.name}", expanded=True):
                            try:
                                df_m1 = pd.read_csv(model1_path)
                                drop_cols = [c for c in df_m1.columns if c.lower() in ("filename", "report_id", "source", "unit_note")]
                                if drop_cols:
                                    df_m1 = df_m1.drop(columns=drop_cols, errors="ignore")
                                st.dataframe(df_m1, use_container_width=True)
                                csv_bytes = df_m1.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download Model-1 CSV",
                                    csv_bytes,
                                    file_name=f"{model1_path.name}",
                                    mime="text/csv",
                                    key=f"dl_model1_live_{model1_path.stem}_{int(time.time())}"
                                )
                            except Exception as e:
                                st.error(f"Could not read Model-1 CSV: {e}")
                    except Exception:
                        pass

                render_progress_card("Step 3 — Model-2 (patterns & probable causes)", "running", "Running deterministic reasoning & pattern detection")
                m2obj, m2msg = run_model2(model1_path, timeout=300)
                if batch_mode:
                    src_m2 = MODEL2_DIR / f"{model1_path.stem}.model2.json"
                    if src_m2.exists():
                        shutil.copy(src_m2, batch_m2_dir / src_m2.name)

                if m2obj is None:
                    render_progress_card("Step 3 — Model-2 (patterns & probable causes)", "failed", m2msg)
                    raise RuntimeError("Model-2 failed: " + m2msg)
                render_progress_card("Step 3 — Model-2 (patterns & probable causes)", "success", "Model-2 completed")

                if not batch_mode:
                    try:
                        with st.expander("Model-2 — Patterns & Probable Causes (live)", expanded=True):
                            md = m2obj or {}
                            meta = md.get("metadata", {}) if isinstance(md, dict) else {}
                            st.markdown(f"**Report:** {meta.get('base','-')}")
                            if isinstance(md.get("derived"), dict) and md.get("derived"):
                                st.subheader("Derived metrics")
                                derived = md.get("derived", {})
                                cols = st.columns(3)
                                i = 0
                                for k, v in derived.items():
                                    with cols[i % 3]:
                                        st.markdown(
                                            f"<div class='param-metric'><strong>{k}</strong><div class='small'>{v}</div></div>",
                                            unsafe_allow_html=True
                                        )
                                    i += 1
                            pats = md.get("patterns", {}).get("patterns", {})
                            if pats:
                                st.subheader("Detected patterns")
                                rows = []
                                for pname, pinfo in pats.items():
                                    rows.append({
                                        "Pattern": pname,
                                        "Present": (
                                            "Yes" if pinfo.get("present")
                                            else "Subclinical" if pinfo.get("subclinical")
                                            else "No"
                                        ),

                                        "Severity": pinfo.get("severity") or pinfo.get("type",""),
                                        "Support": ", ".join(pinfo.get("support", [])[:4])
                                    })
                                st.table(pd.DataFrame(rows))
                            causes = md.get("probable_causes", {}).get("causes", [])
                            if causes:
                                st.subheader("Top probable causes")
                                for c in causes[:6]:
                                    strength = qualitative_strength(c.get("score"))
                                    st.write(
                                        f"- **{c.get('cause','?')}** — {strength} signal "
                                        f"({', '.join(c.get('support',[])[:3])})"
                                    )

                            cardio = md.get("contextual_risks", {}).get("cardiovascular", {})
                            if cardio:
                                st.subheader("Cardiovascular risk")
                                st.metric("Risk band", cardio.get("band","N/A"), cardio.get("score",""))
                            conf = md.get("confidence", {})
                            if conf:
                                st.subheader("Confidence")
                                #st.write(f"Score: {conf.get('score')}")
                                band = conf.get("confidence_band", "Moderate")
                                st.write(f"Confidence: **{band}**")

                                if conf.get("explanation"):
                                    st.caption(conf.get("explanation"))
                            if md.get("notes"):
                                st.subheader("Model-2 notes")
                                st.write(md.get("notes"))
                            if show_debug:
                                st.subheader("Raw Model-2 JSON")
                                st.json(md)
                    except Exception:
                        pass

                user_context_path = None
                preserve_ctx = dict(user_context) if user_context else {}
                if user_context and (not batch_mode):
                    try:
                        try:
                            with open(str(model1_path), newline='', encoding='utf-8') as csvfile:
                                reader = csv.DictReader(csvfile)
                                row = next(reader, None)
                                if row:
                                    age_val = None
                                    gender_val = None
                                    for k in row.keys():
                                        lk = k.strip().lower()
                                        if lk == "age" and (row.get(k) or "").strip() != "":
                                            age_val = (row.get(k) or "").strip()
                                        if lk == "gender" and (row.get(k) or "").strip() != "":
                                            gender_val = (row.get(k) or "").strip()
                                    if age_val:
                                        preserve_ctx.pop("age", None)
                                    if gender_val:
                                        preserve_ctx.pop("gender", None)
                        except Exception:
                            pass

                        user_context_path = MODEL3_DIR / f"{model1_path.stem}.user_context.json"
                        user_context_path.write_text(json.dumps(preserve_ctx, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        user_context_path = None

                if "GEMINI_API_KEY" not in extra_env:
                    render_progress_card("Step 4 — Model-3 (narrative)", "failed", "GEMINI_API_KEY not provided. Paste key in sidebar or set model3/.env")
                    raise RuntimeError("GEMINI_API_KEY not provided")

                render_progress_card("Step 4 — Model-3 (narrative)", "running", "Generating final narrative")
                m2json = OUT / "model2_outputs" / (model1_path.stem + ".model2.json")
                if not m2json.exists():
                    alt = latest_file_in_dir(OUT / "model2_outputs", "*.model2.json")
                    if alt:
                        m2json = alt
                if not m2json.exists():
                    raise RuntimeError("Model-2 JSON missing for Model-3 input")

                m3obj, m3msg = run_model3(m2json, model_name=model_override, timeout=900, extra_env=extra_env, user_context_path=user_context_path)
                if m3obj is None:
                    render_progress_card("Step 4 — Model-3 (narrative)", "failed", m3msg)
                    raise RuntimeError("Model-3 failed: " + m3msg)
                render_progress_card("Step 4 — Model-3 (narrative)", "success", "Narrative generated")

                canonical_base = model1_path.stem
                md_content = make_markdown_report(canonical_base, model1_path, m2obj, m3obj, preserve_ctx if preserve_ctx else None, include_details=include_details)

                if batch_mode:
                    outdir = batch_output_dir
                    md_path = outdir / f"{canonical_base}.report.md"
                else:
                    outdir = MODEL3_DIR
                    md_path = outdir / f"{canonical_base}.report.md"
                md_path.write_text(md_content, encoding="utf-8")
                produced_reports.append(md_path)

                m3_files = resolve_model3_artifacts(canonical_base)
                if batch_mode:
                    if m3_files.get("json") and m3_files["json"].exists():
                        shutil.copy(m3_files["json"], batch_m3_dir / m3_files["json"].name)

                    if m3_files.get("txt") and m3_files["txt"].exists():
                        shutil.copy(m3_files["txt"], batch_m3_dir / m3_files["txt"].name)


                report_id = file_path.stem
                manifest = {
                    "report_id": report_id,
                    "original_filename": file_path.name,
                    "canonical_base": canonical_base,
                    "artifacts": {
                        "model1_csv": str(model1_path),
                        "model2_json": (
                            str(batch_m2_dir / f"{canonical_base}.model2.json")
                            if batch_mode else
                            str(MODEL2_DIR / f"{canonical_base}.model2.json")
                        ),

                        "model3_json": (
                            str(batch_m3_dir / m3_files["json"].name)
                            if batch_mode and m3_files["json"] else
                            str(m3_files["json"]) if m3_files["json"] else None
                        ),
                        "model3_txt": (
                            str(batch_m3_dir / m3_files["txt"].name)
                            if batch_mode and m3_files["txt"] else
                            str(m3_files["txt"]) if m3_files["txt"] else None
                        ),
                        "final_report_md": str(md_path)

                    },
                   "confidence_band": (
                        m2obj.get("confidence", {}).get("confidence_band")
                        if isinstance(m2obj, dict) else "Moderate"
                    ),

                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }

                manifest_path_orig = MANIFEST_DIR / f"{report_id}.json"
                manifest_path_canon = MANIFEST_DIR / f"{canonical_base}.json"
                manifest_path_orig.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                manifest_path_canon.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                st.session_state.last_manifest = manifest
                st.session_state.last_canonical_base = canonical_base
                st.session_state.pipeline_completed = True

                final_txt_str = manifest["artifacts"].get("model3_txt")
                final_txt = Path(final_txt_str) if final_txt_str else None

                batch_dir = None
                if uploaded and uploaded.name.endswith(".zip"):
                    batch_dir = BATCH_OUT / Path(uploaded.name).stem
                    batch_dir.mkdir(parents=True, exist_ok=True)
                elif uploaded and not uploaded.name.endswith(".zip") and sample_choice == "":
                    batch_dir = None

                if batch_dir and final_txt and final_txt.exists():
                    target_name = f"{file_path.stem}.model3.txt"
                    shutil.copy(final_txt, batch_dir / target_name)

                if not batch_mode:
                    try:
                        with st.expander("Outputs & downloads (generated)", expanded=True):
                            m3txt_path = Path(manifest["artifacts"]["model3_txt"]) if manifest["artifacts"].get("model3_txt") else None
                            m3json_path = Path(manifest["artifacts"]["model3_json"]) if manifest["artifacts"].get("model3_json") else None
                            md_file_path = Path(manifest["artifacts"]["final_report_md"]) if manifest["artifacts"].get("final_report_md") else None

                            if m3txt_path and m3txt_path.exists():
                                st.download_button(
                                    "Download Model-3 TXT",
                                    m3txt_path.read_bytes(),
                                    file_name=m3txt_path.name,
                                    mime="text/plain",
                                    key=f"dl_model3_txt_{canonical_base}_{int(time.time())}"
                                )
                            else:
                                st.warning("Model-3 TXT not found for direct download.")

                            if m3json_path and m3json_path.exists():
                                st.download_button(
                                    "Download Model-3 JSON",
                                    m3json_path.read_bytes(),
                                    file_name=m3json_path.name,
                                    mime="application/json",
                                    key=f"dl_model3_json_{canonical_base}_{int(time.time())}"
                                )

                            if md_file_path and md_file_path.exists():
                                st.download_button(
                                    "Download Final Report (Markdown)",
                                    md_file_path.read_bytes(),
                                    file_name=md_file_path.name,
                                    mime="text/markdown",
                                    key=f"dl_final_report_{canonical_base}_{int(time.time())}"
                                )
                    except Exception:
                        pass

                # conf_score = None
                # try:
                #     conf_score = m2obj.get("confidence", {}).get("score")
                # except Exception:
                #     conf_score = None
                file_status_rows.append({
                    "file": fname,
                    "status": "success",
                    "message": "",
                    "confidence": m2obj.get("confidence", {}).get("confidence_band", "Moderate")
                })

            except Exception as e_file:
                msg = str(e_file)
                file_status_rows.append({"file": fname, "status": "failed", "message": msg, "confidence": None})
                with logs_expander:
                    st.error(f"{fname} failed: {msg}")
                    if show_debug:
                        st.exception(e_file)
                        st.code(traceback.format_exc())
            finally:
                progress.progress((idx + 1)/total)

        st.success("Processing finished for all input files.")
        try:
            df_summary = pd.DataFrame(file_status_rows)
            st.subheader("Batch / Run summary")
            st.table(df_summary)
        except Exception:
            pass

        if batch_mode and produced_reports:
            st.info(f"Batch artifacts (final model3 .txt and .md) saved at: outputs/batch/{Path(uploaded.name).stem}/")
            st.markdown("### Produced reports (batch)")
            for p in produced_reports:
                st.write(p.name)

except Exception as e:
    st.error("Pipeline failed: " + str(e))
    if show_debug:
        st.exception(e)
        st.code(traceback.format_exc())

# =====================================================
# Persistent Final Results Renderer (Session-Safe) + Chatbot integration
# =====================================================
def _make_chat_history_key(report_id: str) -> str:
    return f"chat_history_{report_id}"

def render_chatbot_panel(manifest: dict):
    """
    Render Chatbot UI bound to a given manifest.
    Uses chatbot.chatbot_runner.ChatbotRunner when available.
    """
    try:
        # Ensure project root importable (so 'chatbot' is a package)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        # Explicitly load chatbot/.env (already loaded above but safe to re-load)
        chatbot_env = ROOT / "chatbot" / ".env"
        if chatbot_env.exists():
            try:
                load_dotenv(dotenv_path=str(chatbot_env), override=False)
            except Exception:
                pass

        # Import ChatbotRunner lazily and create a runner
        from chatbot.chatbot_runner import ChatbotRunner  # type: ignore
    except Exception as e:
        st.warning("Chatbot module not available. Skipping Chatbot UI. Reason: " + str(e))
        if show_debug:
            st.exception(e)
        return



    # Instantiate runner (index dir placed under chatbot/vectorstore to avoid cross-root conflicts)
    index_dir_candidate = ROOT / "chatbot" / "vectorstore"
    index_dir_arg = str(index_dir_candidate) if index_dir_candidate.exists() else None
    try:
        runner = ChatbotRunner(index_dir=index_dir_arg)
    except Exception as e:
        st.warning("Failed to initialize ChatbotRunner: " + str(e))
        if show_debug:
            st.exception(e)
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>Medicube Assistant — Report Q&A</div>", unsafe_allow_html=True)
    st.markdown(f"**Report:** {manifest.get('report_id')} — Canonical: {manifest.get('canonical_base')}")
    st.markdown("Artifacts:")
    for k, v in manifest.get("artifacts", {}).items():
        st.markdown(f"- {k}: `{v}`")

    # session-scoped history per canonical_base
    report_key = manifest.get("canonical_base") or manifest.get("report_id") or "default"
    hist_key = _make_chat_history_key(report_key)
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    # Chat UI (use form to avoid rerun issues)
    with st.form(f"chat_form_{report_key}", clear_on_submit=False):
        question = st.text_input("Ask a question about this report (within scope):", key=f"chat_input_{report_key}")
        submitted = st.form_submit_button("Send")
        if submitted and question:
            # append user message
            st.session_state[hist_key].append({"role": "user", "content": question})
            # generate answer
            with st.spinner("Generating answer..."):
                try:
                    answer, meta = runner.answer(question, manifest, st.session_state[hist_key])
                    st.session_state[hist_key].append({"role": "assistant", "content": answer})
                except Exception as e:
                    # show fallback message and log
                    st.session_state[hist_key].append({"role": "assistant", "content": "[Chatbot error — fallback used] " + str(e)})
                    if show_debug:
                        st.exception(e)

    # Display chat history
    if st.session_state.get(hist_key):
        for turn in st.session_state[hist_key]:
            if turn["role"] == "user":
                st.markdown(f"**You:** {turn['content']}")
            else:
                st.markdown(f"**Medicube:** {turn['content']}")
    else:
        st.info("No messages yet. Ask something about the selected report.")

    st.markdown("</div>", unsafe_allow_html=True)

try:
    if st.session_state.pipeline_completed and st.session_state.last_manifest:
        manifest = st.session_state.last_manifest
        canonical_base = st.session_state.last_canonical_base or manifest.get("canonical_base", "")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Results — {canonical_base}</div>", unsafe_allow_html=True)

        # Model-1
        st.markdown("**Model-1 — Parameter Interpretation**")
        with st.expander("View Model-1 parameter table", expanded=False):
            model1_csv = Path(manifest["artifacts"].get("model1_csv")) if manifest.get("artifacts", {}).get("model1_csv") else None
            if model1_csv and model1_csv.exists():
                try:
                    df_m1 = pd.read_csv(model1_csv)
                    drop_cols = [c for c in df_m1.columns if c.lower() in ("filename", "report_id", "source", "unit_note")]
                    if drop_cols:
                        df_m1 = df_m1.drop(columns=drop_cols, errors="ignore")
                    st.dataframe(df_m1, use_container_width=True)
                    csv_bytes = df_m1.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Model-1 CSV",
                        csv_bytes,
                        file_name=f"{model1_csv.name}",
                        mime="text/csv",
                        key=f"dl_model1_final_{model1_csv.stem}"
                    )
                except Exception as e:
                    st.error(f"Could not read Model-1 CSV: {e}")
            else:
                st.error(f"Model-1 CSV missing at: {model1_csv}")

        # Model-2
        st.markdown("**Model-2 — Patterns & Probable Causes**")
        with st.expander("Model-2 summary (detailed)", expanded=False):
            md_path_str = manifest.get("artifacts", {}).get("model2_json")
            md = safe_json_load(Path(md_path_str)) if md_path_str else {}
            meta = md.get("metadata", {}) if isinstance(md, dict) else {}
            st.markdown(f"**Report:** {meta.get('base','-')}")

            if isinstance(md.get("derived"), dict) and md.get("derived"):
                st.subheader("Derived metrics")
                derived = md.get("derived", {})
                cols = st.columns(3)
                i = 0
                for k, v in derived.items():
                    with cols[i % 3]:
                        st.markdown(
                            f"<div class='param-metric'><strong>{k}</strong><div class='small'>{v}</div></div>",
                            unsafe_allow_html=True
                        )
                    i += 1

            pats = md.get("patterns", {}).get("patterns", {})
            if pats:
                st.subheader("Detected patterns")
                rows = []
                for pname, pinfo in pats.items():
                    rows.append({
                        "Pattern": pname,
                        "Present": (
                            "Yes" if pinfo.get("present")
                            else "Subclinical" if pinfo.get("subclinical")
                            else "No"
                        ),

                        "Severity": pinfo.get("severity") or pinfo.get("type",""),
                        "Support": ", ".join(pinfo.get("support", [])[:4])
                    })
                st.table(pd.DataFrame(rows))

            causes = md.get("probable_causes", {}).get("causes", [])
            if causes:
                st.subheader("Top probable causes")
                for c in causes[:6]:
                    strength = qualitative_strength(c.get("score"))
                    st.write(
                        f"- **{c.get('cause','?')}** — {strength} signal "
                        f"({', '.join(c.get('support',[])[:3])})"
                    )


            conf = md.get("confidence", {})
            if conf:
                st.subheader("Confidence")
                #st.write(f"Score: {conf.get('score')}")
                band = conf.get("confidence_band", "Moderate")
                st.write(f"Confidence: **{band}**")

                if conf.get("explanation"):
                    st.caption(conf.get("explanation"))

            if show_debug:
                st.subheader("Raw Model-2 JSON")
                st.json(md)

        # Model-3 narrative TXT
        st.markdown("**Model-3 Narrative (TXT)**")
        m3txt_path_str = manifest.get("artifacts", {}).get("model3_txt")
        m3txt_path = Path(m3txt_path_str) if m3txt_path_str else None
        if m3txt_path and m3txt_path.exists():
            with st.expander("View Model-3 narrative text", expanded=False):
                st.text(m3txt_path.read_text(encoding="utf-8"))
        else:
            st.warning("Model-3 TXT narrative not found.")

        st.markdown("**Final Medical Summary (combined & concise)**")
        final_md_path_str = manifest.get("artifacts", {}).get("final_report_md")
        final_md_path = Path(final_md_path_str) if final_md_path_str else None
        if not final_md_path or not final_md_path.exists():
            st.error(f"Final report missing at {final_md_path}")
        else:
            md_content = final_md_path.read_text(encoding="utf-8")
            st.markdown(md_content, unsafe_allow_html=True)
            st.download_button(
                "Download report (Markdown)",
                md_content,
                file_name=f"{canonical_base}.report.md",
                mime="text/markdown",
                key=f"dl_final_report_bottom_{canonical_base}"
            )

        # ----------------- Chatbot panel integration (renders under results) -----------------
        try:
            render_chatbot_panel(manifest)
        except Exception as e:
            if show_debug:
                st.exception(e)

        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error("Error rendering persistent results: " + str(e))
    if show_debug:
        st.exception(e)
        st.code(traceback.format_exc())
