# app.py — Multi-model AI Medical Diagnosis (improved UI + user-context + robust pipeline)
# Place this in your project root. Works with existing repo layout:
# extractor/* , model2/* , model3/* and outputs/ (structured_per_report, model1_per_report, model2_outputs, model3)

import streamlit as st
from pathlib import Path
import sys
import subprocess
import time
import json
import traceback
import os
import tempfile
from typing import Optional, Tuple, Any, Dict

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"
STRUCTURED_DIR = OUT / "structured_per_report"
MODEL1_DIR = OUT / "model1_per_report"
MODEL2_DIR = OUT / "model2_outputs"
MODEL3_DIR = OUT / "model3"

# ensure directories exist
for d in (SAMPLES, OUT, STRUCTURED_DIR, MODEL1_DIR, MODEL2_DIR, MODEL3_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(page_title="Multi-model AI Medical Diagnosis", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Small CSS for nicer UI
# -------------------------
st.markdown(
    """
    <style>
      /* page */
      .big-title { font-size:34px; font-weight:800; margin-bottom:4px; color: #ffffff; }
      .muted { color: #9fb2c8; margin-bottom:16px; }
      .card { background: linear-gradient(180deg,#051021,#0b1220); border-radius:10px; padding:14px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); margin-bottom:12px; }
      .small { font-size:13px; color:#9fb2c8; }
      .progress-badge { font-weight:800; padding:6px 10px; border-radius:8px; color:#fff; }
      .running { background:#f59e0b; }
      .success { background:#16a34a; }
      .failed { background:#ef4444; }
      .pending { background:#6b7280; }
      .left-panel { padding-right: 24px; }
      .param-metric { background: #081021; padding: 10px; border-radius:8px; color:#e6eef8; }
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# Helpers
# -------------------------
def run_subprocess(cmd: list, cwd: Optional[Path] = None, timeout: Optional[int] = None, env: Optional[Dict[str,str]] = None) -> Tuple[int, str, str]:
    """
    Run command. If first element is a .py script path, invoke with current interpreter.
    Accepts env dict to pass to Popen (merge of os.environ by caller recommended).
    Returns (rc, stdout, stderr).
    """
    if not cmd:
        return 1, "", "empty command"
    exec_cmd = cmd
    first = Path(cmd[0])
    if first.suffix == ".py":
        exec_cmd = [sys.executable] + cmd
    # ensure we have an environment dict
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
    """Small visually consistent progress card."""
    cls = {"running": "running", "success": "success", "failed": "failed", "pending": "pending"}.get(status, "pending")
    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:700; font-size:15px; color:#e6eef8;">{title}</div>
            <div class="progress-badge {cls}">{status.capitalize()}</div>
          </div>
          <div style="margin-top:8px; color:#9fb2c8; font-size:13px">{detail}</div>
        </div>
        """, unsafe_allow_html=True
    )

# -------------------------
# Pipeline helpers (same behavior as your previous file)
# -------------------------
def run_extractor_batch(input_sample_path: Path, timeout: int = 600) -> Tuple[bool, str]:
    """
    Prefer a single-file extractor entrypoint if present (process_file.py).
    If not available, fall back to run_batch_model1.py but ALWAYS invoke with the input file argument.
    Do NOT execute the no-arg fallback that causes full-folder batch processing from the UI.
    """
    extractor_candidates = [
        ROOT / "extractor" / "process_file.py",       # preferred: single-file runner (must accept a path)
        ROOT / "extractor" / "run_batch_model1.py",   # fallback: batch runner (we will pass the file arg)
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

    # Build environment so subprocess Python can import local packages
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # Always pass the single-file path as argument. Avoid any automatic no-arg fallback.
    rc, out, err = run_subprocess([str(found), str(input_sample_path)], cwd=ROOT, timeout=timeout, env=env)
    if rc != 0:
        # Provide clear diagnostic; do not try a no-arg fallback from UI
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

    # adapter may expect no args (process all structured files). Keep behavior as-is.
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
    """
    Run model2 as a module to retain package-relative imports.
    Ensure PYTHONPATH includes ROOT so sibling packages (pipeline, etc.) are importable.
    """
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

# -------------------------
# Load model3/.env if present
# -------------------------
model3_env_file = ROOT / "model3" / ".env"
if model3_env_file.exists():
    try:
        load_dotenv(dotenv_path=str(model3_env_file))
    except Exception:
        pass

# -------------------------
# Utility: ensure Model-3 returns required structure
# -------------------------
def ensure_model3_keys(result: dict) -> dict:
    """
    Ensure the LLM output has required keys and conservative fallbacks.
    Also prioritizes 'when_to_consult_doctor' if severity flags exist.
    This is conservative guidance only (not medical advice).
    """
    r = dict(result) if isinstance(result, dict) else {}
    # Required arrays
    r.setdefault("summary", "")
    r.setdefault("possible_explanations", [])
    r.setdefault("lifestyle_guidance", [])
    r.setdefault("notes", "")

    # when_to_consult_doctor: ensure present and conservative
    w = r.get("when_to_consult_doctor")
    if not w or not isinstance(w, str) or len(w.strip()) == 0:
        # generate conservative fallback based on candidate signals in notes/explanations
        severity_flag = False
        # if any explanation or notes contain words like "severe" or "very" or "urgent", escalate wording
        combined_text = " ".join([json.dumps(r.get("possible_explanations","")), str(r.get("notes","")), r.get("summary","")]).lower()
        if any(k in combined_text for k in ("severe", "very_severe", "very severe", "severe_high", "severe_low", "urgent")):
            severity_flag = True

        if severity_flag:
            r["when_to_consult_doctor"] = "Consider contacting a clinician promptly (within 24–48 hours) for clinical correlation."
        else:
            # default conservative phrasing
            r["when_to_consult_doctor"] = ("Discuss results with a clinician for interpretation and follow-up if you have symptoms, "
                                           "if values persist on repeat testing, or if you are concerned.")
    return r

# -------------------------
# Sidebar: API, model override, debug
# -------------------------
st.sidebar.title("Settings & Model-3 API")
session_key = st.sidebar.text_input("Paste GEMINI_API_KEY (session only)", value="", type="password")
if session_key:
    st.sidebar.success("Using session GEMINI_API_KEY (won't be saved to disk).")
else:
    if os.getenv("GEMINI_API_KEY"):
        st.sidebar.info("Using GEMINI_API_KEY from model3/.env or environment")
    else:
        st.sidebar.warning("No GEMINI_API_KEY found. Paste into the field above or create model3/.env")

model_override = st.sidebar.text_input("Model-3 model override (optional)", value=os.getenv("GEMINI_MODEL", "")).strip() or None
show_debug = st.sidebar.checkbox("Show debug info", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("You can set GEMINI_API_KEY in model3/.env (read automatically) or paste for session use.")

# -------------------------
# Main layout
# -------------------------
st.markdown("<div class='big-title'>Multi-model AI Medical Diagnosis</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Medicube pipeline — Extractor → Model-1 → Model-2 → Model-3. Professional summaries, conservative narratives, and downloadable artifacts.</div>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.2, 2])

# Left: input + context
with left_col:
    st.markdown("<div class='card left-panel'>", unsafe_allow_html=True)
    st.subheader("Input")
    uploaded = st.file_uploader("Upload a blood report (PDF / image / JSON)", type=["pdf", "png", "jpg", "jpeg", "json", "tiff", "bmp"])
    st.caption("Limit 200MB. Or pick a file already in samples/ for quick runs.")
    samples = sorted([p.name for p in SAMPLES.glob("*.*")])
    sample_choice = st.selectbox("Choose sample (optional)", options=[""] + samples)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.subheader("Patient & Lifestyle (optional — improves narrative)")
    # use session_state to keep values across runs
    if "ctx_saved" not in st.session_state:
        st.session_state.ctx_saved = False
    with st.form("context_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            ctx_age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("age", 0))
            ctx_gender = st.selectbox("Gender", options=["", "Female", "Male", "Other"], index=0)
            ctx_smoking = st.selectbox("Smoking", options=["", "Never", "Former", "Current", "Occasional"], index=0)
        with col2:
            ctx_alcohol = st.selectbox("Alcohol", options=["", "None", "Occasional", "Moderate", "Regular"], index=0)
            ctx_activity = st.selectbox("Physical activity", options=["", "None", "Low", "Moderate", "High"], index=0)
            ctx_symptoms = st.text_input("Current symptoms (short)", value=st.session_state.get("current_symptoms",""))
        ctx_med_notes = st.text_area("Medical notes / history (short)", placeholder="e.g., hypothyroidism, on meds, etc.", value=st.session_state.get("medical_notes",""))
        submitted_ctx = st.form_submit_button("Save context (session)")
        if submitted_ctx:
            st.session_state.ctx_saved = True
            st.session_state.age = int(ctx_age) if ctx_age else 0
            st.session_state.current_symptoms = ctx_symptoms or ""
            st.session_state.medical_notes = ctx_med_notes or ""
            st.success("Context saved for this session.")

    st.markdown("---")
    run_btn = st.button("Run full pipeline", use_container_width=True)

    st.markdown("### Recent results")
    recent = sorted(MODEL3_DIR.glob("*.model3.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    recent_names = [p.name for p in recent]
    recent_choice = st.selectbox("Open recent result", options=[""] + recent_names, key="recent_choice")
    if st.button("Open selected recent"):
        if recent_choice:
            try:
                rpath = MODEL3_DIR / recent_choice
                rec = safe_json_load(rpath)
                if rec:
                    right_col.subheader("Selected Result (raw JSON)")
                    right_col.json(rec)
            except Exception as e:
                st.error("Could not open selected result: " + str(e))

# Right: run progress + logs + outputs
with right_col:
    st.subheader("Run progress")
    step1_ph = st.empty()
    step2_ph = st.empty()
    step3_ph = st.empty()
    step4_ph = st.empty()
    logs_expander = st.expander("Logs & debug output", expanded=False)

# -------------------------
# Pipeline orchestration
# -------------------------
if run_btn:
    try:
        # prepare input path
        if uploaded:
            dest = SAMPLES / uploaded.name
            if dest.exists():
                suffix = int(time.time())
                dest = SAMPLES / f"{Path(uploaded.name).stem}_{suffix}{Path(uploaded.name).suffix}"
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            input_path = dest
        elif sample_choice:
            input_path = SAMPLES / sample_choice
            if not input_path.exists():
                st.error("Selected sample not found in samples/")
                raise FileNotFoundError("sample missing")
        else:
            st.warning("Please upload a file or select a sample.")
            raise RuntimeError("no input")

        # Step 1 — extractor
        render_progress_card("Step 1 — Extractor (OCR & param extraction)", "running", "Running batch extractor; it will write structured CSV(s) into outputs/structured_per_report")
        step1_ph.markdown("**Extractor:** running...")
        ok, msg = run_extractor_batch(input_path, timeout=600)
        if not ok:
            render_progress_card("Step 1 — Extractor (OCR & param extraction)", "failed", msg)
            with logs_expander:
                st.text(msg)
                if show_debug:
                    st.code(traceback.format_exc())
            raise RuntimeError("Extractor failed: " + msg)
        render_progress_card("Step 1 — Extractor (OCR & param extraction)", "success", msg)

        # locate structured csv (wait briefly)
        base = input_path.stem
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
            render_progress_card("Step 1 — Extractor (OCR & param extraction)", "failed", f"Extractor finished but could not find structured CSV for {base} under outputs/structured_per_report")
            with logs_expander:
                st.warning("Structured CSV not found. Check extractor logs and outputs/structured_per_report.")
            raise RuntimeError("structured CSV missing")

        # Step 2 — Model-1 adapter
        render_progress_card("Step 2 — Model-1 (interpretation & status)", "running", "Converting structured CSV to final per-report model1 CSV (status + notes)")
        model1_path, m1msg = run_model1_adapter_for(base, timeout=300)
        if not model1_path:
            render_progress_card("Step 2 — Model-1 (interpretation & status)", "failed", m1msg)
            with logs_expander:
                st.text(m1msg)
            raise RuntimeError("Model-1 adapter failed: " + m1msg)
        render_progress_card("Step 2 — Model-1 (interpretation & status)", "success", f"{model1_path.name}")

        # model-1 preview
        try:
            with st.expander("Model-1 preview (first 8 rows)"):
                df = pd.read_csv(model1_path)
                st.dataframe(df.head(8), use_container_width=True)
        except Exception as e:
            with logs_expander:
                st.warning("Could not preview Model-1 CSV: " + str(e))

        # Step 3 — Model-2
        render_progress_card("Step 3 — Model-2 (patterns & probable causes)", "running", "Running deterministic reasoning & pattern detection (file-based)")
        m2obj, m2msg = run_model2(model1_path, timeout=300)
        if m2obj is None:
            render_progress_card("Step 3 — Model-2 (patterns & probable causes)", "failed", m2msg)
            with logs_expander:
                st.text(m2msg)
            raise RuntimeError("Model-2 failed: " + m2msg)
        render_progress_card("Step 3 — Model-2 (patterns & probable causes)", "success", "Model-2 completed")

        # Render Model-2 summary
        with st.expander("Model-2 summary (detailed)"):
            try:
                md = m2obj or {}
                meta = md.get("metadata", {}) if isinstance(md, dict) else {}
                st.markdown(f"**Report:** {meta.get('base','-')}")
                # derived metrics (nicely)
                if isinstance(md.get("derived"), dict) and md.get("derived"):
                    st.subheader("Derived metrics")
                    derived = md.get("derived", {})
                    cols = st.columns(3)
                    i = 0
                    for k, v in derived.items():
                        with cols[i % 3]:
                            st.markdown(f"<div class='param-metric'><strong>{k}</strong><div class='small'>{v}</div></div>", unsafe_allow_html=True)
                        i += 1

                # patterns
                pats = md.get("patterns", {}).get("patterns", {}) if isinstance(md.get("patterns"), dict) else {}
                if pats:
                    st.subheader("Detected patterns")
                    rows = []
                    for pname, pinfo in pats.items():
                        present = "Yes" if bool(pinfo.get("present")) else "No"
                        severity = pinfo.get("severity") or pinfo.get("type") or ""
                        support = ", ".join(pinfo.get("support", [])[:4]) if isinstance(pinfo.get("support", []), list) else ""
                        rows.append({"Pattern": pname, "Present": present, "Severity/Type": severity, "Support (examples)": support})
                    st.table(pd.DataFrame(rows))

                causes = md.get("probable_causes", {}).get("causes", []) if isinstance(md.get("probable_causes"), dict) else md.get("probable_causes", [])
                if causes:
                    st.subheader("Top probable causes")
                    for c in causes[:8]:
                        if isinstance(c, dict):
                            st.write(f"- **{c.get('cause','?')}** (score {c.get('score')}) — support: {', '.join(c.get('support',[])[:3])}")
                        else:
                            st.write(f"- {c}")

                cardio = md.get("cardio", {})
                if cardio:
                    st.subheader("Cardiovascular risk")
                    st.metric("Risk band", cardio.get("band", "N/A"), f"score {cardio.get('score','')}")

                conf = md.get("confidence", {})
                if conf:
                    st.subheader("Confidence")
                    st.write(f"Score: {conf.get('score')}")
                    if conf.get("explanation"):
                        st.caption(conf.get("explanation"))

                if md.get("notes"):
                    st.subheader("Model-2 notes")
                    st.write(md.get("notes"))

                if show_debug:
                    st.subheader("Raw Model-2 JSON")
                    st.json(md)
            except Exception as e:
                st.write("Could not render Model-2 summary cleanly; raw object:")
                st.write(m2obj)

        # Step 4 — Model-3: prepare user_context JSON and GEMINI env
        render_progress_card("Step 4 — Model-3 (narrative)", "running", "Generating final narrative (LLM; deterministic fallback if LLM fails)")

        m2json = OUT / "model2_outputs" / (model1_path.stem + ".model2.json")
        if not m2json.exists():
            alt = latest_file_in_dir(OUT / "model2_outputs", "*.model2.json")
            if alt:
                m2json = alt
        if not m2json.exists():
            render_progress_card("Step 4 — Model-3 (narrative)", "failed", "Model-2 JSON not found for Model-3 input")
            with logs_expander:
                st.error("Model-3 aborted: Model-2 JSON missing.")
            raise RuntimeError("Model-2 JSON missing")

        # build user_context dict from saved session or current form
        user_context = {}
        ctx = {}
        if st.session_state.get("ctx_saved"):
            # use session values
            if st.session_state.get("age"):
                ctx["age"] = int(st.session_state.get("age"))
            if st.session_state.get("medical_notes"):
                ctx["medical_notes"] = st.session_state.get("medical_notes")
            if st.session_state.get("current_symptoms"):
                ctx["current_symptoms"] = st.session_state.get("current_symptoms")
        # also prefer the most recent filled form values (if present)
        # The variables ctx_age, ctx_gender, etc exist only inside the form's scope — retrieve from session_state or fallback
        try:
            # safe extraction: if user submitted new context (ctx_saved true) we already have in session_state above
            if 'ctx_age' in locals() and ctx_age:
                ctx["age"] = int(ctx_age)
        except Exception:
            pass

        # recontruct lifestyle from the immediate UI form values if available
        lifestyle = {}
        try:
            if 'ctx_smoking' in locals() and ctx_smoking:
                lifestyle["smoking"] = ctx_smoking
            if 'ctx_alcohol' in locals() and ctx_alcohol:
                lifestyle["alcohol"] = ctx_alcohol
            if 'ctx_activity' in locals() and ctx_activity:
                lifestyle["activity"] = ctx_activity
        except Exception:
            pass

        if lifestyle:
            ctx["lifestyle"] = lifestyle
        # include inline notes/symptoms if present
        try:
            if 'ctx_symptoms' in locals() and ctx_symptoms:
                ctx["current_symptoms"] = ctx_symptoms
            if 'ctx_med_notes' in locals() and ctx_med_notes:
                ctx["medical_notes"] = ctx_med_notes
        except Exception:
            pass

        if ctx:
            user_context = ctx

        # write user_context to MODEL3_DIR as file (auditable)
        user_context_path = None
        if user_context:
            try:
                user_context_path = MODEL3_DIR / f"{model1_path.stem}.user_context.json"
                user_context_path.write_text(json.dumps(user_context, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                user_context_path = None

        # GEMINI key env
        extra_env = {}
        if session_key:
            extra_env["GEMINI_API_KEY"] = session_key
        else:
            if os.getenv("GEMINI_API_KEY"):
                extra_env["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

        if "GEMINI_API_KEY" not in extra_env:
            render_progress_card("Step 4 — Model-3 (narrative)", "failed", "GEMINI_API_KEY not provided. Paste key in sidebar or place it in model3/.env")
            with logs_expander:
                st.error("Model-3 aborted: GEMINI_API_KEY not provided (session or model3/.env).")
            raise RuntimeError("GEMINI_API_KEY not provided")

        # call model3
        m3obj, m3msg = run_model3(m2json, model_name=model_override, timeout=900, extra_env=extra_env, user_context_path=user_context_path)
        if m3obj is None:
            render_progress_card("Step 4 — Model-3 (narrative)", "failed", m3msg)
            with logs_expander:
                st.text(m3msg)
            raise RuntimeError("Model-3 failed: " + m3msg)
        render_progress_card("Step 4 — Model-3 (narrative)", "success", "Narrative generated")

        # Render Model-3 narrative and ensure minimal keys exist
        with st.expander("Final Narrative & Explanation", expanded=True):
            try:
                candidate = m3obj.get("result") if isinstance(m3obj, dict) and "result" in m3obj else (m3obj if isinstance(m3obj, dict) else {})
                candidate = ensure_model3_keys(candidate)
                st.subheader("Summary")
                st.write(candidate.get("summary", ""))

                if candidate.get("possible_explanations"):
                    st.subheader("Possible explanations")
                    for e in candidate.get("possible_explanations", []):
                        st.write(f"- {e}")

                if candidate.get("lifestyle_guidance"):
                    st.subheader("Lifestyle guidance")
                    for g in candidate.get("lifestyle_guidance", []):
                        st.write(f"- {g}")

                if candidate.get("when_to_consult_doctor"):
                    st.subheader("When to consult a doctor")
                    st.write(candidate.get("when_to_consult_doctor"))

                if candidate.get("notes"):
                    st.subheader("Notes")
                    st.write(candidate.get("notes"))

                # show meta if present
                if isinstance(m3obj, dict) and m3obj.get("meta"):
                    st.subheader("Model metadata")
                    st.json(m3obj.get("meta"))

                
                if show_debug:
                    st.subheader("Raw Model-3 JSON")
                    st.json(m3obj)
            except Exception as e:
                st.write("Could not render Model-3 result cleanly; raw output:")
                st.write(m3obj)
                with logs_expander:
                    st.exception(e)
                    st.code(traceback.format_exc())
        # --- Audit section (MUST NOT be inside another expander) ---
        prompt_path = MODEL3_DIR / (m2json.stem + ".model3.prompt.txt")
        if prompt_path.exists():
            with st.expander("Audit: exact prompt used"):
                st.code(prompt_path.read_text(encoding="utf-8"))

        # Artifacts download
        st.markdown("---")
        st.markdown("Artifacts (saved to outputs/)")
        c1, c2, c3 = st.columns(3)
        try:
            if model1_path.exists():
                with c1:
                    st.download_button("Download Model-1 CSV", data=model1_path.read_bytes(), file_name=model1_path.name, mime="text/csv")
            m2candidate = m2json if m2json.exists() else None
            if m2candidate:
                with c2:
                    st.download_button("Download Model-2 JSON", data=m2candidate.read_bytes(), file_name=m2candidate.name, mime="application/json")
            m3json = MODEL3_DIR / (m2json.stem + ".model3.json")
            m3txt = MODEL3_DIR / (m2json.stem + ".model3.txt")
            if m3json.exists():
                with c3:
                    st.download_button("Download Model-3 JSON", data=m3json.read_bytes(), file_name=m3json.name, mime="application/json")
                    if m3txt.exists():
                        st.download_button("Download Model-3 TXT", data=m3txt.read_bytes(), file_name=m3txt.name, mime="text/plain")
        except Exception as e:
            st.warning("Could not prepare downloads: " + str(e))

        # Debug logs
        if show_debug:
            with logs_expander:
                st.header("Debug info")
                st.text("Model-2 message: " + (m2msg or ""))
                st.text("Model-3 message: " + (m3msg or ""))

    except Exception as e:
        st.error("Pipeline failed: " + str(e))
        if show_debug:
            st.exception(e)
            st.code(traceback.format_exc())
