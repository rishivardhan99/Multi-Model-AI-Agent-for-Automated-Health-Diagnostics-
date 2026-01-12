
# 🧠 Multi-Model AI Agent for Automated Health Diagnostics

A **production-grade, research-oriented multi-model AI system** for automated blood report analysis.  
The system integrates **OCR, deterministic clinical reasoning, knowledge-graph–based inference, and LLM-powered narrative synthesis** into a single, auditable pipeline. It includes a new evaluation module (`llmevals/`) for honest model evaluation and a lightweight chatbot assistant (`chatbot/`) for interactive exploration.



> ⚠️ **Medical Disclaimer**  
> This platform is strictly an **assistive decision-support system** intended for educational and research use.  
> It does **not** provide diagnoses, prescriptions, or treatment recommendations. All outputs must be reviewed by a qualified medical professional.

---


## ✨ Key Capabilities

- 📄 Accepts **PDF / Image / JSON** blood reports  
- 🔍 OCR with robust parameter extraction  
- 🧪 Parameter-level clinical interpretation (Normal / Low / High)  
- 🧠 Deterministic pattern detection & probabilistic risk inference (Model-2)  
- 🧩 Knowledge-graph–based causal reasoning  
- ✍️ LLM-based **explainable medical narratives** (Model-3) with safety guardrails  
- ✅ Evaluation framework (`llmevals/`) with explicit abstention and validation logic  
- 🖥️ Interactive **Streamlit UI** and small **chatbot** for exploration  
- 🐳 **Dockerized** for reproducibility

---

## 🧭 System Architecture

```
Input (PDF / Image / JSON)
        ↓
Extractor (OCR + Parsing)
        ↓
Model-1 (Clinical Parameter Normalization)
        ↓
Model-2 (Pattern, Risk & Causal Reasoning)
        ↓
Model-3 (LLM Narrative Synthesis)
        ↓
LLMEVALS (Evaluation, Validation, Aggregation)  ← optional offline step
        ↓
Final Report + Auditable Artifacts + Streamlit UI / Chatbot
```

> The system is intentionally **layered**:
> - Clinical facts are determined deterministically (Model-1 & Model-2)
> - LLMs are used only for **synthesis and communication** (Model-3)
> - Evaluation & safety (llmevals) enforces auditability and honest metrics

---

## 🧩 Component Breakdown

### 1️⃣ Extractor — OCR & Structuring
**Location:** `extractor/`

**Responsibilities**
- OCR for scanned PDFs and images  
- Text normalization and cleanup  
- Parameter detection with plausibility checks  
- Conversion into structured CSV / JSON

**Outputs**
```
outputs/structured_per_report/<file>.structured.csv
outputs/model1_per_report/<file>.model1_final.csv
```

---

### 2️⃣ Model-1 — Clinical Parameter Normalization
**Purpose:** Deterministic interpretation of extracted lab values.

**Key behavior**
- Compares values against reference ranges  
- Assigns status labels: Normal / Low / High  
- Produces parameter-level notes for downstream reasoning

**Output**
```
outputs/model1_per_report/<file>.model1_final.csv
```

> Model-1 contains **no probabilistic logic** — it establishes factual ground truth.

---

### 3️⃣ Model-2 — Pattern, Risk & Causal Reasoning
**Location:** `model2/`

**Design**
- Fully **deterministic and auditable** (no LLM dependency)
- Pattern detection (e.g., anemia), derived metrics, knowledge-graph causal links
- Confidence scoring based on evidence completeness

**Key files**
```
model2/
├── model2_runner.py
├── serializer.py
├── verifier.py
└── pipeline/
    ├── loader.py
    ├── pattern_engine.py
    ├── probable_causes.py
    ├── knowledge_graph.py
    ├── risk_engine.py
    └── confidence.py
```

**Outputs**
```
outputs/model2_outputs/<file>.model2.json
outputs/model2_outputs/<file>.model2.txt
```

> Model-2 performs **reasoning**, not narration.

---

### 4️⃣ Model-3 — LLM Narrative Synthesis
**Location:** `model3/`

**Design philosophy**
- **LLM used only for synthesis/explanation** — not for establishing clinical facts
- Strict prompt contract: do not invent facts, prefer cautious language, return only JSON matching the schema
- Deterministic fallback when LLM fails

**Key files**
```
model3/
├── model3_runner.py
├── prompts.py
├── schema_model3.json
├── guardrails.py
└── gemini_client.py
```

**Important prompt rules added (Model-3):**
- Rule 11: *Do NOT interpret numeric values as abnormal unless Model-2 flags a pattern.*
- Rule 13 (new): *If a complete group of related measurements is provided (e.g., lipid profile, CBC) and no Model-2 abnormal pattern is flagged, explicitly state that no significant abnormal pattern is identified for that group.*  
  — prevents omission ambiguity.
- Rule 14 (new): *If Model-2 relies on a single signal, Model-3 may note limited evidence using cautious language (e.g., "may benefit from clinical correlation") but must NOT negate or reclassify Model-2.*

**Outputs**
```
outputs/model3/<file>.model3.json
outputs/model3/<file>.model3.txt
outputs/model3/<file>.model3.prompt.txt
```

---

### 5️⃣ LLMEVALS — Evaluation & Validation
**Location:** `llmevals/llmevals_pkg`

**Purpose:** Honest, auditable automated evaluation comparing Model-3 output to Model-2 (ground truth) with explicit abstention handling and validation.

**Core design decisions**
- **Do not count technical outages** (LLM or API failures) against model quality — those are *technical abstentions* and excluded from the denominator.
- **Safety-first abstentions** (clinical abstentions) are treated as **correct system behavior** (system passes) — e.g., when Model-3 hallucinates or omits critical high-confidence Model-2 signals.
- Post-hoc validation applies penalties for `hallucination` and `missing` issues and computes a validated score.
- Aggregation excludes technical abstentions from denominator; clinical abstentions are included and count as passes.

**Validation rules (implemented)**
- Penalty weights for issues (configurable):
  - hallucination high/medium/low → 20 / 10 / 5 points
  - missing high/medium/low → 10 / 4 / 2 points
- `critical_missing_count`: counts only **high-severity** missing issues (these represent high-confidence Model-2 signals omitted by Model-3).
- **Abstention types**:
  - `technical` — infra/parse errors (excluded)
  - `clinical` — high-severity hallucination or ≥2 critical missing signals (system abstains for safety; counts as pass)
- Soft pass threshold (env): `SOFT_PASS_THRESHOLD` (default 60.0)
- System pass threshold (env): `SYSTEM_PASS_THRESHOLD` (default 90.0 is a different configured threshold and may be used where desired)

**Key files**
```
llmevals/
├── llmevals_pkg/
│   ├── client.py             # LLM client (groq/openai compat)
│   ├── evaluator.py          # main evaluation driver
│   ├── prompt_templates.py   # pairwise & merge prompt templates
│   ├── report.py             # final report / md export
│   └── validation.py        # validation & abstention rules
└── run_eval.py               # CLI entrypoint
```

**How the aggregate is computed (plain)**
- Denominator = `system_total_evaluated` = number of non-technical evaluations (technical abstentions excluded)
- Numerator = `validated_system_pass_count` = count of items where `system_pass == True`
- `validated_system_accuracy` = Numerator / Denominator × 100
- `abstention_count` counts **clinical abstentions** (safety behavior), `abstention_rate` = abstentions / Denominator

**Outputs**
```
outputs/llmevals/final_report.json
outputs/llmevals/final_report.md
outputs/llmevals/individual_evals/*.eval.json
```

---

## 🗂 Chatbot — lightweight interactive assistant
**Location:** `chatbot/`

**Purpose:** Small conversational tool to:
- query a processed report
- view Model-2 patterns and Model-3 narrative
- test prompt variations quickly
- simulate human-in-the-loop escalation workflows

**Note:** Chatbot is intentionally *exploratory* and not part of the production evaluation pipeline. Use for developer testing and demos.

---

## 🖥️ Streamlit Application
**Entry point:** `app.py`

**Features**
- File upload (PDF / Image / JSON)
- Patient & context input
- Step-by-step pipeline execution
- Model-2 reasoning visualization
- Model-3 narrative output
- Artifact downloads and debug/audit views

Run locally:
```bash
streamlit run app.py
```

---

## 🐳 Docker Setup

**UI Docker (Streamlit)**

Build:
```bash
docker build -f Dockerfile.ui -t medicube-ai .
```

Run:
```bash
docker run -p 8501:8501 --env-file .env medicube-ai
# open http://localhost:8501
```

**LLMEVALS Docker** (example for evaluation runs)
You may prefer a separate image for batch evaluation.

Build:
```bash
docker build -f Dockerfile.llmevals -t medicube-llmevals .
```

Run (PowerShell-friendly example):
```powershell
docker run --rm `
  -e CLIENT_TYPE=groq `
  -e MODEL_NAME=llama-3.1-8b-instant `
  -e MAX_TOKENS=1024 `
  -e MAX_CHARS_PER_ITEM=3000 `
  -e MAX_INPUT_CHARS_PER_REQUEST=12000 `
  -e EVAL_MODE=pairwise `
  --env-file .env `
  -v "${PWD}\inputs:/app/inputs" `
  -v "${PWD}\outputs:/app/outputs" `
  medicube-llmevals
```

> **Security reminder:** never commit `.env` or API keys to Git. Add `.env` to `.gitignore`.

---

## 🔧 Environment variables (key ones used by llmevals & runners)

- `CLIENT_TYPE` — `groq_http` or `openai_compat`
- `GROQ_API_KEY`, `GROQ_API_URL`
- `OPENAI_API_KEY`, `OPENAI_API_BASE`
- `MODEL_NAME` — model identifier
- `MAX_TOKENS` — tokens allocated to evaluator LLM calls
- `MAX_CHARS_PER_ITEM` — compact per-item text cap (default 3000)
- `MAX_INPUT_CHARS_PER_REQUEST` — input limit per merge request (default ~12000)
- `MERGE_BATCH_SIZE` — merge mode hint
- `SOFT_PASS_THRESHOLD` — soft threshold for `system_pass` (default 60.0)
- `SYSTEM_PASS_THRESHOLD` — optional system pass threshold (default 90.0)
- `EVAL_MODE` — `pairwise` or `merge`

---

## ✅ Best practices & guidelines

- **Do not** allow Model-3 to *contradict* Model-2. Model-3 may express **cautious uncertainty** but must not reclassify Model-2 patterns. (See Model-3 rules 14 and 11/13.)
- **Exclude technical failures** from accuracy calculations — these are infra issues.
- **Count clinical abstentions as correct system behavior** — forced abstention for safety is desirable.
- **Keep prompts concise** and include explicit instructions to return valid JSON only.
- **Audit everything**: store inputs, prompts, raw LLM outputs, and parsed JSON in `outputs/` for reproducibility.

---

## 📄 Example llmevals output (short)

`final_report.json` provides:
- `mode` (pairwise / merge)
- `individuals` (array of per-file eval + validation)
- `aggregate`:
  - `llm_raw_count`, `llm_raw_mean`, ...
  - `system_total_evaluated`, `validated_system_pass_count`, `validated_system_accuracy`
  - `abstention_count`, `abstention_rate`
  - `high_severity_issues` list

Per-file `*.eval.json` contains:
```json
{
  "filename": "clean_00012.model1_final",
  "model2_path": "...",
  "model3_path": "...",
  "eval": { ... },
  "validation": {
    "raw_score": 80.0,
    "penalty": 34.0,
    "validated_score": 46.0,
    "is_abstain": true,
    "abstention_type": "clinical",
    "system_pass": true,
    "reasons": ["clinical_abstention_due_to_high_risk_output"]
  }
}
```

---

## 📝 Versioning & release notes (recommended)
When you push changes:
- Tag release versions (e.g., `v0.4-llmevals`)
- Add a short `CHANGELOG.md` noting:
  - `llmevals` added with abstention/validation rules
  - `chatbot` added for interactive testing
  - Model-3 prompt updates (Rule 13 & Rule 14)
  - Any breaking changes to output schemas

---

## 🔐 Safety & compliance notes

- No medication dosing or prescriptive instructions are generated.
- Model-3 must always use cautious phrasing; hallucinations and unsupported claims are penalized.
- All sensitive keys **must** remain out of git.
- For any deployment beyond research, consult clinical advisors and comply with local health regulations.

---

## 📄 License

For academic, educational, and research use only.

---

## 🏁 Quick developer checklist before you push changes

1. `git status` → verify files to commit
2. Make sure `.env` is in `.gitignore`
3. `git add .` → `git commit -m "..."` → `git push origin main`
4. Tag release: `git tag -a v0.4 -m "Add llmevals & chatbot; Model-3 prompt updates"` → `git push --tags`
5. Run a quick llmevals dry run on a small sample to validate aggregation

---
