# model3/prompts.py
"""
Prompt builder for Model-3 (narrative + context analysis).
Provides build_prompt_from_row(merged_context, user_context=None) -> str.

Design goals:
- Produce a compact guarded one-shot prompt.
- Include a small example to encourage JSON output.
- Include an explicit REPORT_VALUES block (numerical grounding).
- Keep tokens focused.
- Use explicit instruction: prefer downgraded language when corroboration is absent.
"""

import json
import textwrap

NUMERIC_KEYS = [
    "Platelets", "Hemoglobin", "MCHC", "MCH", "MCV",
    "Total_Cholesterol", "HDL", "LDL_estimated", "Triglycerides",
    "Urea_BUN", "CRP", "Neutrophils", "Lymphocytes"
]

# Improved minimal example with values + facts
MINIMAL_EXAMPLE = {
    "REPORT_VALUES": {
        "Platelets": 105,
        "CRP": 0.3,
        "LDL_estimated": 132,
        "HDL": 38
    },
    "FACTS": {
        "patterns": {
            "thrombocytopenia": {"present": True, "severity": "moderate"},
            "dyslipidemia": {"present": True}
        }
    },
    "USER_CONTEXT": {
        "age": 42,
        "gender": "Male",
        "current_symptoms": "easy bruising"
    }
}

EXAMPLE_OUTPUT = {
    "summary": "Platelet count is low (105) suggesting thrombocytopenia; LDL 132 with HDL 38 indicates dyslipidemia and raised cardiovascular risk.",
    "possible_explanations": [
        "Isolated low platelets — could be transient or related to lab variability (weak signal).",
        "Lipid abnormalities may reflect dietary/metabolic contributors and increase CV risk."
    ],
    "lifestyle_guidance": [
        "Avoid NSAIDs until clinician review and maintain hydration.",
        "Work with clinician on diet and activity to improve lipid profile."
    ],
    "when_to_consult_doctor": "Seek clinician review within 48–72 hours for platelet count < 150 if new bleeding/bruising occurs, or sooner if symptoms worsen.",
    "notes": "Moderate confidence based on numeric findings and patterns."
}


def compact_facts_from_row(row: dict) -> dict:
    """
    Convert merged_context / model-2 row to a compact facts dict used for prompt.
    Only keep boolean patterns, list of causes, and numeric keys.
    """
    facts = {}

    # patterns (keep flags and severity)
    patterns = row.get("patterns")
    if isinstance(patterns, dict):
        pruned = {}
        for k, v in patterns.items():
            if isinstance(v, dict):
                pruned[k] = {"present": bool(v.get("present", False))}
                if "isolated" in v:
                    pruned[k]["isolated"] = v.get("isolated")
                if "severity" in v:
                    pruned[k]["severity"] = v.get("severity")
        facts["patterns"] = pruned

    # suspected causes, keep small list
    causes = row.get("causes") or row.get("probable_causes") or row.get("causes_suspected") or row.get("causes_suspected_list")
    if causes:
        simple = []
        if isinstance(causes, list):
            for c in causes:
                if isinstance(c, dict) and "cause" in c:
                    simple.append(c["cause"])
                elif isinstance(c, str):
                    simple.append(c)
            if simple:
                facts["causes"] = simple[:5]
        elif isinstance(causes, str):
            facts["causes"] = [causes]

    # important numeric keys (if present)
    for k in NUMERIC_KEYS:
        if k in row:
            facts[k] = row[k]

    # keep short explanation / score / band
    if "explanation" in row:
        facts["explanation"] = str(row.get("explanation"))[:300]
    if "score" in row:
        facts["score"] = row.get("score")
    if "band" in row:
        facts["band"] = row.get("band")

    return facts


def build_report_values_block(row: dict) -> str:
    """
    Build a short human-readable bullet list of numeric values from merged_context.
    This block is separated from 'FACTS' to ensure the LLM sees exact numbers.
    """
    lines = []
    for k in NUMERIC_KEYS:
        if k in row and row[k] is not None:
            try:
                lines.append(f"- {k}: {row[k]}")
            except Exception:
                pass
    # Also include other numeric-ish keys found in row
    # (fallback: include any numeric value)
    for k, v in row.items():
        if k in NUMERIC_KEYS:
            continue
        if isinstance(v, (int, float)):
            lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else ""

def build_prompt_from_row(merged_context: dict, user_context: dict = None) -> str:
    """
    Build a compact guarded prompt string for the LLM.

    merged_context: merged Model-2 context (dict)
    user_context: sanitized user_context (dict). If it also contains an interpretive 'context_summary' key,
                  it will be used by the prompt. We intentionally DO NOT mutate user_context here.
    """
    facts = compact_facts_from_row(merged_context)
    report_values_block = build_report_values_block(merged_context)

    # If user_context contains an interpretive summary, include it compactly
    ctx_summary = ""
    if user_context and isinstance(user_context, dict):
        # prefer explicit context_summary key if present
        ctx_summary = user_context.get("context_summary") or ""
        if not ctx_summary:
            parts = []
            if user_context.get("age") is not None:
                parts.append(f"Age:{user_context.get('age')}")
            if user_context.get("gender"):
                parts.append(f"Gender:{user_context.get('gender')}")
            life = user_context.get("lifestyle") or {}
            sm = life.get("smoking")
            if sm:
                parts.append(f"Smoking:{sm}")
            ac = life.get("activity")
            if ac:
                parts.append(f"Activity:{ac}")
            symptoms = user_context.get("current_symptoms")
            if symptoms:
                parts.append(f"Symptoms:{symptoms}")
            ctx_summary = "; ".join(parts)

    # Compose system preamble (short and strict)
    system_preamble = textwrap.dedent("""\
    SYSTEM: You are a cautious medical-report explainer. FOLLOW THESE RULES EXACTLY:
    1) USE ONLY THE FACTS PROVIDED in REPORT_VALUES and FACTS, and the short USER_CONTEXT summary if supplied. DO NOT invent or infer new clinical facts.
    2) If a suspected cause is supported by a single marginal laboratory signal but lacks corroborating markers (e.g., normal CRP/ESR for infection), LABEL THAT CAUSE AS A 'weak signal' and reflect uncertainty in language.
    3) DO NOT reclassify or contradict upstream Model-2 statuses.
    4) DO NOT prescribe medications, dosages, or recommend surgery.
    5) RETURN ONLY VALID JSON EXACTLY MATCHING THE SCHEMA: {summary, possible_explanations, lifestyle_guidance, when_to_consult_doctor, notes}.
    6) Keep arrays concise (1-4 items). Each item must be 1-2 short sentences.
    7) Use cautious phrasing: 'may', 'could', 'consider', 'discuss with a clinician'.
    8) If uncertain or insufficient facts, state that concisely in possible_explanations.
    9) Do not repeat text across fields. Each field has a distinct role:
       - summary: one concise paragraph (mention at most one numeric value if clinically important)
       - possible_explanations: reasons/evidence (mention numeric values that support the explanation)
       - lifestyle_guidance: practical, non-prescriptive actions (do NOT restate explanations)
       - notes: confidence/limitation statements only.
    IMPORTANT: If USER_CONTEXT_SUMMARY mentions symptoms, explicitly ACKNOWLEDGE those symptoms in the 'summary' sentence even if labs are normal.
    10) If a pattern name implies direction (e.g., neutrophilia, lymphocytosis),
    do NOT describe it using the opposite direction (e.g., "low").
    If values are low, use the correct opposing term or state uncertainty.
    11) Do NOT interpret numeric values as abnormal unless the corresponding
    Model-2 pattern is present or explicitly flagged.
    Unflagged numeric values may be mentioned only as 'within reported range'
    or omitted.
    12) If USER_CONTEXT indicates young age (<25) and no severe Model-2 patterns,
    prefer reassurance-oriented language and avoid alarmist phrasing.
    13) If REPORT_VALUES contain a complete group of related measurements
    (e.g., lipid profile, CBC indices, inflammatory markers) and no abnormal
    Model-2 pattern is flagged, explicitly state in the summary or notes that
    no significant abnormal pattern is identified for that group.
    14) If a Model-2 pattern or suspected cause appears to rely on a single
    laboratory signal or lacks corroborating markers, you may briefly note
    this uncertainty using cautious language (e.g., "based on limited evidence",
    "may benefit from clinical correlation"), but you MUST NOT negate, override,
    or reclassify the Model-2 pattern.
                 
    """)

    # Compose example block (richer one-shot)
    example_block = {
        "EXAMPLE_INPUT": {
            "REPORT_VALUES": MINIMAL_EXAMPLE["REPORT_VALUES"],
            "FACTS": MINIMAL_EXAMPLE["FACTS"],
            "USER_CONTEXT": MINIMAL_EXAMPLE["USER_CONTEXT"]
        },
        "EXAMPLE_OUTPUT": EXAMPLE_OUTPUT
    }

    facts_json = json.dumps(facts, separators=(",", ":"), ensure_ascii=False)
    user_json = json.dumps(user_context or {}, separators=(",", ":"), ensure_ascii=False)
    report_values_text = report_values_block or "No raw numeric values provided."

    instruction = textwrap.dedent("""\
    INSTRUCTION: Using ONLY the REPORT_VALUES (exact numbers), the FACTS (patterns & flags), and the optional USER_CONTEXT summary below, produce JSON only that validates to:
    {
      "summary": "<one paragraph, mention at most one numeric value if critical>",
      "possible_explanations": ["..."],
      "lifestyle_guidance": ["..."],
      "when_to_consult_doctor": "<one line — specific triggers or time-window>",
      "notes": "<optional short note>"
    }
    - Place numerical evidence in 'possible_explanations' when the number supports the explanation.
    - Do NOT repeat the same sentence across fields.
    - Keep each array short (1-4 items).
    - If uncertain, call out 'weak signal' language in possible_explanations.
    """)

    prompt_parts = [
        system_preamble,
        "\nONE-SHOT EXAMPLE (INPUT -> OUTPUT):",
        json.dumps(example_block, indent=2, ensure_ascii=False),
        "",
        "REPORT_VALUES (exact laboratory numbers from the report):",
        report_values_text,
        "",
        "FACTS (Model-2 patterns & compact facts):",
        facts_json,
        ""
    ]

    if ctx_summary:
        prompt_parts += ["USER_CONTEXT_SUMMARY:", json.dumps({"context_summary": ctx_summary}, ensure_ascii=False), ""]

    prompt_parts += [
        "USER_CONTEXT_FULL:",
        user_json,
        "",
        instruction,
        "",
        "OUTPUT: Return JSON only with no surrounding text."
    ]

    prompt = "\n".join(prompt_parts)
    return prompt
