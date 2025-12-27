"""
Prompt builder for Model-3 (narrative + context analysis).
Provides build_prompt_from_row(merged_context, user_context=None) -> str.

Design goals:
- Produce a compact, guarded one-shot prompt.
- Include a very small example (one-shot) to encourage JSON output.
- Only include essential Model-2 facts and optional user_context.
- Keep tokens low: remove long raw text and large keys.
- Use explicit instruction: prefer downgraded language when corroboration is absent.
"""

import json
import textwrap

NUMERIC_KEYS = [
    "Platelets", "Hemoglobin", "MCHC", "MCH", "MCV",
    "Total_Cholesterol", "HDL", "LDL_estimated", "Triglycerides",
    "Urea_BUN", "CRP", "Neutrophils", "Lymphocytes"
]

MINIMAL_EXAMPLE = {
    "FACTS": {
        "patterns": {"thrombocytopenia": {"present": True}},
        "Platelets": 105
    },
    "USER_CONTEXT": {"age": 30, "gender": "Female", "lifestyle": {"smoking": "no"}}
}

EXAMPLE_OUTPUT = {
    "summary": "Platelet count is low (105) suggesting thrombocytopenia; other provided values appear within supplied norms.",
    "possible_explanations": [
        "Isolated low platelets — could be transient, related to lab variability, or other benign causes (weak signal)."
    ],
    "lifestyle_guidance": [
        "Avoid NSAIDs and heavy alcohol until reviewed by a clinician.",
        "Maintain hydration and balanced nutrition."
    ],
    "when_to_consult_doctor": "Seek clinician review within 48–72 hours if new symptoms (fever, unusual bleeding, severe bruising) appear, or sooner if platelet count falls further on repeat testing.",
    "notes": "Moderate confidence based on available facts; not diagnostic."
}


def compact_facts_from_row(row: dict) -> dict:
    """
    Convert merged_context / model-2 row to a compact facts dict used for prompt.
    Only keep boolean patterns, list of causes, and a small set of numeric keys.
    """
    facts = {}

    # patterns (keep as-is if present)
    patterns = row.get("patterns")
    if isinstance(patterns, dict):
        # keep only presence flag and minimal supportive data
        pruned = {}
        for k, v in patterns.items():
            if isinstance(v, dict):
                pruned[k] = {"present": bool(v.get("present", False))}
                # include 'isolated' or 'severity' if present
                if "isolated" in v:
                    pruned[k]["isolated"] = v.get("isolated")
                if "severity" in v:
                    pruned[k]["severity"] = v.get("severity")
        facts["patterns"] = pruned

    # suspected causes, keep small list
    causes = row.get("causes") or row.get("causes_suspected") or row.get("causes_suspected_list")
    if causes:
        # causes might be list of dicts or strings
        if isinstance(causes, list):
            simple = []
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
            # keep the raw numeric value only
            facts[k] = row[k]

    # keep short explanation / score / band
    if "explanation" in row:
        facts["explanation"] = str(row.get("explanation"))[:300]
    if "score" in row:
        facts["score"] = row.get("score")
    if "band" in row:
        facts["band"] = row.get("band")

    return facts


def build_prompt_from_row(merged_context: dict, user_context: dict = None) -> str:
    """
    Build a compact guarded prompt string for the LLM.

    merged_context: merged Model-2 context (dict) -- may include many keys
    user_context: optional dict with age/gender/lifestyle keys (sanitized)
    """
    facts = compact_facts_from_row(merged_context)

    # If user_context contains an interpretive summary, include it compactly
    ctx_summary = ""
    if user_context and isinstance(user_context, dict):
        # If the caller passed an interpret_context result (with context_summary), prefer that.
        ctx_summary = user_context.get("context_summary") or ""
        # If caller supplied raw fields, generate a short line
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
            ctx_summary = "; ".join(parts)

    # Compose system preamble (short and strict) — now includes explicit corroboration instruction
    system_preamble = textwrap.dedent("""\
    SYSTEM: You are a cautious medical-report explainer. FOLLOW THESE RULES EXACTLY:
    1) USE ONLY THE FACTS PROVIDED in the FACTS block and the short USER_CONTEXT summary if supplied. DO NOT invent or infer new clinical facts.
    2) If a suspected cause is supported by a single marginal laboratory signal but lacks corroborating markers (e.g., normal CRP/ESR for infection), LABEL THAT CAUSE AS A 'weak signal' and reflect uncertainty in language.
    3) DO NOT reclassify or contradict upstream Model-2 statuses.
    4) DO NOT prescribe medications, dosages, or recommend surgery.
    5) RETURN ONLY VALID JSON EXACTLY MATCHING THE SCHEMA: {summary, possible_explanations, lifestyle_guidance, when_to_consult_doctor, notes}.
    6) Keep arrays concise (1-4 items). Each item must be 1-2 short sentences.
    7) Use cautious phrasing: 'may', 'could', 'consider', 'discuss with a clinician'.
    8) If uncertain or insufficient facts, state that concisely in possible_explanations.
    9) DO NOT include extra fields or explanatory text outside the JSON.
    """)

    # Compose example block (small one-shot)
    example_block = {
        "EXAMPLE_INPUT": {
            "FACTS": MINIMAL_EXAMPLE["FACTS"],
            "USER_CONTEXT": MINIMAL_EXAMPLE["USER_CONTEXT"]
        },
        "EXAMPLE_OUTPUT": EXAMPLE_OUTPUT
    }

    # Build the final facts/user_context JSONs to insert
    facts_json = json.dumps(facts, separators=(",", ":"), ensure_ascii=False)
    # For user_json: if user_context is the interpreted dict (having context_summary), include it as-is.
    user_json = json.dumps(user_context or {}, separators=(",", ":"), ensure_ascii=False)

    # Build instruction
    instruction = textwrap.dedent("""\
    INSTRUCTION: Using ONLY the FACTS and optional USER_CONTEXT below, produce JSON only that validates to:
    {
      "summary": "<one paragraph>",
      "possible_explanations": ["..."],
      "lifestyle_guidance": ["..."],
      "when_to_consult_doctor": "<one line>",
      "notes": "<optional short note>"
    }
    Keep outputs concise and conservative. If a probable cause lacks corroboration, call it a 'weak signal' in possible_explanations.
    """)

    # Compose prompt string (compact) and explicitly include the compact user context summary (if present)
    prompt_parts = [
        system_preamble,
        "\nONE-SHOT EXAMPLE (INPUT -> OUTPUT):",
        json.dumps(example_block, indent=2, ensure_ascii=False),
        "",
        "FACTS:",
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
