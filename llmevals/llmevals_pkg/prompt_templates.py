# llmevals/llmevals_pkg/prompt_templates.py
from typing import Dict

PAIRWISE_SCHEMA_REQ = {
    "description": "Return a single JSON object for this single file evaluation. DO NOT output any non-JSON text.",
}

def build_pairwise_prompt(model2_text: str, model3_text: str, tone: str = 'concise', max_lines: int = 200) -> str:
    # NOTE: The caller should ensure model*_text are compacted/truncated already.
    prompt = f"""
You are a strict evaluation assistant. You will be given two JSON blobs (Model2 and Model3 outputs) for the SAME clinical report.
Carefully compare Model3's interpretation to Model2's signals and produce a single JSON object (no surrounding text, no commentary). Be concise.

Return exactly one JSON object with these fields:
- filename: string
- overall_score: integer 0-100 (numeric only)
- summary: short textual summary (1-2 sentences)
- issues: array of objects with fields {{ "type": string, "severity": "low|medium|high", "message": string }}
- strengths: array of short strings
- suggested_fix: array of short strings
- metrics: object with optional numeric keys precision_like, recall_like, hallucination_rate
- raw_eval: concise human reasoning (max 1-3 sentences)

If you cannot evaluate (e.g., input corrupted or token limits), return a single JSON object: {{ "error": "explain reason" }}.

IMPORTANT:
- OUTPUT ONLY VALID JSON as the whole response.
- Keep 'raw_eval' very short (1-3 sentences).
- Do not include long quoted text or the original documents.
- If Model3 introduces a claim, cause, or severity level that is not supported by Model2, mark it as 'hallucination'. 
  Cautious language that explicitly acknowledges uncertainty or limited evidence should NOT be treated as hallucination.
- If Model3 omits a key signal present in Model2, mark it as type 'missing'.

Model2 JSON (compact):
{model2_text}

Model3 JSON (compact):
{model3_text}

Tone: {tone}
""".strip()
    return prompt


def build_merge_prompt(combined_snippets: str, tone: str = 'concise') -> str:
    prompt = f"""
You are a strict evaluation assistant. You will be given many per-file blocks in this exact format:

===<stem>===
Model2:
<json>
Model3:
<json>

For each block, produce one compact JSON object identical to the pairwise schema described earlier. After the per-file objects, produce ONE top-level aggregated JSON object with keys:
- overall_score: averaged number
- summary: 1-3 sentence summary of the batch
- issues: aggregated list (deduplicated by message)
- strengths: deduplicated list
- suggested_fix: deduplicated list
- metrics: averages for numeric metrics

Important:
- OUTPUT exactly ONE valid JSON value (an object with "files": [ ... ] and "aggregate": { ... }) or a single error object if you cannot process.
- Keep each per-file object compact. No extra commentary text.
- If you hit token limits, return { "error": "token_limit" } or an explanatory error message.

Input:
{combined_snippets}

Tone: {tone}
""".strip()
    return prompt
