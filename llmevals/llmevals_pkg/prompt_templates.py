# llmevals/llmevals_pkg/prompt_templates.py
from typing import Dict

PAIRWISE_SCHEMA_REQ = {
    "description": "Return a single JSON object for this single file evaluation. DO NOT output any non-JSON text.",
}

def build_pairwise_prompt(model2_text: str, model3_text: str, filename: str, tone: str = 'concise') -> str:
    prompt = f"""
You are a strict evaluation assistant. You will be given two JSON blobs (Model2 and Model3 outputs) for the SAME clinical report.

Your task is to evaluate how faithfully Model3 reflects Model2's signals, patterns, themes, and confidence.

Return EXACTLY one JSON object with NO surrounding text.

Required schema:
- filename: string (use the provided filename exactly)
- overall_score: integer 0-100
- summary: 1-2 sentences
- issues: array of objects {{
    "type": "hallucination|missing|overclaim|tone|format",
    "severity": "low|medium|high",
    "message": string
  }}
- strengths: array of short strings
- suggested_fix: array of short strings
- metrics: object (include only if applicable)
- raw_eval: 1-3 sentence internal reasoning

Scoring rubric:
- 90–100: Faithful, cautious, no hallucinations, excellent alignment
- 75–89: Minor omissions or wording issues
- 50–74: Noticeable missing signals or mild over-interpretation
- <50: Hallucinations, unsupported causes, or unsafe claims

Rules:
- Treat Model2 as ground truth And more importantly the blood report parameters in model2 as absolute.
- If Model3 introduces a cause, severity, or risk NOT supported by Model2 → issue type = "hallucination".
- If Model3 omits an important Model2 signal → issue type = "missing".
- Cautious language acknowledging uncertainty is NOT hallucination.
- Do NOT quote long text.
- If evaluation is impossible, return {{ "error": "reason" }} only.

Model2 JSON (compact):
{model2_text}

Model3 JSON (compact):
{model3_text}

Filename: {filename}
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
- overall_score: arithmetic mean of per-file overall_score values (rounded)
-For per-file objects, preserve the filename already associated with that block.
Do NOT invent new filenames.
The aggregate object should NOT include a filename field.


Input:
{combined_snippets}

Tone: {tone}
""".strip()
    return prompt
