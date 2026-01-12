# LLMEvals - Lightweight evaluation runner (Groq/OpenAI-compatible)

Drop `model2.json` and corresponding `model3.json` files into:
 - llmevals/inputs/model2/
 - llmevals/inputs/model3/

Filename convention:
 - Use the same stem for pairs, e.g. `clean_00001.model2.json` and `clean_00001.model3.json`.
 - The script pairs by filename stem.

Configuration:
 - Copy `.env.example` -> `.env` and update GROQ_API_KEY / GROQ_API_URL, or OPENAI_API_KEY/OPENAI_API_BASE.
 - Set CLIENT_TYPE to `groq_http` (default) or `openai_compat`.

Run:
 - Locally: `python llmevals_pkg/run_eval.py --inputs llmevals/inputs --out ./llmevals/outputs`
 - Docker: `docker-compose up --build llmevals`

Outputs:
 - `outputs/individual_evals/<stem>.eval.json` (per-sample)
 - `outputs/final_report.json` and `outputs/final_report.md`
