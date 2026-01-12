#!/usr/bin/env python3
# llmevals/llmevals_pkg/run_eval.py
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from llmevals_pkg.client import LLMClient
from llmevals_pkg.evaluator import LLMEvaluator
from llmevals_pkg.report import save_final_report


def find_pairs(input_dir: Path):
    # expects input_dir/model2 and input_dir/model3
    m2_dir = input_dir / "model2"
    m3_dir = input_dir / "model3"
    if not m2_dir.exists() or not m3_dir.exists():
        raise FileNotFoundError("Inputs must contain model2/ and model3/ directories")

    def base_stem(p: Path) -> str:
        name = p.stem  # removes extension
        # repeatedly strip trailing .model2 or .model3 tokens
        while True:
            if name.endswith('.model2'):
                name = name[: -len('.model2')]
            elif name.endswith('.model3'):
                name = name[: -len('.model3')]
            else:
                break
        return name

    m2_files = {}
    for p in sorted(m2_dir.glob('*.json')):
        m2_files.setdefault(base_stem(p), []).append(p)
    m3_files = {}
    for p in sorted(m3_dir.glob('*.json')):
        m3_files.setdefault(base_stem(p), []).append(p)

    pairs = []
    for stem, m2list in m2_files.items():
        m3list = m3_files.get(stem)
        if not m3list:
            continue
        for i in range(min(len(m2list), len(m3list))):
            pairs.append({"stem": stem, "model2": m2list[i], "model3": m3list[i]})
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="./inputs", help="inputs folder")
    p.add_argument("--out", default="./outputs", help="output folder")
    p.add_argument("--mode", default=None, choices=["pairwise", "merge"], help="evaluation mode (overrides .env)")
    p.add_argument("--tone", default=None, help="prompt tone override")
    args = p.parse_args()

    load_dotenv(dotenv_path=".env")

    input_dir = Path(args.inputs).resolve()
    out_dir = Path(args.out).resolve()

    client_cfg = {
        # allow "groq" or "groq_http" in CLIENT_TYPE
        "CLIENT_TYPE": os.getenv("CLIENT_TYPE", "groq"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GROQ_API_URL": os.getenv("GROQ_API_URL"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "llama-3-8b")
    }
    client = LLMClient(client_cfg)

    pairs = find_pairs(input_dir)

    if not pairs:
        print("No matching pairs found in inputs/model2 and inputs/model3.")
        return

    mode = args.mode or os.getenv("EVAL_MODE", "pairwise")
    tone = args.tone or os.getenv("PROMPT_TONE", "concise")

    max_tokens = int(os.getenv('MAX_TOKENS', '1024'))
    evaluator = LLMEvaluator(client, out_dir, tone=tone, mode=mode, max_tokens=max_tokens)
    run_result = evaluator.run(pairs)

    # save final report
    saved = save_final_report(out_dir, run_result)
    print("Saved final reports:", saved)


if __name__ == "__main__":
    main()
