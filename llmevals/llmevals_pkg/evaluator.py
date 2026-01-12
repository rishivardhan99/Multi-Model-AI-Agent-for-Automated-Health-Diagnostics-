# llmevals_pkg/evaluator.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from llmevals_pkg.client import LLMClient
from llmevals_pkg.prompt_templates import build_pairwise_prompt, build_merge_prompt
from tqdm import tqdm
import os
from llmevals_pkg import validation


class LLMEvaluator:
    def __init__(self, client: LLMClient, outdir: Path, tone: str = "concise", mode: str = "pairwise", max_tokens: int = 1024):
        self.client = client
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.indiv_dir = self.outdir / "individual_evals"
        self.indiv_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.tone = tone
        self.max_tokens = int(max_tokens)
        # configuration knobs (env overrides)
        self.max_input_chars_per_request = int(os.getenv('MAX_INPUT_CHARS_PER_REQUEST', str(self.client._estimate_chars_from_tokens(self.max_tokens) // 2)))
        # explicit per-item char cap for safe pairwise usage
        self.max_chars_per_item = int(os.getenv('MAX_CHARS_PER_ITEM', str(3000)))
        self.merge_batch_size_hint = int(os.getenv('MERGE_BATCH_SIZE', '10'))

    def _read_json_file(self, p: Path) -> str:
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read()

    def _compact_json_text(self, text: str, max_chars: int) -> str:
        if not text:
            return text
        try:
            obj = json.loads(text)
        except Exception:
            if len(text) <= max_chars:
                return text
            return text[:max_chars-12] + '...<truncated>'
        def truncate_strings(o, max_str_len):
            if isinstance(o, dict):
                return {k: truncate_strings(v, max_str_len) for k, v in o.items()}
            if isinstance(o, list):
                if len(o) > 50:
                    o = o[:50]
                return [truncate_strings(v, max_str_len) for v in o]
            if isinstance(o, str):
                if len(o) > max_str_len:
                    return o[:max_str_len] + '...'
                return o
            return o
        max_str_len = 1200
        s = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        if len(s) <= max_chars:
            return s
        for _ in range(10):
            obj2 = truncate_strings(obj, max_str_len)
            s2 = json.dumps(obj2, ensure_ascii=False, separators=(',', ':'))
            if len(s2) <= max_chars:
                return s2
            max_str_len = max(30, int(max_str_len * 0.6))
        s3 = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        return (s3[:max_chars-12] + '...') if len(s3) > max_chars else s3

    def pairwise_eval(self, pairs: List[Dict[str, Path]]) -> List[Dict[str, Any]]:
        results = []
        for pair in tqdm(pairs, desc="Evaluating pairs"):
            name = pair["stem"]
            m2p = pair["model2"]
            m3p = pair["model3"]
            m2txt = self._read_json_file(m2p)
            m3txt = self._read_json_file(m3p)
            # compact both sides to per-item cap
            m2_compact = self._compact_json_text(m2txt, self.max_chars_per_item)
            m3_compact = self._compact_json_text(m3txt, self.max_chars_per_item)
            prompt = build_pairwise_prompt(m2_compact, m3_compact, tone=self.tone)
            try:
                resp = self.client.call(prompt, max_tokens=self.max_tokens, temperature=0.0)
                text = resp.get("text", "")
            except Exception as e:
                text = json.dumps({"error": str(e)})
            # try parse JSON
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = {"error": "failed_to_parse_llm_output", "raw_text": text}

            # run validation assessment (post-hoc) and attach to per-file output
            try:
                validation_summary = validation.assess(parsed if isinstance(parsed, dict) else {})
            except Exception as e:
                validation_summary = {"error": f"validation_failed: {e}"}

            outp = {
                "filename": name,
                "model2_path": str(m2p),
                "model3_path": str(m3p),
                "eval": parsed,
                "validation": validation_summary
            }
            # save per-file (pretty)
            with open(self.indiv_dir / f"{name}.eval.json", "w", encoding="utf-8") as fh:
                json.dump(outp, fh, indent=2, ensure_ascii=False)
            results.append(outp)
        return results

    def merge_eval(self, pairs: List[Dict[str, Path]]) -> List[Dict[str, Any]]:
        # unchanged merge logic (compaction + batching) - keep as-is and let validation operate on parsed outputs
        snippets = []
        for pair in pairs:
            name = pair["stem"]
            m2txt = self._read_json_file(pair["model2"])
            m3txt = self._read_json_file(pair["model3"])
            per_item_limit = min(self.max_chars_per_item, max(1000, self.max_input_chars_per_request // max(1, self.merge_batch_size_hint)))
            m2c = self._compact_json_text(m2txt, per_item_limit)
            m3c = self._compact_json_text(m3txt, per_item_limit)
            snippets.append((name, m2c, m3c))

        batches = []
        current_batch = []
        current_chars = 0
        overhead_estimate = 2000
        limit = int(os.getenv('MAX_INPUT_CHARS_PER_REQUEST', str(self.max_input_chars_per_request)))
        for name, m2c, m3c in snippets:
            item_text = f"==={name}===\nModel2:\n{m2c}\nModel3:\n{m3c}\n\n"
            item_len = len(item_text)
            if current_batch and (current_chars + item_len + overhead_estimate) > limit:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(item_text)
            current_chars += item_len
        if current_batch:
            batches.append(current_batch)

        aggregated = []
        for idx, batch in enumerate(batches):
            combined = "\n".join(batch)
            prompt = build_merge_prompt(combined, tone=self.tone)
            try:
                resp = self.client.call(prompt, max_tokens=self.max_tokens, temperature=0.0)
                text = resp.get('text', '')
                parsed = json.loads(text)
            except Exception as e:
                parsed = {"error": str(e), "raw_text": text if 'text' in locals() else None}
            with open(self.indiv_dir / f"merge_batch_{idx}.eval.json", 'w', encoding='utf-8') as fh:
                json.dump(parsed, fh, indent=2, ensure_ascii=False)
            aggregated.append(parsed)
        return aggregated

    def run(self, pairs: List[Dict[str, Path]]) -> Dict[str, Any]:
        if self.mode == "pairwise":
            indiv = self.pairwise_eval(pairs)
            agg = self.aggregate(indiv)
            return {"mode": "pairwise", "individuals": indiv, "aggregate": agg}
        else:
            merged = self.merge_eval(pairs)
            return {"mode": "merge", "merged_eval": merged}

    def aggregate(self, indiv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        import statistics
        # llm-level stats
        raw_scores = []
        llm_count = 0
        # system-level stats
        system_pass_count = 0
        system_total = 0
        abstained_count = 0
        hallucination_rejections = 0

        high_severity_issues = []

        for r in indiv_results:
            ev = r.get('eval', {})
            val = r.get('validation', {})

            # capture raw LLM score if present
            if isinstance(ev, dict) and 'overall_score' in ev:
                try:
                    raw_scores.append(float(ev['overall_score']))
                    llm_count += 1
                except Exception:
                    pass

            # validation info
            if isinstance(val, dict):
                sp = val.get('system_pass')
                abst_type = val.get('abstention_type')

                # technical abstentions are EXCLUDED from denominator
                if abst_type == 'technical':
                    continue

                # everything else counts toward system evaluation
                system_total += 1

                if sp is True:
                    system_pass_count += 1

                if abst_type == 'clinical':
                    abstained_count += 1

                # count high severity hallucinations seen
                issues = ev.get('issues') if isinstance(ev, dict) else []
                if issues:
                    for it in issues:
                        if it.get('severity') == 'high':
                            high_severity_issues.append({'file': r.get('filename'), 'issue': it})
                            if it.get('type') == 'hallucination':
                                hallucination_rejections += 1

        summary = {}
        if raw_scores:
            summary['llm_raw_count'] = llm_count
            summary['llm_raw_mean'] = statistics.mean(raw_scores)
            summary['llm_raw_median'] = statistics.median(raw_scores)
            summary['llm_raw_min'] = min(raw_scores)
            summary['llm_raw_max'] = max(raw_scores)
        # system-level
        summary['system_total_evaluated'] = system_total
        summary['validated_system_pass_count'] = system_pass_count
        summary['validated_system_accuracy'] = (system_pass_count / system_total * 100.0) if system_total else None
        summary['abstention_count'] = abstained_count
        summary['abstention_rate'] = (abstained_count / system_total * 100.0) if system_total else None
        summary['hallucination_rejections'] = hallucination_rejections
        summary['high_severity_issues'] = high_severity_issues

        return summary
