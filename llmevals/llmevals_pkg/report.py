# llmevals/llmevals_pkg/report.py
import json
from pathlib import Path
from typing import Any, Dict


def save_final_report(outdir: Path, run_result: Dict[str, Any]) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / 'final_report.json'
    md_path = outdir / 'final_report.md'
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(run_result, fh, indent=2, ensure_ascii=False)

    # make a short human-readable markdown summary
    lines = [f'# LLMEVALS Final Report', f'* mode: {run_result.get("mode")}', '']

    agg = run_result.get('aggregate', {})

    # LLM raw metrics
    if 'llm_raw_mean' in agg:
        lines.append('## LLM (raw) Metrics')
        lines.append('')
        lines.append(f'- **count**: {agg.get("llm_raw_count") or "-"}')
        lines.append(f'- **mean**: {agg.get("llm_raw_mean") or "-"}')
        lines.append(f'- **median**: {agg.get("llm_raw_median") or "-"}')
        lines.append(f'- **min**: {agg.get("llm_raw_min") or "-"}')
        lines.append(f'- **max**: {agg.get("llm_raw_max") or "-"}')
        lines.append('')

    # System validated metrics
    lines.append('## System-validated Metrics')
    lines.append('')
    lines.append(f'- **total evaluated**: {agg.get("system_total_evaluated") or 0}')
    lines.append(f'- **validated_system_pass_count**: {agg.get("validated_system_pass_count") or 0}')
    vs_acc = agg.get('validated_system_accuracy')
    lines.append(f'- **validated_system_accuracy**: {vs_acc:.2f}%' if vs_acc is not None else '-')
    lines.append(f'- **abstention_count**: {agg.get("abstention_count") or 0}')
    ar = agg.get('abstention_rate')
    lines.append(f'- **abstention_rate**: {ar:.2f}%' if ar is not None else '-')
    lines.append('')

    lines.append('## High severity issues (sample)')
    lines.append('')
    for it in agg.get('high_severity_issues', []):
        lines.append(f'- **{it.get("file")}**: {it.get("issue", {})}')
    lines.append('')

    lines.append('## Individual results (summary)')
    lines.append('')
    if run_result.get('mode') == 'pairwise':
        for item in run_result.get('individuals', []):
            lines.append(f'### {item.get("filename")}')
            ev = item.get('eval', {})
            val = item.get('validation', {})
            raw = ev.get('overall_score') if isinstance(ev, dict) else ev
            lines.append(f'- raw_overall_score: `{raw if raw is not None else "(no score)"}`')
            lines.append(f'- validation: `{val}`')
            lines.append('')
    else:
        lines.append('## Merged output saved in individual_evals/merge_batch_*.eval.json')

    with open(md_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    return {'json': str(json_path), 'md': str(md_path)}

