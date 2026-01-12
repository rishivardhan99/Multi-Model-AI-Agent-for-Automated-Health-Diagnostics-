# chatbot/context_builder.py
from chatbot.config_loader import cfg, templates
from chatbot.utils.file_utils import read_text_file, read_json_file

def build_context(retrieved_chunks, manifest: dict, chat_history: list, user_question: str) -> str:
    context_parts = []
    m_summary = f"Report ID: {manifest.get('report_id')}\nCanonical base: {manifest.get('canonical_base')}\n"
    context_parts.append("MANIFEST:\n" + m_summary)

    # Prefer model3_txt (primary)
    model3_path = manifest.get("artifacts", {}).get("model3_txt")
    model3_text = read_text_file(model3_path)
    if model3_text:
        context_parts.append("MODEL-3 REPORT (primary source):\n" + model3_text[:8000])
    else:
        model2_path = manifest.get("artifacts", {}).get("model2_json")
        model2 = read_json_file(model2_path)
        if model2:
            parts = []
            patterns = model2.get("patterns", {})
            positives = [k for k, v in patterns.items() if v]
            if positives:
                parts.append("Detected patterns: " + ", ".join(positives) + ".")
            derived = model2.get("derived_metrics", {})
            if derived:
                parts.append(
                    "Derived metrics: "
                    + ", ".join(f"{k} = {v}" for k, v in derived.items())
                    + "."
                )
            if parts:
                context_parts.append("MODEL-2 HIGHLIGHTS (fallback):\n" + "\n".join(parts))
            else:
                context_parts.append("No model2/model3 artifacts available.")

    # Optionally include retrieved evidence
    if retrieved_chunks:
        context_parts.append("RETRIEVED EVIDENCE (optional):")
        for c in retrieved_chunks:
            context_parts.append(f"[{c['chunk_id']}] {c['source']} (score={c.get('score'):.3f})\n{c['content'][:600]}")

    history_text = ""
    if chat_history:
        for turn in chat_history[-cfg.get("chat_history_turns", 3):]:
            history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    context = templates.get("system_instructions", "") + "\n\n"
    context += "\n\n".join(context_parts) + "\n\n"
    if history_text:
        context += "RECENT CHAT HISTORY:\n" + history_text + "\n\n"

    prompt = templates.get("answer_template", "{context}\n\n{question}").format(context=context, question=user_question)
    return prompt
