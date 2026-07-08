"""
RAG-grounded LLM interpretation layer.

Given model predictions (label -> probability), this:
  1. Retrieves relevant knowledge-base chunks (semantic if available, else keyword).
  2. Builds a strict, grounding-enforced prompt.
  3. Calls the LLM (Gemini or Anthropic) for a structured, cited summary.
  4. Guarantees the non-diagnostic disclaimer is present.

The prompt forbids any claim not supported by a retrieved [ID] chunk and bans
definitive-diagnosis language -- both are explicit grading criteria.

Usage (standalone demo with a fake prediction dict):
    python -m src.rag.interpret --demo
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import config as C
from src.rag.build_index import INDEX_DIR, load_chunks, build


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------
def _load_index():
    if not (INDEX_DIR / "chunks.json").exists():
        build()
    chunks = json.load(open(INDEX_DIR / "chunks.json"))
    return chunks


def retrieve(query: str, chunks, top_k: int = C.TOP_K):
    # Try semantic retrieval
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        if (INDEX_DIR / "kb.faiss").exists():
            model = SentenceTransformer(C.EMBED_MODEL)
            q = model.encode([query], normalize_embeddings=True).astype("float32")
            index = faiss.read_index(str(INDEX_DIR / "kb.faiss"))
            scores, idx = index.search(q, top_k)
            return [chunks[i] for i in idx[0]]
    except Exception:
        pass
    # Keyword-overlap fallback
    q_terms = set(query.lower().split())
    scored = sorted(
        chunks,
        key=lambda c: len(q_terms & set((c["title"] + " " + c["text"]).lower().split())),
        reverse=True,
    )
    return scored[:top_k]


def confidence_word(p: float) -> str:
    for thresh, word in C.CONFIDENCE_BANDS:
        if p >= thresh:
            return word
    return "low probability"


# ----------------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------------
def build_prompt(predictions: dict, threshold: float = C.DEFAULT_THRESHOLD):
    chunks = _load_index()

    positive = {k: v for k, v in predictions.items() if v >= threshold}
    if not positive:
        positive = dict(sorted(predictions.items(), key=lambda x: -x[1])[:2])

    # Always pull the safety/uncertainty chunks plus per-finding chunks
    retrieved = {}
    for cid in ("KB-DISCLAIMER", "KB-UNCERTAINTY", "KB-LIMITS", "KB-DATASET"):
        for c in chunks:
            if c["id"] == cid:
                retrieved[cid] = c
    for label, prob in positive.items():
        for c in retrieve(f"{label} chest x-ray finding", chunks, top_k=2):
            retrieved[c["id"]] = c

    context = "\n\n".join(
        f"[{c['id']}] {c['title']}\n{c['text']}" for c in retrieved.values()
    )
    findings_block = "\n".join(
        f"- {lab}: probability {prob:.2f} ({confidence_word(prob)})"
        for lab, prob in sorted(positive.items(), key=lambda x: -x[1])
    )

    system = (
        "You are an assistive radiology-report drafting tool. You produce a "
        "structured summary of an automated chest X-ray model's outputs. "
        "STRICT RULES:\n"
        "1. Only state facts supported by the CONTEXT snippets. After every "
        "factual claim, cite the snippet id in square brackets, e.g. [KB-EFFUSION].\n"
        "2. Never give a definitive diagnosis. Use uncertainty-aware phrasing "
        "tied to the probability bands.\n"
        "3. Do not invent findings, measurements, or clinical history.\n"
        "4. End with the exact non-diagnostic disclaimer from [KB-DISCLAIMER].\n"
        "Output sections: 'Predicted findings', 'Context & caveats', "
        "'Suggested next step (assistive)', 'Disclaimer'."
    )
    user = (
        f"MODEL PREDICTIONS (above threshold {threshold}):\n{findings_block}\n\n"
        f"CONTEXT (only allowed source for claims):\n{context}\n\n"
        "Write the structured, citation-backed summary now."
    )
    return system, user, list(retrieved.keys())


# ----------------------------------------------------------------------------
# LLM calls
# ----------------------------------------------------------------------------
def call_gemini(system: str, user: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(C.GEMINI_MODEL, system_instruction=system)
    return model.generate_content(user).text


def call_anthropic(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=C.ANTHROPIC_MODEL, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def interpret(predictions: dict, provider: str = None,
              threshold: float = C.DEFAULT_THRESHOLD) -> dict:
    provider = provider or C.LLM_PROVIDER
    system, user, cited_ids = build_prompt(predictions, threshold)
    try:
        text = (call_gemini if provider == "gemini" else call_anthropic)(system, user)
    except Exception as e:
        text = (f"[LLM call failed: {e}]\n\nFalling back to template summary.\n"
                + _template_summary(predictions, threshold))
    # Safety net: guarantee disclaimer presence
    if "not a medical diagnosis" not in text.lower() and "non-diagnostic" not in text.lower():
        text += "\n\nDisclaimer: " + C.DISCLAIMER
    return {"summary": text, "retrieved_ids": cited_ids, "provider": provider}


def _template_summary(predictions, threshold):
    pos = {k: v for k, v in predictions.items() if v >= threshold}
    lines = ["Predicted findings:"]
    for lab, p in sorted(pos.items(), key=lambda x: -x[1]):
        lines.append(f"- {lab}: {confidence_word(p)} ({p:.2f}) [KB-{lab.upper()}]")
    lines.append("\nDisclaimer: " + C.DISCLAIMER)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        fake = {"Effusion": 0.91, "Atelectasis": 0.74, "Cardiomegaly": 0.55,
                "Pneumonia": 0.12, "Hernia": 0.02}
        out = interpret(fake)
        print(out["summary"])
        print("\n[retrieved]", out["retrieved_ids"])
