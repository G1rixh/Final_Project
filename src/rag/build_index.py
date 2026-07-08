"""
Build the RAG retrieval index from knowledge_base.md.

Chunks the file on each `## [ID] ...` heading so every chunk keeps its citation
ID. Embeds with sentence-transformers and stores a FAISS index. If FAISS or the
embedding model is unavailable, a keyword-overlap retriever is used instead so
the pipeline still runs end-to-end.

Usage:
    python -m src.rag.build_index            # builds + saves index
"""
import json
import re
from pathlib import Path

import numpy as np

import config as C

KB_PATH = Path(__file__).parent / "knowledge_base.md"
INDEX_DIR = C.OUTPUT_DIR / "rag_index"


def load_chunks(kb_path: Path = KB_PATH):
    text = kb_path.read_text()
    # Split on headings like "## [KB-XXX] Title"
    parts = re.split(r"\n(?=## \[)", text)
    chunks = []
    for p in parts:
        m = re.match(r"## \[([^\]]+)\]\s*(.*)", p.strip())
        if not m:
            continue
        cid, title = m.group(1), m.group(2).splitlines()[0]
        body = p.strip().split("\n", 1)[1].strip() if "\n" in p.strip() else ""
        chunks.append({"id": cid, "title": title, "text": body})
    return chunks


def build(kb_path: Path = KB_PATH):
    C.ensure_dirs()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(kb_path)
    json.dump(chunks, open(INDEX_DIR / "chunks.json", "w"), indent=2)

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        model = SentenceTransformer(C.EMBED_MODEL)
        embs = model.encode([c["text"] for c in chunks],
                            normalize_embeddings=True).astype("float32")
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, str(INDEX_DIR / "kb.faiss"))
        np.save(INDEX_DIR / "embeddings.npy", embs)
        print(f"[rag] built FAISS index with {len(chunks)} chunks "
              f"(dim={embs.shape[1]})")
    except Exception as e:
        print(f"[rag] embeddings/FAISS unavailable ({e}); "
              f"keyword fallback will be used at query time.")
    return chunks


if __name__ == "__main__":
    build()
