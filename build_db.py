"""Build the FAISS vector index from PDF training literature in Data/books/."""

import logging
import os
import sys
import json
import pickle
import subprocess
import fitz
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import FAISS_DIR, BOOKS_DIR, PDF_SUBPROCESS_TIMEOUT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_pdfs(path: str) -> list[str]:
    """Read all PDFs using a subprocess per page to avoid hangs."""
    script = r'''
import sys, json, fitz
pdf = fitz.open(sys.argv[1])
page = pdf[int(sys.argv[2])]
print(json.dumps(page.get_text()))
'''
    python = sys.executable
    docs = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".pdf"):
            continue
        fpath = os.path.join(path, fname)
        try:
            page_count = len(fitz.open(fpath))
        except Exception as e:
            logger.warning("SKIP %s: can't open (%s)", fname, e)
            continue

        pages, skipped = [], 0
        for i in range(page_count):
            try:
                r = subprocess.run(
                    [python, "-c", script, fpath, str(i)],
                    capture_output=True, text=True, timeout=PDF_SUBPROCESS_TIMEOUT,
                )
                if r.returncode == 0 and r.stdout.strip():
                    pages.append(json.loads(r.stdout.strip()))
                else:
                    skipped += 1
            except subprocess.TimeoutExpired:
                skipped += 1
            except Exception:
                skipped += 1

        text = "".join(pages)
        if text.strip():
            docs.append({"text": text, "metadata": {"source": fname}})
        suffix = f" (skipped {skipped}/{page_count} pages)" if skipped else ""
        logger.info("%s (%s pages)%s", fname, page_count, suffix)
    return docs


def chunk_documents(docs: list[str], chunk_size=1000, chunk_overlap=200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    for doc in docs:
        for piece in splitter.split_text(doc["text"]):
            chunks.append({"text": piece, "metadata": doc["metadata"]})
    return chunks


def build_faiss_index(chunks, embedding_model):
    """Embed all chunks and build a FAISS index. Returns (index, texts, metadatas)."""
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info("Embedding %s chunks (this may take a while)...", len(texts))
    batch_size = 50
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embedding_model.embed_documents(batch)
        all_vecs.extend(vecs)
        logger.info("Embedded %s/%s", min(i + batch_size, len(texts)), len(texts))

    matrix = np.array(all_vecs, dtype="float32")
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)          # inner-product (cosine after L2-norm)
    faiss.normalize_L2(matrix)              # normalise so IP == cosine similarity
    index.add(matrix)

    return index, texts, metadatas


def save_index(index, texts, metadatas, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)
    logger.info("Saved FAISS index + metadata to %s/", out_dir)


def build_index(chunks: list[str]) -> None:
    from agent_system.setup_api import setup_embeddings

    if not chunks:
        logger.warning("No chunks provided — nothing to index.")
        return

    embedding_model = setup_embeddings(model="gemini-embedding-001")
    index, texts, metadatas = build_faiss_index(chunks, embedding_model)
    save_index(index, texts, metadatas, FAISS_DIR)
    logger.info("Done — %s vectors in the index.", index.ntotal)


def main():
    from agent_system.setup_api import setup_embeddings

    if not os.path.exists(BOOKS_DIR):
        logger.warning("Directory '%s' not found.", BOOKS_DIR)
        return

    logger.info("Loading PDFs...")
    docs = load_pdfs(BOOKS_DIR)
    if not docs:
        logger.warning("No documents found.")
        return

    logger.info("Chunking %s documents...", len(docs))
    chunks = chunk_documents(docs)
    logger.info("Created %s chunks", len(chunks))

    embedding_model = setup_embeddings(model="gemini-embedding-001")
    index, texts, metadatas = build_faiss_index(chunks, embedding_model)

    save_index(index, texts, metadatas, FAISS_DIR)
    logger.info("Done — %s vectors in the index.", index.ntotal)


if __name__ == "__main__":
    try:
        build_index(chunk_documents(load_pdfs(BOOKS_DIR)))
        logger.info("FAISS index built successfully at %s", FAISS_DIR)
    except Exception:
        logger.exception("Fatal error building FAISS index")
        sys.exit(1)
