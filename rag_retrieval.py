import os
import pickle
from functools import lru_cache
import faiss
import numpy as np

FAISS_DIR = os.path.join("Data", "faiss_db")

_embedding_model = None
_generate_response = None
_faiss_index = None
_doc_texts = None
_doc_metadatas = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from agent_system.setup_api import setup_embeddings
        _embedding_model = setup_embeddings(model="gemini-embedding-001")
    return _embedding_model


def _get_generate_response():
    global _generate_response
    if _generate_response is None:
        from agent_system.setup_api import setup_llm
        _generate_response = setup_llm(model="gemini-2.5-flash", max_tokens=1000, temperature=0.3)
    return _generate_response


def _get_index(directory=FAISS_DIR):
    """Lazy-load the FAISS index and doc metadata on first use."""
    global _faiss_index, _doc_texts, _doc_metadatas
    if _faiss_index is None:
        _faiss_index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "docs.pkl"), "rb") as f:
            store = pickle.load(f)
        _doc_texts = store["texts"]
        _doc_metadatas = store["metadatas"]
    return _faiss_index, _doc_texts, _doc_metadatas


@lru_cache(maxsize=128)
def _embed_query(query: str) -> np.ndarray:
    """Embed a query string; cached so identical queries skip the API call."""
    return np.array(_get_embedding_model().embed_query(query), dtype="float32")


def retrieve_context(query, k=8):
    """Retrieve top-k relevant chunks from FAISS. Returns (context, summary, sources)."""
    index, texts, metadatas = _get_index()
    vec = _embed_query(query).copy().reshape(1, -1)  # copy: normalize_L2 mutates in-place
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, k)

    # rerank by chunk length (longer = more informative)
    hits = sorted(
        [(texts[i], metadatas[i], scores[0][rank]) for rank, i in enumerate(indices[0]) if i != -1],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    context = "\n\n".join(t for t, _, _ in hits)
    summary = context[:200] + "..." if len(context) > 200 else context
    sources = [{"content": t[:100] + "...", "metadata": m} for t, m, _ in hits]
    return context, summary, sources


def retrieve_and_generate(query, specialized_instructions=""):
    """Retrieve relevant context and generate a response using Gemini."""
    context, summary, sources = retrieve_context(query)
    prompt = f"""You are a specialized strength training expert.
{specialized_instructions}

Using the following excerpts from strength training books, programs.
Include practical advice recommendations, and clear explanations.
Do not answer outside the scope of the query.

Summary of Retrieved Information:
{summary}

Context:
{context}

Query: {query}

Answer:"""
    return _get_generate_response()(prompt), sources
