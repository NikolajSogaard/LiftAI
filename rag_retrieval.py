import logging
import os
import pickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import faiss
import numpy as np

from config import FAISS_DIR, RAG_TOP_K

logger = logging.getLogger(__name__)

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
        _generate_response = setup_llm(model="gemini-3-flash-preview", max_tokens=1000, temperature=0.3)
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


def _embed_uncached(text: str) -> np.ndarray:
    """Embed without caching — used for HyDE hypothetical documents."""
    return np.array(_get_embedding_model().embed_query(text), dtype="float32")


def _generate_hypothetical_document(query: str) -> str:
    """Generate a hypothetical ideal answer for *query* using HyDE technique.

    Embeds this synthetic answer rather than the raw query for significantly
    better semantic retrieval — the embedding space for answers is denser and
    more discriminative than for questions.
    """
    llm = _get_generate_response()
    prompt = (
        "You are a strength training researcher. "
        "Write a concise expert passage (3-5 sentences) from a strength training "
        "textbook or peer-reviewed paper that would directly and specifically answer:\n\n"
        f'"{query}"\n\n'
        "Write only the expert content — no preamble, no attribution, no meta-commentary."
    )
    return llm(prompt)


def _grade_chunk(chunk: str, query: str, grader) -> bool:
    """CRAG: grade whether a chunk is relevant to the query using the fast model."""
    prompt = (
        "Is the following text relevant to answering this query?\n\n"
        f"Query: {query}\n\n"
        f"Text: {chunk[:500]}\n\n"
        "Reply with only 'yes' or 'no'."
    )
    try:
        result = grader(prompt).strip().lower()
        return result.startswith('y')
    except Exception:
        logger.warning("CRAG grading failed for chunk, dropping it", exc_info=True)
        return False


def _crag_filter(hits: list, query: str) -> list:
    """Filter retrieved chunks by relevance using the CRAG technique.

    Runs a lightweight LLM relevance check on each chunk in parallel.
    Chunks graded irrelevant are dropped. If all chunks are filtered,
    returns the original hits as a fallback.
    """
    grader = _get_generate_response()

    def _grade(hit):
        return _grade_chunk(hit[0], query, grader)

    with ThreadPoolExecutor(max_workers=min(len(hits), 4)) as ex:
        grades = list(ex.map(_grade, hits))

    relevant = [h for h, keep in zip(hits, grades) if keep]
    if not relevant:
        logger.warning("CRAG: all chunks graded irrelevant — keeping top 3 as fallback")
        return hits[:3]
    removed = len(hits) - len(relevant)
    if removed:
        logger.info("CRAG: filtered %d/%d irrelevant chunks", removed, len(hits))
    return relevant


def retrieve_context(query: str, k: int = RAG_TOP_K, use_hyde: bool = True, use_crag: bool = True):
    """Retrieve top-k relevant chunks from FAISS.

    When use_hyde=True (default), embeds a hypothetical ideal answer instead of
    the raw query for significantly better semantic retrieval (HyDE technique).
    Results are ranked by cosine similarity score, not chunk length.
    """
    index, texts, metadatas = _get_index()

    if use_hyde:
        try:
            embed_text = _generate_hypothetical_document(query)
        except Exception as e:
            logger.warning("HyDE generation failed, falling back to direct query embedding: %s", e)
            embed_text = query
        vec = _embed_uncached(embed_text).copy().reshape(1, -1)
    else:
        vec = _embed_query(query).copy().reshape(1, -1)

    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, k)

    # Rank by actual cosine similarity score (higher = more relevant)
    hits = [
        (texts[i], metadatas[i], float(scores[0][rank]))
        for rank, i in enumerate(indices[0])
        if i != -1
    ]
    hits.sort(key=lambda x: x[2], reverse=True)

    # CRAG: filter out irrelevant chunks before passing to the LLM
    if use_crag and len(hits) > 1:
        hits = _crag_filter(hits, query)

    context = "\n\n".join(t for t, _, _ in hits)
    summary = context[:200] + "..." if len(context) > 200 else context
    sources = [{"content": t[:100] + "...", "metadata": m} for t, m, _ in hits]
    return context, summary, sources


def retrieve_and_generate(query: str, specialized_instructions: str = "") -> tuple[str, str]:
    """Retrieve relevant context and synthesise a response using Gemini."""
    context, summary, sources = retrieve_context(query)
    prompt = f"""You are a specialized strength training expert.
{specialized_instructions}

Using the following excerpts from strength training books and programs, provide a focused,
evidence-based answer. Include practical recommendations and clear explanations.
Do not answer outside the scope of the query.

Summary of Retrieved Information:
{summary}

Context:
{context}

Query: {query}

Answer:"""
    return _get_generate_response()(prompt), sources
