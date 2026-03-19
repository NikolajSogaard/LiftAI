# LiftAI — Claude Code Guide

## Project Overview

LiftAI is a multi-agent AI system that generates personalized strength training programs using RAG (Retrieval-Augmented Generation). It grounds recommendations in peer-reviewed training literature via a FAISS vector database.

**Architecture:** User → Flask API → LangGraph workflow (Writer → Critic → Editor) → JSON program

## Tech Stack

- **Backend:** Python, Flask, Flask-Session
- **Agent Orchestration:** LangGraph
- **LLM:** Google Gemini API (`google-genai` SDK) — default model `gemini-3-flash-preview`
- **RAG:** FAISS (vector search), LangChain (text splitting), PyMuPDF (PDF parsing)
- **Embeddings:** Google Gemini `gemini-embedding-001`
- **Frontend:** HTML/CSS/JS templates, Server-Sent Events (SSE) for real-time progress

## Project Structure

```
app.py                  # Flask web app — routes, SSE streaming, session management
build_db.py             # One-time FAISS database build from PDFs in Data/books/
rag_retrieval.py        # RAG engine — HyDE technique + CRAG filtering
agent_system/
  generator.py          # LangGraph workflow orchestrator
  agents/
    writer.py           # Program generation/revision with RAG
    critic.py           # Parallel 5-task critique
    editor.py           # Schema validation & final JSON formatting
    critique_task.py    # Critic task configuration
  setup_api.py          # Google Gemini API client (git-ignored)
  utils.py              # JSON parsing utilities
prompts/
  writer_prompts.py     # Writer role/task prompts
  critic_prompts.py     # Critic evaluation criteria
templates/
  generate.html         # Program generation form
  index.html            # Program viewer
Data/
  books/                # PDF training literature (git-ignored)
  faiss_db/             # Built FAISS index + metadata (git-ignored)
  personas/             # Test personas (personas_vers2.json)
  SavedPrograms/        # Generated programs stored as JSON (git-ignored)
```

## Setup

1. **API credentials** — create `cre.env`:
   ```
   GOOGLE_GEMINI_API_KEY="your-api-key"
   ```

2. **Add training literature** — place PDFs in `Data/books/`

3. **Build FAISS database** (one-time):
   ```bash
   python build_db.py
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

## Key Patterns

**LangGraph workflow:** `START → Writer → Critic → [Editor | back to Writer]`
- Reflexion pattern: Critic distills lessons that persist into subsequent Writer iterations
- Conditional routing based on critique pass/fail

**RAG pipeline:**
- HyDE: generates a hypothetical ideal answer, embeds that for retrieval (better semantic match)
- CRAG: parallel relevance grading of retrieved chunks, filters low-quality results

**JSON output:** Gemini is configured with `response_mime_type="application/json"` for structured output

**SSE streaming:** Background thread runs the LangGraph workflow; `on_status` callbacks push progress events to the frontend

**Lazy init + LRU caching:** FAISS index, embedding model, and LLMs are loaded on first use; query embeddings are cached

## Default Configuration (`app.py`)

```python
DEFAULT_CONFIG = {
    'model': 'gemini-3-flash-preview',         # Writer & Editor
    'critic_model': 'gemini-3-flash-preview',  # Critic
    'max_tokens': 8000,
    'writer_temperature': 0.4,
    'writer_top_p': 0.9,
    'writer_prompt_settings': 'v1',
    'critic_prompt_settings': 'week1',
    'max_iterations': 1,
    'thinking_budget': None,
}
```

## Notes

- `setup_api.py` is git-ignored (contains API client setup)
- `Data/` is almost entirely git-ignored except screenshots
- No automated test suite — testing is done via predefined personas in `Data/personas/personas_vers2.json`
- PDF extraction runs in a subprocess with a 30-second timeout to prevent hangs
