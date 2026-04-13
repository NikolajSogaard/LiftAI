# config.py
"""Central configuration — import from here instead of hardcoding values."""
import os

# ── LLM defaults ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_CRITIC_MODEL = "gemini-3-flash-preview"
DEFAULT_MAX_TOKENS = 8000
DEFAULT_WRITER_TEMPERATURE = 0.4
DEFAULT_WRITER_TOP_P = 0.9
DEFAULT_MAX_ITERATIONS = 1

# ── Timeouts (seconds) ───────────────────────────────────────────────────────
QUEUE_TIMEOUT = 120          # SSE queue wait in app.py
PDF_SUBPROCESS_TIMEOUT = 30  # per-page subprocess in build_db.py

# ── RAG ──────────────────────────────────────────────────────────────────────
FAISS_DIR = os.path.join("Data", "faiss_db")
BOOKS_DIR = os.path.join("Data", "books")
RAG_TOP_K = 8

# ── Reflexion ────────────────────────────────────────────────────────────────
LESSON_MAX_CHARS = 300       # truncation applied to distilled lessons

# ── Mesocycle & autoregulation ──────────────────────────────────────────────
DEFAULT_MESOCYCLE_LENGTH = 4         # weeks per training block
STAGNATION_THRESHOLD_WEEKS = 2      # consecutive weeks with no progress → flagged
FATIGUE_SCORE_DELOAD_TRIGGER = 0.7  # fatigue above this → deload
STALL_RATIO_REVIEW_TRIGGER = 0.5    # fraction of exercises stalled → mesocycle review
DELOAD_VOLUME_REDUCTION = 0.5       # reduce sets by ~50% during deload

# ── Embedding retry ──────────────────────────────────────────────────────────
EMBEDDING_RETRIES = 3
EMBEDDING_RETRY_DELAY = 2    # seconds; doubles on each retry

# ── Input validation ─────────────────────────────────────────────────────────
MAX_USER_INPUT_CHARS = 5000
MAX_CHAT_MESSAGE_CHARS = 2000
MAX_CHAT_HISTORY_TURNS = 50
