from dotenv import load_dotenv
import os
import json
import time
import logging
from google import genai
from google.genai import types
from config import EMBEDDING_RETRIES, EMBEDDING_RETRY_DELAY

logger = logging.getLogger(__name__)


def _get_client():
    """Return a configured genai Client, loading env vars once."""
    load_dotenv('cre.env')
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_GEMINI_API_KEY is missing from cre.env")
    return genai.Client(api_key=api_key)


def setup_llm(
        model: str,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        respond_as_json: bool = False,
        response_schema=None,
        thinking_budget: int | None = None,
):
    """Build and return a Gemini generation function.

    Parameters
    ----------
    respond_as_json : bool
        When True, sets response_mime_type="application/json" so the model is
        *guaranteed* to emit valid JSON — no markdown fences, no prose wrap.
    response_schema : Pydantic BaseModel subclass | None
        If provided, Gemini enforces this schema on the output (structured output).
        Only meaningful when respond_as_json=True.
    """
    client = _get_client()

    config_kwargs: dict = dict(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    if respond_as_json:
        # Guarantee clean JSON output — eliminates all markdown-fence / prose fallbacks
        config_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    config = types.GenerateContentConfig(**config_kwargs)

    def generate_response(prompt):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        text = response.text.strip()

        if not respond_as_json:
            return text

        # With response_mime_type="application/json" the model always outputs
        # valid JSON, so this should never raise — but we keep the fallback.
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode failed (unexpected): %s | Raw: %.200s", e, text)
            return {"weekly_program": {"Day 1": []}, "message": text}

    return generate_response


class _EmbeddingModel:
    """Thin wrapper around google.genai embeddings to keep embed_query / embed_documents API."""

    def __init__(self, client: "genai.Client", model: str) -> None:
        self._client = client
        self._model = model

    def embed_query(self, text: str) -> list[float]:
        """Embed a single string and return the embedding vector."""
        result = self._client.models.embed_content(model=self._model, contents=text)
        return result.embeddings[0].values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings and return a list of embedding vectors."""
        result = self._client.models.embed_content(model=self._model, contents=texts)
        return [e.values for e in result.embeddings]


def setup_embeddings(model: str = "gemini-embedding-001") -> _EmbeddingModel:
    """Set up and return an embedding model with retry logic.

    Parameters
    ----------
    model:
        The Gemini embedding model ID to use.

    Returns
    -------
    _EmbeddingModel
        A ready-to-use embedding wrapper.

    Raises
    ------
    ValueError
        If the embedding model fails to initialise after all retries.
    """
    client = _get_client()
    logger.info("Setting up embedding model: %s", model)

    retries, delay = EMBEDDING_RETRIES, EMBEDDING_RETRY_DELAY
    for attempt in range(retries):
        try:
            wrapper = _EmbeddingModel(client, model)
            wrapper.embed_query("test")
            logger.info("Embedding model ready: %s", model)
            return wrapper
        except Exception as e:
            logger.warning("Embedding init attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2

    raise ValueError(f"Embedding model init failed after {retries} attempts")
