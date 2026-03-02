from dotenv import load_dotenv
import os
import json
import time
from google import genai
from google.genai import types


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
):
    client = _get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    def generate_response(prompt):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        text = response.text.strip()
        if not respond_as_json:
            return text
        try:
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif not (text.startswith("{") and text.endswith("}")):
                print("Plain text response, wrapping as JSON")
                return {"weekly_program": {"Day 1": []}, "message": text}
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON decode failed: {e}\nRaw: {text[:200]}")
            return {"weekly_program": {"Day 1": []}, "message": text}

    return generate_response


class _EmbeddingModel:
    """Thin wrapper around google.genai embeddings to keep embed_query / embed_documents API."""

    def __init__(self, client, model):
        self._client = client
        self._model = model

    def embed_query(self, text):
        result = self._client.models.embed_content(model=self._model, contents=text)
        return result.embeddings[0].values

    def embed_documents(self, texts):
        result = self._client.models.embed_content(model=self._model, contents=texts)
        return [e.values for e in result.embeddings]


def setup_embeddings(model="gemini-embedding-001"):
    """Set up and return an embedding model with retry logic."""
    client = _get_client()
    print(f"Setting up embedding model: {model}")

    retries, delay = 3, 2
    for attempt in range(retries):
        try:
            wrapper = _EmbeddingModel(client, model)
            wrapper.embed_query("test")
            print(f"Embedding model ready: {model}")
            return wrapper
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2

    raise ValueError(f"Embedding model init failed after {retries} attempts")
