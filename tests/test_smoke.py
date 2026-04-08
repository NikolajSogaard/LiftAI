"""Smoke test — verifies program generation runs end-to-end without crashing.

Run with:
    pytest tests/test_smoke.py -v -m integration

Requires a valid cre.env with GOOGLE_GEMINI_API_KEY set.
"""

import json
import os
import pytest

# Skip automatically if no API key is present
pytestmark = pytest.mark.integration


@pytest.fixture()
def first_persona() -> str:
    """Load the first persona from the test personas file."""
    personas_path = os.path.join("Data", "personas", "personas_vers2.json")
    with open(personas_path) as f:
        data = json.load(f)
    personas = data["Personas"]
    first_key = next(iter(personas))
    return f"Generate a strength training program.\nTarget Persona: {personas[first_key]}"


def test_program_generation_smoke(first_persona: str) -> None:
    """Full pipeline smoke: Writer → Critic → Editor produces a valid program."""
    from agent_system import setup_llm, Writer, Critic, Editor, ProgramGenerator
    from prompts import WRITER_PROMPT_SETTINGS, CRITIC_PROMPT_SETTINGS
    from config import DEFAULT_MODEL

    writer = Writer(
        model=setup_llm(DEFAULT_MODEL, max_tokens=8000, temperature=0.4, respond_as_json=True),
        prompt_settings=WRITER_PROMPT_SETTINGS["v1"],
    )
    critic = Critic(
        model=setup_llm(DEFAULT_MODEL, max_tokens=8000, respond_as_json=False),
        prompt_settings=CRITIC_PROMPT_SETTINGS["week1"],
    )
    editor = Editor()
    generator = ProgramGenerator(writer=writer, critic=critic, editor=editor, max_iterations=1)

    result = generator.create_program(first_persona)

    assert result is not None, "create_program returned None"
    assert "formatted" in result, "Result missing 'formatted' key"
    weekly = result["formatted"].get("weekly_program", {})
    assert len(weekly) > 0, "weekly_program is empty"
    first_day = next(iter(weekly.values()))
    assert isinstance(first_day, list), "Day exercises should be a list"
    assert len(first_day) > 0, "First day has no exercises"
