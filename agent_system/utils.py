"""Shared utilities for the agent system."""

import json
import logging

logger = logging.getLogger(__name__)


def parse_json_draft(data: "dict | str | None") -> dict:
    """Parse a weekly_program dict from various formats.

    Handles: raw dict, nested dict with 'weekly_program' / 'formatted' / 'draft'
    keys, plain JSON strings, and markdown ```json ... ``` blocks.
    Returns the weekly_program dict, or an empty dict if nothing can be parsed.
    """
    if isinstance(data, str):
        return _parse_json_string(data)

    if isinstance(data, dict):
        if 'weekly_program' in data:
            return data['weekly_program']
        if 'formatted' in data and isinstance(data['formatted'], dict):
            fmt = data['formatted']
            return fmt.get('weekly_program', fmt)
        if 'draft' in data:
            return parse_json_draft(data['draft'])
        if 'message' in data and isinstance(data['message'], str):
            return _parse_json_string(data['message'])

    return {}


def _parse_json_string(text: str) -> dict:
    """Try to parse weekly_program from a raw JSON string or a ```json block."""
    # Try markdown code block first (more specific)
    if "```json" in text:
        try:
            chunk = text.split("```json", 1)[1].split("```", 1)[0].strip()
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                return parsed.get('weekly_program', parsed)
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse markdown JSON block, trying direct parse")

    # Try direct JSON parse
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed.get('weekly_program', parsed)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON string: %.200s", stripped)

    return {}
