"""Gemini-powered chatbot for live program editing on the program viewer page."""

import json
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are LiftAI Coach, an expert strength training assistant embedded in the LiftAI program viewer.

The user has a generated training program in front of them. You help them:
1. Live-edit the program (change exercises, sets, reps, RPE, cues) using tools
2. Answer training questions about why certain choices were made
3. Explain the structure of the program and the science behind it
4. Suggest improvements and apply them immediately if the user agrees

When the user asks you to change something, use the appropriate tool to make the edit, then confirm what you changed.
If something is unclear, ask a clarifying question before making an edit.
Keep replies concise. You are a coach, not a textbook.

Current program:
{program_json}
"""

_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="edit_exercise",
        description="Edit one or more fields of an existing exercise in the training program.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "day": {
                    "type": "STRING",
                    "description": "The exact day key as it appears in the program (e.g. 'Monday: Upper A')"
                },
                "exercise_index": {
                    "type": "INTEGER",
                    "description": "0-based index of the exercise within that day"
                },
                "name": {
                    "type": "STRING",
                    "description": "New exercise name (optional, omit to keep current)"
                },
                "sets": {
                    "type": "INTEGER",
                    "description": "New number of sets (optional)"
                },
                "reps": {
                    "type": "STRING",
                    "description": "New rep range, e.g. '8-12' or '5-8' (optional)"
                },
                "target_rpe": {
                    "type": "STRING",
                    "description": "New RPE target, e.g. '8-9' or '7-8' (optional)"
                },
                "cues": {
                    "type": "STRING",
                    "description": "New coaching cues (optional)"
                },
                "rest": {
                    "type": "STRING",
                    "description": "New rest period, e.g. '90-120 seconds' (optional)"
                },
            },
            "required": ["day", "exercise_index"],
        },
    ),
    types.FunctionDeclaration(
        name="add_exercise",
        description="Add a new exercise to a training day.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "day": {
                    "type": "STRING",
                    "description": "The exact day key to add the exercise to"
                },
                "name": {"type": "STRING", "description": "Exercise name"},
                "sets": {"type": "INTEGER", "description": "Number of sets"},
                "reps": {"type": "STRING", "description": "Rep range, e.g. '8-12'"},
                "target_rpe": {"type": "STRING", "description": "RPE target, e.g. '8-9'"},
                "cues": {"type": "STRING", "description": "Coaching cues"},
                "rest": {"type": "STRING", "description": "Rest period, e.g. '90 seconds'"},
            },
            "required": ["day", "name", "sets", "reps", "target_rpe"],
        },
    ),
    types.FunctionDeclaration(
        name="remove_exercise",
        description="Remove an exercise from a training day.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "day": {
                    "type": "STRING",
                    "description": "The exact day key"
                },
                "exercise_index": {
                    "type": "INTEGER",
                    "description": "0-based index of the exercise to remove"
                },
            },
            "required": ["day", "exercise_index"],
        },
    ),
]


class ProgramChatbot:
    def __init__(self, model_name: str, client: genai.Client):
        self.model_name = model_name
        self.client = client
        self._tool = types.Tool(function_declarations=_FUNCTION_DECLARATIONS)

    def _apply_function_call(self, program: dict, fn_name: str, args: dict) -> tuple[dict, str]:
        """Execute a function call against the program dict. Returns (updated_program, result_text)."""
        program = {day: list(exs) for day, exs in program.items()}  # shallow copy

        if fn_name == "edit_exercise":
            day = args.get("day")
            idx = args.get("exercise_index", 0)
            if day not in program or idx >= len(program[day]):
                return program, f"Error: day '{day}' or exercise index {idx} not found."
            ex = dict(program[day][idx])
            for field in ("name", "sets", "reps", "target_rpe", "cues", "rest"):
                if field in args and args[field] is not None:
                    ex[field] = args[field]
            program[day][idx] = ex
            return program, f"Updated '{ex['name']}' on {day}."

        if fn_name == "add_exercise":
            day = args.get("day")
            if day not in program:
                return program, f"Error: day '{day}' not found."
            new_ex = {
                "name": args.get("name", "New Exercise"),
                "sets": args.get("sets", 3),
                "reps": args.get("reps", "8-12"),
                "target_rpe": args.get("target_rpe", "8-9"),
                "cues": args.get("cues", ""),
                "rest": args.get("rest", "90 seconds"),
            }
            program[day].append(new_ex)
            return program, f"Added '{new_ex['name']}' to {day}."

        if fn_name == "remove_exercise":
            day = args.get("day")
            idx = args.get("exercise_index", 0)
            if day not in program or idx >= len(program[day]):
                return program, f"Error: day '{day}' or exercise index {idx} not found."
            removed = program[day].pop(idx)
            return program, f"Removed '{removed['name']}' from {day}."

        return program, f"Unknown function: {fn_name}"

    def chat(self, message: str, program: dict, history: list = None) -> dict:
        """
        Send a message and get back a reply + optional updated program.

        Parameters
        ----------
        message : str
            The user's chat message.
        program : dict
            Current weekly_program dict {day: [exercises]}.
        history : list
            Prior conversation turns as [{'role': 'user'/'model', 'parts': [{'text': ...}]}].

        Returns
        -------
        dict with keys:
            reply        : str  — assistant text reply
            updated_program : dict | None — modified program, or None if unchanged
            function_results : list — descriptions of edits made
        """
        history = history or []
        updated_program = None
        function_results = []

        system = _SYSTEM_PROMPT.format(program_json=json.dumps(program, indent=2))

        # Build contents list: history + new user message
        contents = list(history) + [{"role": "user", "parts": [{"text": message}]}]

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[self._tool],
            temperature=0.4,
            max_output_tokens=1024,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            logger.exception("Chatbot Gemini call failed")
            return {"reply": f"Sorry, I encountered an error: {e}", "updated_program": None, "function_results": []}

        # Collect all parts from the response
        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Process any function calls
        current_program = dict(program)
        fn_call_parts = []
        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                current_program, result_text = self._apply_function_call(current_program, fc.name, args)
                function_results.append(result_text)
                fn_call_parts.append((fc.name, args, result_text))
                updated_program = current_program

        # If there were function calls, do a follow-up to get a natural text reply
        if fn_call_parts:
            # Build function response parts
            fn_response_parts = []
            for fn_name, args, result_text in fn_call_parts:
                fn_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result_text},
                    )
                )
            # Add model turn + function results to contents for follow-up
            follow_up_contents = contents + [
                {"role": "model", "parts": parts},
                {"role": "user", "parts": fn_response_parts},
            ]
            try:
                follow_up = self.client.models.generate_content(
                    model=self.model_name,
                    contents=follow_up_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.4,
                        max_output_tokens=512,
                    ),
                )
                reply = follow_up.text.strip()
            except Exception:
                reply = " ".join(function_results)
        else:
            reply = response.text.strip() if response.text else "I'm not sure how to help with that."

        return {
            "reply": reply,
            "updated_program": updated_program,
            "function_results": function_results,
        }
