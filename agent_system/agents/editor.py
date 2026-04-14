"""
Note: For simplicity in the report, the final implementation round is described
as occurring in the writer agent, but it actually happens here in the editor.
"""

import json
import logging
from agent_system.utils import parse_json_draft

logger = logging.getLogger(__name__)

class Editor:
    def __init__(self, writer=None):
        self.writer = writer
        self.on_status = None

    def _emit(self, message):
        if self.on_status:
            self.on_status({"step": "editor", "message": message})

    def implement_final_feedback(self, program: dict) -> dict:
        """Run one last revision pass if there's unprocessed feedback.

        Parameters
        ----------
        program:
            The LangGraph state dict. Must have 'feedback', 'draft', and
            optionally 'week_number'.

        Returns
        -------
        dict
            The (possibly updated) program state.
        """
        if not program.get('feedback') or not self.writer:
            return program

        week = program.get('week_number') or program.get('week_in_mesocycle', 1)
        if week > 1:
            logger.info("Editor: skipping final revision for progression (week %d)", week)
            return program

        logger.info("Editor: implementing final round of feedback")
        self._emit("Implementing final feedback round...")
        try:
            logger.info("Using revision mode (Week %d)", week)
            program['draft'] = self.writer.revise(program, override_type="revision")
            logger.info("Final feedback applied successfully")
        except Exception as e:
            logger.exception("Error implementing final feedback")
        return program



    def extract_weekly_program(self, data: "dict | str | None") -> dict:
        """Extract weekly_program dict from various nested/stringified formats."""
        return parse_json_draft(data)

    def format_program(self, program: dict) -> dict:
        """Validate and normalise the program into a consistent format for the web app.

        Reads `program['draft']`, extracts the weekly_program structure, and
        normalises every exercise entry to have the expected fields with defaults.

        Parameters
        ----------
        program:
            LangGraph state dict. Must contain a 'draft' key.

        Returns
        -------
        dict
            ``{"weekly_program": {day: [exercise_dicts]}}``
        """
        weekly_program = self.extract_weekly_program(program['draft'])
        if not weekly_program:
            logger.warning("Editor.format_program: empty weekly_program extracted from draft")

        validated = {}
        for day, exercises in weekly_program.items():
            validated[day] = []
            for ex in exercises:
                entry = {
                    "name": ex.get("name", "Unnamed Exercise"),
                    "sets": ex.get("sets", 3),
                    "reps": ex.get("reps", "8-12"),
                    "target_rpe": ex.get("target_rpe", "7-8"),
                    "rest": ex.get("rest", "60-90 seconds"),
                    "cues": ex.get("cues", "Focus on proper form")
                }
                # Carry over progression suggestions if present
                suggestion = (
                    ex.get("AI Progression")
                    or ex.get("suggestion")
                    or ex.get("ai progression")
                )
                if suggestion:
                    entry["suggestion"] = suggestion
                validated[day].append(entry)

        return {"weekly_program": validated}

    def __call__(self, program: dict[str, str | None]) -> dict[str, str | None]:
        # First implement any final feedback
        self._emit("Implementing final feedback...")
        program = self.implement_final_feedback(program)
        self._emit("Formatting final program output...")
        formatted = self.format_program(program)
        if 'feedback' in program:
            formatted['critic_feedback'] = program['feedback']
        if 'week_number' in program:
            formatted['week_number'] = program['week_number']
        program['formatted'] = formatted
        return program
