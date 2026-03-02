"""
Note: For simplicity in the report, the final implementation round is described 
as occurring in the writer agent, but it actually happens here in the editor.
"""

import json

class Editor:
    def __init__(self, writer=None):
        self.writer = writer
        self.on_status = None

    def _emit(self, message):
        if self.on_status:
            self.on_status({"step": "editor", "message": message})

    def implement_final_feedback(self, program):
        """Run one last revision pass if there's unprocessed feedback."""
        if not program.get('feedback') or not self.writer:
            return program

        print("\n=== EDITOR: Implementing final round of feedback ===")
        try:
            week = program.get('week_number', 1)
            override_type = "progression" if week > 1 else "revision"
            print(f"Using {override_type} mode (Week {week})")
            program['draft'] = self.writer.revise(program, override_type=override_type)
            print("Final feedback applied")
        except Exception as e:
            print(f"Error implementing final feedback: {e}")
        return program



    def extract_weekly_program(self, data) -> dict:
        """Extract weekly_program dict from various nested/stringified formats."""
        if isinstance(data, dict):
            if 'weekly_program' in data:
                return data['weekly_program']
            if 'formatted' in data and isinstance(data['formatted'], dict):
                fmt = data['formatted']
                return fmt.get('weekly_program', fmt)
            if 'draft' in data:
                return self.extract_weekly_program(data['draft'])
            if 'message' in data and isinstance(data['message'], str):
                return self._try_parse_json(data['message'])

        elif isinstance(data, str):
            return self._try_parse_json(data)

        return {}

    def _try_parse_json(self, text):
        """Try to parse JSON from raw text or ```json blocks."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed.get('weekly_program', parsed)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        if "```json" in text:
            try:
                chunk = text.split("```json", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(chunk)
                if isinstance(parsed, dict):
                    return parsed.get('weekly_program', parsed)
            except (json.JSONDecodeError, IndexError):
                pass
        return {}

    def format_program(self, program: dict[str, str | None]) -> dict:
        """Validate and normalize the program into a consistent format for the web app."""
        weekly_program = self.extract_weekly_program(program['draft'])

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
