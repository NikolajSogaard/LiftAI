import json
import logging
import re
from typing import Dict, Optional, Callable
from rag_retrieval import retrieve_and_generate
from agent_system.utils import parse_json_draft
from config import LESSON_MAX_CHARS

logger = logging.getLogger(__name__)

class Writer:
    def __init__(
            self,
            model,
            role: dict[str, str],
            structure: str,
            task: Optional[str] = None,
            task_revision: Optional[str] = None,
            task_progression: Optional[str] = None,
            writer_type: str = "initial",
            retrieval_fn: Optional[Callable] = None,
            ):
        self.model = model
        self.role = role
        self.structure = structure
        self.task = task
        self.task_revision = task_revision
        self.task_progression = task_progression
        self.writer_type = writer_type
        self.retrieval_fn = retrieval_fn or retrieve_and_generate
        self.on_status = None
        
        self.specialized_instructions = {
            "initial": "Focus on creating the best strength training program based on the {user_input}. Consider appropriate training splits and frequency, rep-ranges, exercises that fits the user and the order of these exercises, set volume for each week and intensity.",
            "revision": "Focus on how to improve and correct an existing strength training program based on specific critique feedback. Retrieve evidence-based guidance on fixing the identified issues while preserving the parts that work well.",
            "progression": "Focus on progressive overload principles and autoregulation strategies. Retrieve guidance on when to increase weight versus reps, how to interpret RPE feedback, and how to adjust training load week over week.",
            "deload": "Focus on deload week programming. Retrieve guidance on how to reduce training volume while maintaining movement patterns for recovery.",
            "new_block": "Focus on mesocycle transitions and exercise rotation. Retrieve guidance on how to set up the first week of a new training block with fresh exercises.",
        }
    
    def _emit(self, message, detail=False):
        if self.on_status:
            payload = {"step": "writer", "message": message}
            if detail:
                payload["detail"] = True
            self.on_status(payload)

    def get_retrieval_query(self, program: dict[str, str | None]) -> str:
        user_input = program.get('user-input', '')
        if self.writer_type == "initial":
            return f"Best practices for designing a strength training program for someone with these goals and preferences: {user_input}"
        if self.writer_type == "progression":
            return f"Progressive overload principles and autoregulation strategies for strength training: {user_input}"
        if self.writer_type == "revision":
            return f"How to revise and improve a strength training program based on feedback: {user_input}"
        if self.writer_type == "deload":
            return "How to program deload weeks in strength training — volume and intensity reduction protocols"
        if self.writer_type == "new_block":
            return f"How to structure mesocycle transitions, exercise rotation and fresh training blocks for continued progression: {user_input}"
        return f"Strength training program design best practices: {user_input}"

    def format_previous_week_program(self, program: dict) -> str:
        """Extract and format the previous week's program as JSON for progression prompts."""
        from .editor import Editor
        editor = Editor()
        
        prev = None
        if 'formatted' in program and isinstance(program['formatted'], dict):
            prev = program['formatted'].get('weekly_program')
        if prev is None and 'draft' in program:
            prev = editor.extract_weekly_program(program['draft'])
        if prev is None:
            prev = editor.extract_weekly_program(program)
        
        if prev:
            return json.dumps({"weekly_program": prev}, indent=2)
        return json.dumps(program, indent=2) if isinstance(program, dict) else str(program)

    def _build_prompt(self, parts):
        """Flatten a list of role/content dicts into a single prompt string."""
        return "\n".join(
            item.get("content", "") if isinstance(item, dict) else str(item)
            for item in parts
        )

    def write(self, program: dict) -> dict:
        """Generate the initial training program draft.

        Retrieves relevant context from the RAG index, builds a prompt from
        the user input and writer role, calls the LLM, and stores the result
        in ``program['draft']``.

        Parameters
        ----------
        program:
            LangGraph state dict. Must contain 'user-input'.

        Returns
        -------
        dict
            Updated state with 'draft' populated.
        """
        query = self.get_retrieval_query(program)
        user_input = program.get('user-input', '')
        instructions = self.specialized_instructions.get(self.writer_type, "")
        if '{user_input}' in instructions:
            instructions = instructions.format(user_input=user_input)
        if not self.task:
            raise ValueError(f"Writer '{self.writer_type}' has no task for initial creation")

        enhanced_task = self.task
        if query:
            logger.info("Writer retrieving context for %s", self.writer_type)
            self._emit("Retrieving context from training literature...")
            result, _ = self.retrieval_fn(query, instructions)
            enhanced_task += f"\nRelevant context from training literature:\n{result}\n"
            self._emit(f"Retrieved context:\n{result.strip()}", detail=True)

        prompt = self._build_prompt([
            self.role,
            {'role': 'user', 'content': enhanced_task.format(program['user-input'], self.structure)},
        ])

        logger.info("Generating initial program...")
        self._emit("Generating initial program draft...")
        draft = self.model(prompt)
        if isinstance(draft, str):
            draft = {"weekly_program": {"Day 1": []}, "message": draft}
        return draft

    def revise(self, program: dict, override_type: str | None = None) -> dict:
        """Revise an existing program draft based on critic feedback.

        Supports three modes via ``override_type`` or ``self.writer_type``:
        - ``revision``: standard feedback-driven rewrite of the draft
        - ``progression``: update suggestions only, preserving structure

        Parameters
        ----------
        program:
            LangGraph state dict with 'draft', 'feedback', and optionally
            'lessons' and 'week_number'.
        override_type:
            Force a specific writer mode. Defaults to ``self.writer_type``.

        Returns
        -------
        dict
            The updated draft (weekly_program dict).
        """
        current_type = override_type or self.writer_type

        # Resolve revision task
        revision_task = self.task_revision if current_type in ("revision", "progression") else None
        if not revision_task:
            from prompts.writer_prompts import TASK_REVISION
            revision_task = TASK_REVISION
        if not revision_task:
            raise ValueError(f"No revision task available for writer type '{current_type}'")
        
        is_progression = (current_type == "progression")
        previous_program_formatted = None
        
        if is_progression:
            logger.info("Progression mode: maintaining structure, updating suggestions")
            if self.task_progression is not None:
                revision_task = self.task_progression
                previous_program_formatted = self.format_previous_week_program(program)
                # Append format reminder
                revision_task += (
                    "\n\nFINAL FORMAT REMINDER:\n"
                    "Your response for each exercise MUST contain ONLY:\n"
                    "- One line per set with performance data: Set X:(Y reps @ Zkg, RPE W)\n"
                    "- One line with just the adjustment: [number]kg ↑ or [number] reps ↓\n"
                    "- NO additional text or explanations whatsoever\n"
                )
        
        # RAG retrieval for revision/progression context
        query = self.get_retrieval_query(program)
        if query:
            logger.info("Writer retrieving context for %s", current_type)
            self._emit("Retrieving context from training literature...")
            try:
                result, _ = self.retrieval_fn(query, self.specialized_instructions.get(current_type, ""))
                revision_task += f"\nRelevant context from training literature:\n{result}\n"
                self._emit(f"Retrieved context:\n{result.strip()}", detail=True)
            except Exception:
                logger.warning("RAG retrieval failed in revise(), continuing without context", exc_info=True)

        # Build prompt — prepend Reflexion lessons if any exist
        if is_progression and previous_program_formatted:
            content = revision_task.format(previous_program_formatted, program['feedback'], self.structure)
        else:
            draft_str = json.dumps(program['draft'], indent=2) if isinstance(program['draft'], dict) else program['draft']
            content = revision_task.format(draft_str, program['feedback'], self.structure)

        lessons = program.get('lessons', [])
        if lessons:
            lessons_block = "Key lessons from previous revision attempts (do NOT repeat these mistakes):\n"
            lessons_block += "\n".join(f"- {l}" for l in lessons)
            content = lessons_block + "\n\n" + content
        
        prompt = self._build_prompt([self.role, {'role': 'user', 'content': content}])

        logger.info("Revising program...")
        self._emit("Revising program based on critic feedback...")
        try:
            draft = self.model(prompt)

            # Merge progression suggestions back into original structure
            if is_progression and isinstance(draft, dict) and 'weekly_program' in draft:
                draft = self._merge_progression(program, draft)

            # Handle string responses
            if isinstance(draft, str):
                draft = self._parse_string_draft(draft)

        except Exception as e:
            logger.exception("Error during revision")
            draft = {"weekly_program": {"Day 1": []}, "message": f"Error: {e}"}

        # Single-pass normalization for progression format
        if is_progression and isinstance(draft, dict) and 'weekly_program' in draft:
            self._normalize_progression_suggestions(draft['weekly_program'], program)

        return draft

    def write_deload(self, program: dict) -> dict:
        """Generate a deload week — same exercises, reduced volume.

        Parameters
        ----------
        program:
            LangGraph state dict. Must contain 'draft' (previous week's program)
            and 'analyst_decision' with the deload plan.

        Returns
        -------
        dict
            The deload program draft.
        """
        if not self.task:
            raise ValueError("Writer 'deload' has no task template")

        analyst_decision = program.get("analyst_decision", {})
        previous_program = self.format_previous_week_program(program)

        query = self.get_retrieval_query(program)
        enhanced_task = self.task
        if query:
            self._emit("Retrieving context for deload programming...")
            try:
                result, _ = self.retrieval_fn(query, self.specialized_instructions.get("deload", ""))
                enhanced_task += f"\nRelevant context from training literature:\n{result}\n"
            except Exception:
                logger.warning("RAG retrieval failed for deload writer", exc_info=True)

        prompt = self._build_prompt([
            self.role,
            {'role': 'user', 'content': enhanced_task.format(
                previous_program,
                json.dumps(analyst_decision, indent=2),
                self.structure,
            )},
        ])

        self._emit("Generating deload week program...")
        draft = self.model(prompt)
        if isinstance(draft, str):
            draft = self._parse_string_draft(draft)
        return draft

    def write_new_block(self, program: dict) -> dict:
        """Generate Week 1 of a new mesocycle based on analyst recommendations.

        Parameters
        ----------
        program:
            LangGraph state dict. Must contain 'draft', 'analyst_decision',
            and optionally 'previous_block_summaries'.

        Returns
        -------
        dict
            The new block's Week 1 program draft.
        """
        if not self.task:
            raise ValueError("Writer 'new_block' has no task template")

        analyst_decision = program.get("analyst_decision", {})
        previous_program = self.format_previous_week_program(program)
        block_summaries = json.dumps(program.get("previous_block_summaries", []), indent=2)

        query = self.get_retrieval_query(program)
        enhanced_task = self.task
        if query:
            self._emit("Retrieving context for new training block...")
            try:
                result, _ = self.retrieval_fn(query, self.specialized_instructions.get("new_block", ""))
                enhanced_task += f"\nRelevant context from training literature:\n{result}\n"
            except Exception:
                logger.warning("RAG retrieval failed for new block writer", exc_info=True)

        prompt = self._build_prompt([
            self.role,
            {'role': 'user', 'content': enhanced_task.format(
                previous_program,
                json.dumps(analyst_decision, indent=2),
                block_summaries,
                self.structure,
            )},
        ])

        self._emit("Generating new training block Week 1...")
        draft = self.model(prompt)
        if isinstance(draft, str):
            draft = self._parse_string_draft(draft)
        return draft

    # --- Helper methods for revision post-processing ---

    def _merge_progression(self, program, draft):
        """Merge new progression suggestions into the original program structure."""
        original = None
        if isinstance(program.get('draft'), dict) and 'weekly_program' in program['draft']:
            original = program['draft']['weekly_program']
        if original is None:
            return draft

        merged = {}
        new_prog = draft['weekly_program']
        for day, orig_exercises in original.items():
            merged[day] = []
            for i, orig_ex in enumerate(orig_exercises):
                ex = orig_ex.copy()
                if day in new_prog and i < len(new_prog[day]):
                    new_ex = new_prog[day][i]
                    suggestion = new_ex.get("AI Progression") or new_ex.get("suggestion")
                    if suggestion:
                        ex["AI Progression"] = suggestion
                        ex["suggestion"] = suggestion
                merged[day].append(ex)
        draft['weekly_program'] = merged
        return draft

    def _normalize_progression_suggestions(self, weekly_program, program):
        """Single-pass normalization: clean, enforce format, and sync suggestion fields."""
        for day, exercises in weekly_program.items():
            for ex in exercises:
                # Prefer 'AI Progression', fall back to 'suggestion'
                val = ex.get("AI Progression") or ex.get("suggestion")
                if not val or not isinstance(val, str):
                    continue

                normalized = self._extract_and_format_suggestion(val, program, day, exercises, ex)
                if normalized:
                    ex["AI Progression"] = normalized
                    ex["suggestion"] = normalized

    def _extract_and_format_suggestion(self, val, program, day, exercises, ex):
        """Parse a suggestion string into canonical Set X:(...) + adjustment format."""
        lines = val.strip().split('\n')

        # Fast path: already well-formed (starts with "Set 1:" and has parens)
        if val.strip().startswith("Set 1:") and "(" in val:
            perf = [l.strip() for l in lines if l.strip().startswith("Set ")]
            adj = next((l.strip() for l in lines if l.strip() and not l.strip().startswith("Set ")), None)
            if adj:
                perf.append(adj)
            return "\n".join(perf) if perf else None

        # Try to extract Set X: lines and adjustment from a messier string
        cleaned, adjustment = [], None
        for line in lines:
            line = line.strip()
            if line.startswith("Set ") and "(" in line:
                cleaned.append(line)
            elif any(m in line for m in ["kg ↑", "kg ↓", "reps ↑", "reps ↓"]):
                adjustment = line
                break

        if cleaned:
            if adjustment:
                cleaned.append(adjustment)
            return "\n".join(cleaned)

        # Last resort: regex-extract rep/weight numbers and recover perf lines
        rep_m = re.search(r'(\d+)\s*reps?', val)
        wt_m = re.search(r'(\d+(?:\.\d+)?)\s*kg', val)
        if not (rep_m or wt_m):
            return None

        perf_lines = self._get_original_perf_lines(program, day, exercises, ex)
        if rep_m and "reps" in val.lower():
            adj = f"        {rep_m.group(1)} reps ↑"
        elif wt_m:
            adj = f"        {wt_m.group(1)}kg ↑"
        else:
            adj = "        Maintain current weight and reps"
        return "\n".join(perf_lines + [adj])

    def _parse_string_draft(self, text: str) -> dict:
        """Try to parse a string draft into a dict, with fallback."""
        parsed = parse_json_draft(text)
        if parsed:
            return {"weekly_program": parsed}
        return {"weekly_program": {"Day 1": []}, "message": text}


    def _get_original_perf_lines(self, program, day, exercises, exercise):
        """Look up original performance data for an exercise from the previous draft."""
        if not isinstance(program.get('draft'), dict):
            return ["Set 1:(Performance data unavailable)"]
        orig_prog = program['draft'].get('weekly_program', {})
        if day not in orig_prog:
            return ["Set 1:(Performance data unavailable)"]
        idx = next((i for i, e in enumerate(exercises) if e is exercise), -1)
        if 0 <= idx < len(orig_prog[day]):
            orig_val = orig_prog[day][idx].get('AI Progression', '')
            if isinstance(orig_val, str) and orig_val.strip().startswith("Set 1:"):
                return [l.strip() for l in orig_val.strip().split('\n') if l.strip().startswith("Set ")]
        return ["Set 1:(Performance data unavailable)"]

    def __call__(self, program: dict[str, str | None]) -> dict[str, str | None]:
        if self.writer_type == "deload":
            logger.info("Deload writer")
            draft = self.write_deload(program)
        elif self.writer_type == "new_block":
            logger.info("New block writer (mesocycle transition)")
            draft = self.write_new_block(program)
        elif self.writer_type == "progression" and 'feedback' in program:
            logger.info("Progression writer (Week 2+)")
            draft = self.revise(program, override_type="progression")
        elif program.get('draft') is None:
            logger.info("Initial program creation")
            draft = self.write(program)
        elif 'feedback' in program:
            logger.info("Revising based on feedback")
            draft = self.revise(program, override_type="revision")
        else:
            logger.info("Fallback: initial write")
            draft = self.write(program)

        program['draft'] = draft
        return program
