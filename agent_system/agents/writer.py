import json
import re
from typing import Dict, Optional, Callable
from rag_retrieval import retrieve_and_generate

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
            "initial": "Focus on creating the best strength training program based on the {user_input}. Consider appropriate training splits and frequency, rep-ranges, exercises that fits the user and the order of these exercises, set volume for each week and intensity."
        }
    
    def _emit(self, message, detail=False):
        if self.on_status:
            payload = {"step": "writer", "message": message}
            if detail:
                payload["detail"] = True
            self.on_status(payload)

    def get_retrieval_query(self, program: dict[str, str | None]) -> str:
        if self.writer_type == "initial":
            return "Best practices for designing a strength training program based on {user_input} and preferences."
        return ""

    def format_previous_week_program(self, program: dict[str, str | None]) -> str:
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

    def write(self, program: dict[str, str | None]):
        query = self.get_retrieval_query(program)
        user_input = program.get('user-input', '')
        instructions = self.specialized_instructions.get(self.writer_type, "")
        if '{user_input}' in instructions:
            instructions = instructions.format(user_input=user_input)
        if not self.task:
            raise ValueError(f"Writer '{self.writer_type}' has no task for initial creation")

        enhanced_task = self.task
        if self.writer_type == "initial" and query:
            print("\n--- Writer retrieving context ---")
            self._emit("Retrieving context from training literature...")
            result, _ = self.retrieval_fn(query, instructions)
            enhanced_task += f"\nRelevant context from training literature:\n{result}\n"
            # Show what the RAG retrieved
            self._emit(f"Retrieved context:\n{result.strip()}", detail=True)
        
        prompt = self._build_prompt([
            self.role,
            {'role': 'user', 'content': enhanced_task.format(program['user-input'], self.structure)},
        ])
        
        print("Generating initial program...")
        self._emit("Generating initial program draft...")
        draft = self.model(prompt)
        if isinstance(draft, str):
            draft = {"weekly_program": {"Day 1": []}, "message": draft}
        return draft

    def revise(self, program: dict[str, str | None], override_type: str = None):
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
            print("Progression mode: maintaining structure, updating suggestions")
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
        
        # Build prompt
        if is_progression and previous_program_formatted:
            content = revision_task.format(previous_program_formatted, program['feedback'], self.structure)
        else:
            content = revision_task.format(program['draft'], program['feedback'], self.structure)
        
        prompt = self._build_prompt([self.role, {'role': 'user', 'content': content}])

        print("Revising program...")
        self._emit("Revising program based on critic feedback...")
        try:
            draft = self.model(prompt)
            
            # Merge progression suggestions back into original structure
            if is_progression and isinstance(draft, dict) and 'weekly_program' in draft:
                draft = self._merge_progression(program, draft)
            
            # Clean up suggestion formatting
            if isinstance(draft, dict) and 'weekly_program' in draft:
                self._clean_suggestions(draft['weekly_program'])
            
            # Handle string responses
            if isinstance(draft, str):
                draft = self._parse_string_draft(draft)

        except Exception as e:
            print(f"Error during revision: {e}")
            draft = {"weekly_program": {"Day 1": []}, "message": f"Error: {e}"}
        
        # Final pass for progression format enforcement
        if is_progression and isinstance(draft, dict) and 'weekly_program' in draft:
            self._enforce_progression_format(draft, program)
            self._sync_suggestion_fields(draft['weekly_program'])

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

    def _clean_suggestions(self, weekly_program):
        """Extract set-data and adjustment lines from suggestion text."""
        for day, exercises in weekly_program.items():
            for ex in exercises:
                suggestion = ex.get("suggestion")
                if not suggestion or not isinstance(suggestion, str):
                    continue
                lines = suggestion.split('\n')
                cleaned, adjustment = [], None
                for line in lines:
                    line = line.strip()
                    if line.startswith("Set ") and "(" in line:
                        cleaned.append(line)
                    elif any(m in line for m in ["kg ↑", "kg ↓", "reps ↑", "reps ↓"]):
                        adjustment = line
                        break
                if adjustment:
                    cleaned.append(adjustment)
                if cleaned:
                    formatted = "\n".join(cleaned)
                    ex["suggestion"] = formatted
                    if "AI Progression" in ex:
                        ex["AI Progression"] = formatted

    def _parse_string_draft(self, draft_str):
        """Try to parse a string draft into a dict, with fallback."""
        try:
            if "```json" in draft_str:
                chunk = draft_str.split("```json", 1)[1].split("```", 1)[0].strip()
                return json.loads(chunk)
            elif draft_str.strip().startswith("{") and draft_str.strip().endswith("}"):
                return json.loads(draft_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed: {e}")
        return {"weekly_program": {"Day 1": []}, "message": draft_str}

    def _enforce_progression_format(self, draft, program):
        """Ensure progression suggestions follow the Set X:(...) + adjustment format."""
        for day, exercises in draft['weekly_program'].items():
            for ex in exercises:
                for field in ("AI Progression", "suggestion"):
                    val = ex.get(field)
                    if not val or not isinstance(val, str):
                        continue
                    
                    is_formatted = val.strip().startswith("Set 1:") and "(" in val
                    lines = val.strip().split('\n')
                    
                    if is_formatted:
                        perf = [l.strip() for l in lines if l.strip().startswith("Set ")]
                        adj = next((l.strip() for l in lines if l.strip() and not l.strip().startswith("Set ")), None)
                        if adj:
                            perf.append(adj)
                        ex[field] = "\n".join(perf)
                    else:
                        rep_m = re.search(r'(\d+)\s*reps?', val)
                        wt_m = re.search(r'(\d+(?:\.\d+)?)\s*kg', val)
                        if not (rep_m or wt_m):
                            continue
                        # Try to recover original performance lines
                        perf_lines = self._get_original_perf_lines(program, day, exercises, ex)
                        if rep_m and "reps" in val.lower():
                            adj = f"        {rep_m.group(1)} reps ↑"
                        elif wt_m:
                            adj = f"        {wt_m.group(1)}kg ↑"
                        else:
                            adj = "        Maintain current weight and reps"
                        ex[field] = "\n".join(perf_lines + [adj])

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

    def _sync_suggestion_fields(self, weekly_program):
        """Keep 'AI Progression' and 'suggestion' in sync."""
        for day, exercises in weekly_program.items():
            for ex in exercises:
                if ex.get("AI Progression"):
                    ex["suggestion"] = ex["AI Progression"]
                elif ex.get("suggestion"):
                    ex["AI Progression"] = ex["suggestion"]

    def __call__(self, program: dict[str, str | None]) -> dict[str, str | None]:
        if self.writer_type == "progression" and 'feedback' in program:
            print("Progression writer (Week 2+)")
            draft = self.revise(program, override_type="progression")
        elif program.get('draft') is None:
            print("Initial program creation")
            draft = self.write(program)
        elif 'feedback' in program:
            print("Revising based on feedback")
            draft = self.revise(program, override_type="revision")
        else:
            print("Fallback: initial write")
            draft = self.write(program)

        program['draft'] = draft
        return program
