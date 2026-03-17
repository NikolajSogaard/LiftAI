from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Callable, List
from rag_retrieval import retrieve_and_generate, retrieve_context
from .critique_task import CritiqueTask

class Critic:
    def __init__(
            self,
            model,
            role: dict[str, str],
            tasks: Dict[str, str] = None, 
            retrieval_fn: Optional[Callable] = None,
            ):
        self.model = model
        self.role = role
        self.tasks = tasks or {}
        self.retrieval_fn = retrieval_fn or retrieve_and_generate
        self.on_status = None

        self._task_labels = {
            "frequency_and_split": "Evaluating training frequency and split",
            "exercise_selection": "Reviewing exercise selection",
            "set_volume": "Checking weekly set volume",
            "rep_ranges": "Analyzing rep ranges",
            "rpe": "Assessing RPE targets",
            "progression": "Evaluating progression strategy",
        }
        
        self.specialized_instructions = {
            "frequency_and_split": "Provide concise guidance tailored to the user's training goals. Focus on structuring workout frequency and splits to ensure balanced coverage of muscle groups and key movement patterns. Adapt recommendations based on the user's training experience (beginner or advanced), specialization (e.g., bodybuilding or powerlifting), and overall objectives",
            "exercise_selection": "Provide concise guidance. Retrieve information about exercise selection principles based on specific user goals, experience level, and any physical limitations. Provide the answer as a list of exercises for each goal and muscle group ",            
            "rpe": "Provide consise guidance, and do not answer outside the scope of the query. Retrieve information about appropriate RPE (Rating of Perceived Exertion) targets for different exercise types and experience levels. Include guidance on when to use absolute RPE values (like 8) versus RPE ranges (like 7-8), and how RPE should differ between compound and isolation exercises.",
            "rep_ranges": "Provide concise guidance on rep ranges for different exercises, experience levels and goals. Include information on optimal rep ranges for compound and isolation exercises, as well as how rep ranges can vary based on strength, hypertrophy, or endurance goals.",
            "progression": "Focus on clear decision-making between weight or rep increases. Provide specific guidance on when to increase weight versus when to increase reps based on RPE, performance data, and position within the target rep range. For RPE below target range, consider weight increases if the user is in the middle/upper end of the rep range, but favor rep increases if the user is at the lower end of their rep range. Always consider the prescribed rep range when deciding between weight or rep increases."
        }
        
        # Determine task types based on available tasks
        if tasks and "progression" in tasks and len(tasks) == 1:
            self.task_types = ["progression"]
            self.is_week2plus = True
        else:
            self.task_types = ["frequency_and_split", "exercise_selection", "set_volume", "rep_ranges", "rpe"]
            self.is_week2plus = False

        self._init_task_configs()

    def _init_task_configs(self):
        """Build task configuration objects for each critique type."""
        volume_guidelines = {
            "beginner": {
                "Upper_horizontal_push": {"min": 6, "max": 10, "description": "Chest/pressing"},
                "Upper_horizontal_pull": {"min": 6, "max": 10, "description": "Rows/rear back"},
                "Upper_vertical_push": {"min": 6, "max": 10, "description": "Overhead/shoulders"},
                "Upper_vertical_pull": {"min": 6, "max": 10, "description": "Pull-ups/lats"},
                "Lower_anterior_chain": {"min": 6, "max": 10, "description": "Quads"},
                "Lower_posterior_chain": {"min": 6, "max": 10, "description": "Glutes/Hams"}
            },
            "intermediate": {
                "Upper_horizontal_push": {"min": 10, "max": 16, "description": "Chest/pressing"},
                "Upper_horizontal_pull": {"min": 10, "max": 16, "description": "Rows/rear back"},
                "Upper_vertical_push": {"min": 8, "max": 14, "description": "Overhead/shoulders"},
                "Upper_vertical_pull": {"min": 10, "max": 18, "description": "Pull-ups/lats"},
                "Lower_anterior_chain": {"min": 10, "max": 16, "description": "Quads"},
                "Lower_posterior_chain": {"min": 10, "max": 16, "description": "Glutes/Hams"}
            },
            "advanced": {
                "Upper_horizontal_push": {"min": 10, "max": 16, "description": "Chest/pressing"},
                "Upper_horizontal_pull": {"min": 10, "max": 16, "description": "Rows/rear back"},
                "Upper_vertical_push": {"min": 8, "max": 14, "description": "Overhead/shoulders"},
                "Upper_vertical_pull": {"min": 10, "max": 18, "description": "Pull-ups/lats"},
                "Lower_anterior_chain": {"min": 10, "max": 16, "description": "Quads"},
                "Lower_posterior_chain": {"min": 10, "max": 16, "description": "Glutes/Hams"}
            }
        }

        self.task_configs = {
            "frequency_and_split": CritiqueTask(
                name="frequency_and_split",
                template=self.tasks.get("frequency_and_split", ""),
                needs_retrieval=True,
                retrieval_query="How do I structure a training plan for {user_input}? What are good training frequency and training splits for strength training programs?",
                specialized_instructions=self.specialized_instructions.get("frequency_and_split", ""),
                dependencies=[],
            ),
            "exercise_selection": CritiqueTask(
                name="exercise_selection",
                template=self.tasks.get("exercise_selection", ""),
                needs_retrieval=True,
                retrieval_query="What exercises are most effective and appropriate for different muscle groups based on: {user_input}. Give 3 exampel exercises for each movement pattern: Upper: Horizontal Push (Chest/pressing), Upper: Horizontal Pull (Rows/rear back), Upper: Vertical Push (Overhead/shoulders), Upper: Vertical Pull (Pull-ups/lats), Lower: Anterior Chain (Quads), Lower: Posterior Chain (Glutes/Hams)",
                specialized_instructions=self.specialized_instructions.get("exercise_selection", ""),
                dependencies=["frequency_and_split"],
            ),
            "set_volume": CritiqueTask(
                name="set_volume",
                template=self.tasks.get("set_volume", ""),
                needs_retrieval=False,
                dependencies=["frequency_and_split", "exercise_selection"],
                reference_data={"volume_guidelines": volume_guidelines}
            ),
            "rep_ranges": CritiqueTask(
                name="rep_ranges",
                template=self.tasks.get("rep_ranges", ""),
                needs_retrieval=True,
                retrieval_query="What are optimal rep ranges for specific exercises and for different strength training goals?",
                specialized_instructions=self.specialized_instructions.get("rep_ranges", ""),
                dependencies=["frequency_and_split", "exercise_selection", "set_volume"],
            ),
            "rpe": CritiqueTask(
                name="rpe",
                template=self.tasks.get("rpe", ""),
                needs_retrieval=True,
                retrieval_query="How should RPE targets be assigned in strength training for different types exercises and experience levels?",
                specialized_instructions=self.specialized_instructions.get("rpe", ""),
                dependencies=["frequency_and_split", "exercise_selection", "set_volume", "rep_ranges"],
            ),
            "progression": CritiqueTask(
                name="progression",
                template=self.tasks.get("progression", ""),
                needs_retrieval=True,
                retrieval_query="What are the best practices for progressive overload, and when should weight be increased/decreasing versus reps? Come with consise guidance on how to choose between increasing/decreasing weight versus increasing reps for progressive overload. When should I prioritize rep increases/decreasing over weight increases if the user is at the lower end of their target rep range? How should RPE feedback influence whether to add weight or reps?",
                specialized_instructions=self.specialized_instructions.get("progression", ""),
                dependencies=[],
            )
        }

    def _emit(self, message, detail=False):
        if self.on_status:
            payload = {"step": "critic", "message": message}
            if detail:
                payload["detail"] = True
            self.on_status(payload)

    def get_task_query(self, program: dict[str, str | None], task_type: str) -> str:
        """Build a retrieval query appropriate for the given task type."""
        if self.is_week2plus and task_type == "progression":
            return "What are the best practices for progressive overload, and when should weight be increased/decreasing versus reps? Come with consise guidance on how to choose between increasing/decreasing weight versus increasing reps for progressive overload. When should I prioritize rep increases/decreasing over weight increases if the user is at the lower end of their target rep range? How should RPE feedback influence whether to add weight or reps?"
        
        queries = {
            "frequency_and_split": "How do I structure a training plan for {user_input}? What are good training frequency and training splits for strength training programs?",
            "exercise_selection": "What exercises are most effective and appropriate for different muscle groups based on: {user_input}. Give 3 exampel exercises for each movement pattern: Upper: Horizontal Push (Chest/pressing), Upper: Horizontal Pull (Rows/rear back), Upper: Vertical Push (Overhead/shoulders), Upper: Vertical Pull (Pull-ups/lats), Lower: Anterior Chain (Quads), Lower: Posterior Chain (Glutes/Hams)",            
            "rep_ranges": "What are optimal rep ranges for specific exercises and for different strength training goals?",
            "rpe": "How should RPE targets be assigned in strength training for different types exercises and experience levels?",
        }
        return queries.get(task_type, f"Best practices for {task_type} in strength training programs")

    def run_single_critique(self, program, task_type, previous_results=None):
        """Run one critique pass with its own RAG retrieval."""
        previous_results = previous_results or {}
        label = self._task_labels.get(task_type, task_type.replace('_', ' ').title())
        self._emit(f"{label}...")
        print(f"\n--- Running {task_type.upper()} critique ---")
        
        task_config = self.task_configs.get(task_type)
        if not task_config:
            task_config = CritiqueTask(
                name=task_type, template=self.tasks.get(task_type, ""),
                needs_retrieval=True,
                retrieval_query=f"Best practices for {task_type} in strength training programs",
                specialized_instructions="", dependencies=[],
            )
        
        dependency_context = task_config.get_context_from_dependencies(previous_results)
        
        # Include volume reference data if applicable
        ref_context = ""
        if task_type == "set_volume" and task_config.reference_data.get("volume_guidelines"):
            ref_context = "\nVolume guidelines from reference data:\n"
            for level, muscles in task_config.reference_data["volume_guidelines"].items():
                ref_context += f"\n{level.capitalize()} level:\n"
                for muscle, ranges in muscles.items():
                    ref_context += f"- {muscle.capitalize()}: {ranges.get('min', '?')}-{ranges.get('max', '?')} sets per week\n"
        
        # Retrieve context if task needs it
        context = ""
        if task_config.needs_retrieval:
            retrieval_query = task_config.retrieval_query
            if "{user_input}" in retrieval_query:
                retrieval_query = retrieval_query.format(user_input=program.get('user-input', ''))
            result, _ = self.retrieval_fn(retrieval_query, task_config.specialized_instructions)
            context = f"\nRelevant context from training literature:\n{result}\n"
        
        if ref_context:
            context = ref_context + "\n" + context
        if dependency_context:
            context = f"\nConsiderations from previous critiques:\n{dependency_context}\n{context}"
        # Serialize program content
        program_content = program.get('draft')
        if isinstance(program_content, dict) and 'weekly_program' in program_content:
            import json
            program_content = json.dumps(program_content, indent=2)
        
        task_template = self.tasks.get(task_type)
        if task_template is None:
            task_template = f'''
            Your colleague has written the following training program:
            {{}}
            For an individual who provided the following input:
            {{}}
            Focus specifically on the {task_type.upper()}. Provide feedback if any... otherwise only return "None"
            '''
        
        # Format prompt differently for week 2+ progression
        if self.is_week2plus and task_type == "progression":
            week = program.get('week_number', 2)
            feedback_data = program.get('feedback', '{}')
            task_template = task_template.replace("{week_number}", str(week))
            prompt_content = task_template.format(program_content, program.get('user-input', ''), feedback_data) + context
        else:
            prompt_content = task_template.format(program_content, program.get('user-input', '')) + context
        
        full_prompt = f"{self.role.get('content', '')}\n\n{prompt_content}"
        
        print(f"Generating {task_type} critique...")
        self._emit(f"Generating {label} critique...")
        try:
            feedback = self.model(full_prompt)
            return feedback or None
        except Exception as e:
            print(f"Error in {task_type.upper()} critique: {e}")
            return f"Error in {task_type} critique: {e}"

    def _process_task_result(self, task_type: str, feedback) -> tuple[str, str | None]:
        """Validate and clean a single task's feedback. Returns (task_type, processed|None)."""
        label = self._task_labels.get(task_type, task_type.replace('_', ' ').title())
        if not feedback or not isinstance(feedback, str) or len(feedback.strip()) <= 10:
            print(f"{task_type.upper()} - No significant feedback")
            self._emit(f"{label}: No issues found ✓", detail=True)
            return task_type, None

        processed = feedback.rstrip("None").strip() if feedback.strip().endswith("None") else feedback
        if len(processed.strip()) <= 10:
            self._emit(f"{label}: No issues found ✓", detail=True)
            return task_type, None

        if 'no changes' in processed.lower() or 'therefore, no changes' in processed.lower():
            print(f"{task_type.upper()} - No changes needed")
            self._emit(f"{label}: No changes needed ✓", detail=True)
            return task_type, None

        return task_type, processed

    def critique(self, program: dict[str, str | None]) -> dict[str, str | None]:
        """Run all critique tasks in parallel, each with its own RAG retrieval."""
        print("\n========== CRITIQUE PROCESS STARTED ==========")

        # Run all tasks concurrently — tasks are independent (dependency context is
        # informational only and not required for correctness).
        with ThreadPoolExecutor(max_workers=len(self.task_types)) as executor:
            futures = {
                executor.submit(self.run_single_critique, program, task_type, {}): task_type
                for task_type in self.task_types
            }
            raw_results: dict[str, str | None] = {}
            for future in as_completed(futures):
                task_type = futures[future]
                try:
                    raw_results[task_type] = future.result()
                except Exception as e:
                    print(f"Error in {task_type.upper()} critique: {e}")
                    raw_results[task_type] = None

        # Process results in original task order for consistent output
        all_feedback = []
        for task_type in self.task_types:
            label = self._task_labels.get(task_type, task_type.replace('_', ' ').title())
            _, processed = self._process_task_result(task_type, raw_results.get(task_type))
            if processed is None:
                continue

            all_feedback.append(f"[{task_type.upper()} FEEDBACK]:\n{processed}\n")
            self._emit(f"{label} feedback:\n{processed}", detail=True)
            print(f"\n{'='*50}\n{task_type.upper()} CRITIQUE:\n{'='*50}")
            line = ""
            for word in processed.split():
                if len(line) + len(word) > 80:
                    print(line)
                    line = word + " "
                else:
                    line += word + " "
            if line:
                print(line)
            print(f"{'='*50}\n")

        if not all_feedback:
            print('No actionable feedback from any critique task')
            self._emit("Critique complete — no actionable feedback")
            return {'feedback': None}

        combined = "\n".join(all_feedback)
        self._emit(f"Critique complete — {len(all_feedback)} area(s) with suggestions")
        print(f"\n========== CRITIQUE COMPLETE ({len(all_feedback)}/{len(self.task_types)}) ==========")
        return {'feedback': combined}

    def __call__(self, article: dict[str, str | None]) -> dict[str, str | None]:
        article.update(self.critique(article))
        return article
