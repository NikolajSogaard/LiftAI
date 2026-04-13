from __future__ import annotations
import logging
from typing import Any
from langgraph.graph import StateGraph, START, END
from config import LESSON_MAX_CHARS

logger = logging.getLogger(__name__)

from .analytics import analyze_training_history
from .agents import (
    Writer,
    Critic,
    Editor,
    Analyst,
)


class ProgramGenerator:
    def __init__(
            self,
            writer: Writer,
            critic: Critic,
            editor: Editor,
            max_iterations: int = 3,
            ):
        self.writer = writer
        self.critic = critic

        if not hasattr(editor, 'writer') or editor.writer is None:
            editor.writer = writer
        self.editor = editor
        self.max_iterations = max_iterations
        self.on_status = None

        graph = StateGraph(dict)
        graph.add_node('writer', self.writer)
        graph.add_node('critic', self.critic)
        graph.add_node('reflector', self._reflect)
        graph.add_node('editor', self.editor)

        graph.add_edge(START, 'writer')
        graph.add_edge('writer', 'critic')
        graph.add_conditional_edges(
            source='critic',
            path=self.provide_critique,
            path_map={'accept': 'editor', 'reflect': 'reflector'},
        )
        graph.add_edge('reflector', 'writer')
        graph.add_edge('editor', END)
        self.app = graph.compile()

    def provide_critique(self, program: dict[str, Any]) -> str:
        program.setdefault('iteration_count', 0)
        program['iteration_count'] += 1
        if program['iteration_count'] >= self.max_iterations:
            logger.info("Max iterations reached (%d), accepting.", self.max_iterations)
            return 'accept'
        return 'accept' if program['feedback'] is None else 'reflect'

    def _reflect(self, program: dict[str, Any]) -> dict[str, Any]:
        """Reflexion: distil the critic feedback into a one-sentence lesson stored
        in memory so the writer can avoid repeating the same mistake next iteration."""
        feedback = program.get('feedback') or ''
        iteration = program.get('iteration_count', 1)
        prompt = (
            "You are reviewing critique feedback for a strength training program.\n\n"
            f"Critique:\n{feedback}\n\n"
            "In ONE concise sentence, state the single most important lesson "
            "the writer must remember when revising this program."
        )
        try:
            lesson = self.writer.model(prompt)
            if isinstance(lesson, str):
                lesson = lesson.strip()[:LESSON_MAX_CHARS]
            elif isinstance(lesson, dict):
                lesson = str(lesson)
        except Exception as e:
            logger.exception("Reflection failed")
            lesson = "Address all critic feedback systematically."

        lessons = program.get('lessons', [])
        lessons.append(f"Attempt {iteration}: {lesson}")
        program['lessons'] = lessons
        logger.info("=== REFLEXION (attempt %d) ===\n%s", iteration, lesson)
        if self.on_status:
            self.on_status({"step": "critic", "message": f"Reflexion: {lesson}"})
        return program

    def create_program(self, user_input: str) -> dict[str, Any]:
        """Run the full Writer → Critic → Editor LangGraph workflow.

        Parameters
        ----------
        user_input:
            The raw user prompt (may include persona info prepended by app.py).

        Returns
        -------
        dict
            Final LangGraph state dict. The formatted program is at
            ``result['formatted']['weekly_program']``.
        """
        # Propagate status callback to agents
        if self.on_status:
            self.writer.on_status = self.on_status
            self.critic.on_status = self.on_status
            self.editor.on_status = self.on_status

        program = {
            'user-input': user_input,
            'draft': None,
            'feedback': None,
            'formatted': None,
            'iteration_count': 0,
            'lessons': [],   # Reflexion memory — accumulated across revision attempts
        }
        return self.app.invoke(program)


class ProgressionProgramGenerator:
    """Week 2+ workflow: Analytics → [Analyst] → Writer → [Critic] → Editor.

    Handles three review types:
    - normal: skip Analyst, run progression Writer + progression Critic
    - deload: run Analyst, run deload Writer, skip Critic
    - mesocycle_review: run Analyst, pause for user approval, run new_block Writer + full Critic
    """

    def __init__(
        self,
        writer: "Writer",
        critic: "Critic",
        editor: "Editor",
        analyst: "Analyst",
        mesocycle_length: int = 4,
        max_iterations: int = 1,
    ):
        self.writer = writer
        self.critic = critic
        self.editor = editor
        self.analyst = analyst
        self.mesocycle_length = mesocycle_length
        self.max_iterations = max_iterations
        self.on_status = None

        if not hasattr(editor, 'writer') or editor.writer is None:
            editor.writer = writer

    def _propagate_status(self) -> None:
        """Push the on_status callback to all agents."""
        if self.on_status:
            self.writer.on_status = self.on_status
            self.critic.on_status = self.on_status
            self.editor.on_status = self.on_status
            self.analyst.on_status = self.on_status

    def create_program(
        self,
        user_input: str,
        current_mesocycle_history: list[dict],
        week_in_mesocycle: int,
        previous_block_summaries: list[dict] | None = None,
        feedback: dict | None = None,
        previous_draft: dict | None = None,
    ) -> dict:
        """Run the full progression workflow.

        Returns
        -------
        dict
            Final state dict. If review_type is 'deload' or 'mesocycle_review',
            state['analyst_decision'] contains the decision document and
            state['needs_approval'] is True — the caller must pause for user
            approval before calling continue_after_approval().
            If review_type is 'normal', the full pipeline runs and
            state['formatted'] contains the final program.
        """
        self._propagate_status()

        if self.on_status:
            self.on_status({"step": "analytics", "message": "Analyzing training history..."})

        state = {
            "user-input": user_input,
            "draft": previous_draft,
            "feedback": feedback,
            "formatted": None,
            "iteration_count": 0,
            "lessons": [],
            "current_mesocycle_history": current_mesocycle_history,
            "week_in_mesocycle": week_in_mesocycle,
            "previous_block_summaries": previous_block_summaries or [],
        }

        # Phase 1: Analytics
        analytics = analyze_training_history(
            weeks=current_mesocycle_history,
            week_in_mesocycle=week_in_mesocycle,
            mesocycle_length=self.mesocycle_length,
        )
        state["analytics"] = analytics
        state["exercise_flags"] = analytics.get("exercise_flags", {})

        review_type = analytics["review_type"]
        if self.on_status:
            self.on_status({
                "step": "analytics",
                "message": f"Analysis complete — review type: {review_type}",
            })
        logger.info("Analytics result: review_type=%s, triggers=%s", review_type, analytics["triggers"])

        if review_type == "normal":
            # Normal progression — straight to Writer → Critic → Editor
            state = self.writer(state)
            state = self.critic(state)
            state = self.editor(state)
            return state

        # Phase 2: Analyst (for deload or mesocycle_review)
        state = self.analyst(state)
        state["needs_approval"] = True
        return state

    def continue_after_approval(self, state: dict) -> dict:
        """Resume the workflow after user approves the analyst decision.

        Call this after the user reviews and approves the analyst_decision.

        Parameters
        ----------
        state:
            The state dict returned by create_program() with needs_approval=True.

        Returns
        -------
        dict
            Final state with 'formatted' containing the program.
        """
        self._propagate_status()
        state.pop("needs_approval", None)

        review_type = state["analytics"]["review_type"]

        # Writer phase
        state = self.writer(state)

        # Critic phase — skip for deloads
        if review_type != "deload":
            state = self.critic(state)

        # Editor phase
        state = self.editor(state)
        return state
