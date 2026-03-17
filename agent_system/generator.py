from __future__ import annotations
from typing import Any
from langgraph.graph import StateGraph, START, END

from .agents import (
    Writer,
    Critic,
    Editor,
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
            print(f"Max iterations reached ({self.max_iterations}), accepting.")
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
                lesson = lesson.strip()[:300]
            elif isinstance(lesson, dict):
                lesson = str(lesson)
        except Exception as e:
            print(f"Reflection failed: {e}")
            lesson = "Address all critic feedback systematically."

        lessons = program.get('lessons', [])
        lessons.append(f"Attempt {iteration}: {lesson}")
        program['lessons'] = lessons
        print(f"\n=== REFLEXION (attempt {iteration}) ===\n{lesson}\n")
        if self.on_status:
            self.on_status({"step": "critic", "message": f"Reflexion: {lesson}"})
        return program

    def create_program(self, user_input: str) -> dict[str, Any]:
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
