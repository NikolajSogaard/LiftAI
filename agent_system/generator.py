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
        graph.add_node('editor', self.editor)

        graph.add_edge(START, 'writer')
        graph.add_edge('writer', 'critic')
        graph.add_conditional_edges(
            source='critic',
            path=self.provide_critique,
            path_map={'accept': 'editor', 'revise': 'writer'},
        )
        graph.add_edge('editor', END)
        self.app = graph.compile()

    def provide_critique(self, program: dict[str, Any]) -> str:
        program.setdefault('iteration_count', 0)
        program['iteration_count'] += 1
        if program['iteration_count'] >= self.max_iterations:
            print(f"Max iterations reached ({self.max_iterations}), accepting.")
            return 'accept'
        return 'accept' if program['feedback'] is None else 'revise'

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
        }
        return self.app.invoke(program)
