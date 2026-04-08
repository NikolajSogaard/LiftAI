"""Configuration dataclass for a single critique task."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any


@dataclass
class CritiqueTask:
    """Represents a single critique task with its configuration.

    Attributes
    ----------
    name:
        Human-readable label for this task (e.g. "volume").
    template:
        Prompt template string. Formatted by the Critic at runtime.
    needs_retrieval:
        Whether to fetch supporting context from the RAG index.
    retrieval_query:
        Optional override query for RAG retrieval. Defaults to the user input.
    specialized_instructions:
        Extra instructions injected into the retrieval call.
    dependencies:
        Names of other tasks whose feedback should be prepended as context.
    reference_data:
        Arbitrary reference values (e.g. set ranges) used when building the prompt.
    """

    name: str
    template: str
    needs_retrieval: bool = True
    retrieval_query: Optional[str] = None
    specialized_instructions: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    reference_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CritiqueTask.name must not be empty")
        if not self.template:
            raise ValueError("CritiqueTask.template must not be empty")

    def get_context_from_dependencies(self, previous_results: Dict[str, Optional[str]]) -> str:
        """Return a block of context drawn from dependent task results.

        Parameters
        ----------
        previous_results:
            Mapping of task name → feedback string (or None if task passed).

        Returns
        -------
        str
            Newline-joined sentences, one per dependency that has a result.
            Empty string if no dependencies have results.
        """
        context = []
        for dep in self.dependencies:
            if dep in previous_results and previous_results[dep] is not None:
                context.append(f"Previous {dep.upper()} critique suggested: {previous_results[dep]}")
        return "\n".join(context)
