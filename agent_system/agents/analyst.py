"""Analyst agent — interprets analytics metrics into a structured decision document."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from prompts.analyst_prompts import (
    ANALYST_ROLE,
    ANALYST_DECISION_STRUCTURE,
    ANALYST_DELOAD_STRUCTURE,
    TASK_MESOCYCLE_REVIEW,
    TASK_DELOAD,
)
from rag_retrieval import retrieve_and_generate

logger = logging.getLogger(__name__)


class Analyst:
    """Produces a decision document (exercise swaps, deload plan, volume changes)
    based on computed analytics and training history."""

    def __init__(
        self,
        model,
        retrieval_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.role = ANALYST_ROLE
        self.retrieval_fn = retrieval_fn or retrieve_and_generate
        self.on_status = None

    def _emit(self, message: str, detail: bool = False) -> None:
        if self.on_status:
            payload = {"step": "analyst", "message": message}
            if detail:
                payload["detail"] = True
            self.on_status(payload)

    def _get_rag_context(self, review_type: str, exercise_flags: dict) -> str:
        """Retrieve relevant training literature based on the review type."""
        if review_type == "deload":
            query = "How to program deload weeks — volume and intensity recommendations for strength training"
            instructions = "Provide concise guidance on deload protocols: how much to reduce volume and intensity."
        else:
            stalled = [name for name, m in exercise_flags.items() if m.get("flag") == "stalled"]
            stalled_str = ", ".join(stalled[:5]) if stalled else "general exercises"
            query = (
                f"When should you swap exercises in a strength program and what are good "
                f"variations for {stalled_str}? How to structure mesocycle transitions."
            )
            instructions = (
                "Provide concise guidance on exercise rotation and mesocycle periodization. "
                "Suggest specific exercise variations when appropriate."
            )

        try:
            self._emit("Retrieving training literature for analysis...")
            result, _ = self.retrieval_fn(query, instructions)
            return result
        except Exception:
            logger.warning("RAG retrieval failed for analyst, continuing without context", exc_info=True)
            return ""

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Produce a decision document from analytics and training history.

        Parameters
        ----------
        state:
            Must contain: analytics, current_mesocycle_history, user-input.
            Optional: previous_block_summaries.

        Returns
        -------
        dict
            The state dict with 'analyst_decision' populated.
        """
        analytics = state.get("analytics", {})
        review_type = analytics.get("review_type", "normal")

        if review_type == "normal":
            logger.info("Analyst skipped — normal progression week")
            return state

        self._emit(f"Analyzing training data for {review_type.replace('_', ' ')}...")

        exercise_flags = analytics.get("exercise_flags", {})
        rag_context = self._get_rag_context(review_type, exercise_flags)

        analytics_str = json.dumps(analytics, indent=2)
        history_str = json.dumps(state.get("current_mesocycle_history", []), indent=2)
        user_input = state.get("user-input", "")

        if review_type == "deload":
            task = TASK_DELOAD.format(
                analytics_str, history_str, user_input, ANALYST_DELOAD_STRUCTURE
            )
        else:
            summaries_str = json.dumps(state.get("previous_block_summaries", []), indent=2)
            task = TASK_MESOCYCLE_REVIEW.format(
                analytics_str, history_str, summaries_str, user_input,
                ANALYST_DECISION_STRUCTURE
            )

        if rag_context:
            task += f"\n\nRelevant context from training literature:\n{rag_context}\n"

        prompt = f"{self.role['content']}\n\n{task}"

        self._emit(f"Generating {review_type.replace('_', ' ')} recommendations...")
        try:
            response = self.model(prompt)
            if isinstance(response, str):
                try:
                    decision = json.loads(response)
                except json.JSONDecodeError:
                    decision = {"review_type": review_type, "reasoning": response, "recommendations": []}
            elif isinstance(response, dict):
                decision = response
            else:
                decision = {"review_type": review_type, "reasoning": str(response), "recommendations": []}
        except Exception as e:
            logger.exception("Analyst LLM call failed")
            decision = {
                "review_type": review_type,
                "reasoning": f"Analysis failed: {e}",
                "recommendations": [],
            }

        state["analyst_decision"] = decision
        self._emit(f"Analysis complete: {decision.get('reasoning', '')[:200]}", detail=True)
        logger.info("Analyst decision: %s", json.dumps(decision, indent=2)[:500])
        return state

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(state)
