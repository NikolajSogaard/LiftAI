"""Pure-Python training analytics — no LLM calls.

Computes per-exercise and global metrics from accumulated weekly training
records to determine whether a normal progression, deload, or mesocycle
review is warranted.
"""
from __future__ import annotations

import logging
from typing import Any

from config import STAGNATION_THRESHOLD_WEEKS

logger = logging.getLogger(__name__)


def _extract_exercise_history(weeks: list[dict]) -> dict[str, list[dict]]:
    """Group per-set feedback by exercise name across all weeks.

    Returns {exercise_name: [{week, max_weight, max_reps, avg_rpe}, ...]}
    """
    history: dict[str, list[dict]] = {}
    for record in weeks:
        feedback = record.get("feedback") or {}
        for day, exercises in feedback.items():
            for ex in exercises:
                name = ex.get("name")
                if not name:
                    continue
                sets = ex.get("sets_data", [])
                if not sets:
                    continue

                weights, reps, rpes = [], [], []
                for s in sets:
                    try:
                        w = float(s.get("weight", 0) or 0)
                        r = float(s.get("reps", 0) or 0)
                        weights.append(w)
                        reps.append(r)
                    except (ValueError, TypeError):
                        pass
                    try:
                        rpe = float(s.get("actual_rpe", 0) or 0)
                        if rpe > 0:
                            rpes.append(rpe)
                    except (ValueError, TypeError):
                        pass

                if not weights:
                    continue

                history.setdefault(name, []).append({
                    "week": record.get("week", 0),
                    "max_weight": max(weights),
                    "max_reps": max(reps),
                    "avg_rpe": sum(rpes) / len(rpes) if rpes else 0.0,
                })
    return history


def _compute_stagnation(entries: list[dict]) -> int:
    """Count consecutive weeks (from the end) with no weight or rep increase."""
    if len(entries) < 2:
        return 0
    stagnation = 0
    for i in range(len(entries) - 1, 0, -1):
        curr = entries[i]
        prev = entries[i - 1]
        if curr["max_weight"] > prev["max_weight"] or curr["max_reps"] > prev["max_reps"]:
            break
        stagnation += 1
    return stagnation


def _compute_rpe_trend(entries: list[dict]) -> str:
    """Determine RPE direction: 'rising', 'falling', or 'stable'."""
    rpes = [e["avg_rpe"] for e in entries if e["avg_rpe"] > 0]
    if len(rpes) < 2:
        return "stable"
    first_half = sum(rpes[: len(rpes) // 2]) / (len(rpes) // 2)
    second_half = sum(rpes[len(rpes) // 2 :]) / (len(rpes) - len(rpes) // 2)
    diff = second_half - first_half
    if diff > 0.5:
        return "rising"
    if diff < -0.5:
        return "falling"
    return "stable"


def compute_exercise_metrics(weeks: list[dict]) -> dict[str, dict[str, Any]]:
    """Compute per-exercise metrics from weekly training records.

    Parameters
    ----------
    weeks:
        List of enriched weekly records, each with 'feedback' containing
        per-exercise sets_data.

    Returns
    -------
    dict
        {exercise_name: {stagnation_weeks, rpe_trend, load_progression, flag}}
    """
    history = _extract_exercise_history(weeks)
    metrics: dict[str, dict[str, Any]] = {}

    for name, entries in history.items():
        entries.sort(key=lambda e: e["week"])
        stagnation = _compute_stagnation(entries)
        rpe_trend = _compute_rpe_trend(entries)
        load_start = entries[0]["max_weight"]
        load_end = entries[-1]["max_weight"]

        flag = "stalled" if stagnation >= STAGNATION_THRESHOLD_WEEKS else "progressing"

        metrics[name] = {
            "stagnation_weeks": stagnation,
            "rpe_trend": rpe_trend,
            "load_progression": load_end - load_start,
            "flag": flag,
        }

    return metrics
