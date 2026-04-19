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

    Returns {exercise_name: [{week, max_weight, max_reps, avg_rir}, ...]}
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

                weights, reps, rirs = [], [], []
                for s in sets:
                    try:
                        w = float(s.get("weight", 0) or 0)
                        r = float(s.get("reps", 0) or 0)
                        weights.append(w)
                        reps.append(r)
                    except (ValueError, TypeError):
                        pass
                    try:
                        rir = float(s.get("actual_rir", 0) or 0)
                        if rir > 0:
                            rirs.append(rir)
                    except (ValueError, TypeError):
                        pass

                if not weights:
                    continue

                history.setdefault(name, []).append({
                    "week": record.get("week", 0),
                    "max_weight": max(weights),
                    "max_reps": max(reps),
                    "avg_rir": sum(rirs) / len(rirs) if rirs else 0.0,
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


def _compute_rir_trend(entries: list[dict]) -> str:
    """Determine RIR direction: 'rising' (easier), 'falling' (harder = fatigue signal), or 'stable'."""
    rirs = [e["avg_rir"] for e in entries if e["avg_rir"] > 0]
    if len(rirs) < 2:
        return "stable"
    first_half = sum(rirs[: len(rirs) // 2]) / (len(rirs) // 2)
    second_half = sum(rirs[len(rirs) // 2 :]) / (len(rirs) - len(rirs) // 2)
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
        {exercise_name: {stagnation_weeks, rir_trend, load_progression, flag}}
    """
    history = _extract_exercise_history(weeks)
    metrics: dict[str, dict[str, Any]] = {}

    for name, entries in history.items():
        entries.sort(key=lambda e: e["week"])
        stagnation = _compute_stagnation(entries)
        rir_trend = _compute_rir_trend(entries)
        load_start = entries[0]["max_weight"]
        load_end = entries[-1]["max_weight"]

        flag = "stalled" if stagnation >= STAGNATION_THRESHOLD_WEEKS else "progressing"

        metrics[name] = {
            "stagnation_weeks": stagnation,
            "rir_trend": rir_trend,
            "load_progression": load_end - load_start,
            "flag": flag,
        }

    return metrics


def compute_global_metrics(
    exercise_metrics: dict[str, dict[str, Any]],
    weeks: list[dict],
) -> dict[str, Any]:
    """Compute global training metrics across all exercises.

    Parameters
    ----------
    exercise_metrics:
        Output of compute_exercise_metrics().
    weeks:
        The weekly records (used for per-week average RIR calculation).

    Returns
    -------
    dict
        {avg_rir_trend, fatigue_score, stalled_exercise_ratio}
    """
    # Per-week average RIR across all exercises
    weekly_rirs: list[float] = []
    for record in weeks:
        feedback = record.get("feedback") or {}
        all_rirs: list[float] = []
        for day_exercises in feedback.values():
            for ex in day_exercises:
                for s in ex.get("sets_data", []):
                    try:
                        rir = float(s.get("actual_rir", 0) or 0)
                        if rir > 0:
                            all_rirs.append(rir)
                    except (ValueError, TypeError):
                        pass
        if all_rirs:
            weekly_rirs.append(sum(all_rirs) / len(all_rirs))

    # Average RIR trend (falling RIR = fatigue signal)
    rir_diff = 0.0
    if len(weekly_rirs) >= 2:
        first = sum(weekly_rirs[: len(weekly_rirs) // 2]) / (len(weekly_rirs) // 2)
        second = sum(weekly_rirs[len(weekly_rirs) // 2 :]) / (len(weekly_rirs) - len(weekly_rirs) // 2)
        rir_diff = second - first
        if rir_diff > 0.5:
            avg_rir_trend = "rising"
        elif rir_diff < -0.5:
            avg_rir_trend = "falling"
        else:
            avg_rir_trend = "stable"
    else:
        avg_rir_trend = "stable"

    # Stalled exercise ratio
    total = len(exercise_metrics)
    stalled = sum(1 for m in exercise_metrics.values() if m["flag"] == "stalled")
    stalled_ratio = stalled / total if total > 0 else 0.0

    # RIR fall component (0..1): how much average RIR fell, capped at 2.0 RIR drop
    rir_fall = 0.0
    if len(weekly_rirs) >= 2:
        rir_fall = min(max(-rir_diff, 0.0) / 2.0, 1.0)

    # Rep decline component: approximate with stalled_ratio
    rep_decline = stalled_ratio

    # Fatigue score composite
    fatigue = (0.4 * rir_fall) + (0.3 * rep_decline) + (0.3 * stalled_ratio)
    fatigue = min(fatigue, 1.0)

    return {
        "avg_rir_trend": avg_rir_trend,
        "fatigue_score": round(fatigue, 3),
        "stalled_exercise_ratio": round(stalled_ratio, 3),
    }


def decide_review_type(
    global_metrics: dict[str, Any],
    week_in_mesocycle: int,
    mesocycle_length: int,
) -> dict[str, Any]:
    """Decide whether this week is normal, deload, or mesocycle review.

    Parameters
    ----------
    global_metrics:
        Output of compute_global_metrics().
    week_in_mesocycle:
        Current position in the mesocycle (1-indexed).
    mesocycle_length:
        Total weeks in the mesocycle.

    Returns
    -------
    dict
        {review_type: str, triggers: list[str]}
    """
    from config import FATIGUE_SCORE_DELOAD_TRIGGER, STALL_RATIO_REVIEW_TRIGGER

    triggers: list[str] = []
    fatigue = global_metrics["fatigue_score"]
    stall_ratio = global_metrics["stalled_exercise_ratio"]

    # Deload takes priority — acute fatigue needs recovery first
    if fatigue > FATIGUE_SCORE_DELOAD_TRIGGER:
        triggers.append(f"Fatigue score {fatigue:.2f} exceeds threshold {FATIGUE_SCORE_DELOAD_TRIGGER}")
        return {"review_type": "deload", "triggers": triggers}

    # Mesocycle review triggers
    if week_in_mesocycle >= mesocycle_length:
        triggers.append(f"End of mesocycle (week {week_in_mesocycle}/{mesocycle_length})")

    if stall_ratio > STALL_RATIO_REVIEW_TRIGGER:
        triggers.append(f"Stalled exercise ratio {stall_ratio:.2f} exceeds threshold {STALL_RATIO_REVIEW_TRIGGER}")

    if triggers:
        return {"review_type": "mesocycle_review", "triggers": triggers}

    return {"review_type": "normal", "triggers": []}


def analyze_training_history(
    weeks: list[dict],
    week_in_mesocycle: int,
    mesocycle_length: int,
) -> dict[str, Any]:
    """Top-level entry point: compute all metrics and decide review type.

    Parameters
    ----------
    weeks:
        Current mesocycle's weekly records with feedback.
    week_in_mesocycle:
        Current position in the mesocycle (1-indexed).
    mesocycle_length:
        Configured mesocycle length.

    Returns
    -------
    dict
        {review_type, triggers, exercise_flags, global_metrics}
    """
    exercise_metrics = compute_exercise_metrics(weeks)
    global_metrics = compute_global_metrics(exercise_metrics, weeks)
    global_metrics["mesocycle_position"] = round(
        week_in_mesocycle / mesocycle_length, 2
    ) if mesocycle_length > 0 else 0.0

    decision = decide_review_type(global_metrics, week_in_mesocycle, mesocycle_length)

    return {
        "review_type": decision["review_type"],
        "triggers": decision["triggers"],
        "exercise_flags": exercise_metrics,
        "global_metrics": global_metrics,
    }
