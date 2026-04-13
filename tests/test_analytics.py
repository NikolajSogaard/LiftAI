"""Tests for agent_system.analytics — pure-Python training metrics."""
import pytest
from agent_system.analytics import compute_exercise_metrics


def _make_week(week, exercises_by_day):
    """Helper: build a weekly record with feedback data."""
    feedback = {}
    program = {}
    for day, exercises in exercises_by_day.items():
        feedback[day] = []
        program[day] = []
        for ex in exercises:
            program[day].append({
                "name": ex["name"],
                "sets": len(ex["sets_data"]),
                "reps": ex.get("reps", "8-12"),
                "target_rpe": ex.get("target_rpe", "7-8"),
            })
            feedback[day].append({
                "name": ex["name"],
                "sets_data": ex["sets_data"],
                "overall_feedback": "",
            })
    return {"week": week, "program": program, "feedback": feedback}


class TestComputeExerciseMetrics:
    def test_progressing_exercise(self):
        """Weight increases each week → stagnation_weeks=0, flag='progressing'."""
        weeks = [
            _make_week(1, {"Day 1": [{"name": "Bench Press", "sets_data": [
                {"weight": "70", "reps": "8", "actual_rpe": "7"}
            ]}]}),
            _make_week(2, {"Day 1": [{"name": "Bench Press", "sets_data": [
                {"weight": "72.5", "reps": "8", "actual_rpe": "7"}
            ]}]}),
            _make_week(3, {"Day 1": [{"name": "Bench Press", "sets_data": [
                {"weight": "75", "reps": "8", "actual_rpe": "7"}
            ]}]}),
        ]
        metrics = compute_exercise_metrics(weeks)
        bench = metrics["Bench Press"]
        assert bench["stagnation_weeks"] == 0
        assert bench["flag"] == "progressing"

    def test_stalled_exercise(self):
        """Same weight and reps for 3 weeks → stagnation_weeks=2, flag='stalled'."""
        weeks = [
            _make_week(1, {"Day 1": [{"name": "RDL", "sets_data": [
                {"weight": "80", "reps": "10", "actual_rpe": "8"}
            ]}]}),
            _make_week(2, {"Day 1": [{"name": "RDL", "sets_data": [
                {"weight": "80", "reps": "10", "actual_rpe": "8"}
            ]}]}),
            _make_week(3, {"Day 1": [{"name": "RDL", "sets_data": [
                {"weight": "80", "reps": "10", "actual_rpe": "9"}
            ]}]}),
        ]
        metrics = compute_exercise_metrics(weeks)
        rdl = metrics["RDL"]
        assert rdl["stagnation_weeks"] == 2
        assert rdl["flag"] == "stalled"

    def test_rep_increase_counts_as_progress(self):
        """Same weight but more reps → not stalled."""
        weeks = [
            _make_week(1, {"Day 1": [{"name": "Squat", "sets_data": [
                {"weight": "100", "reps": "5", "actual_rpe": "8"}
            ]}]}),
            _make_week(2, {"Day 1": [{"name": "Squat", "sets_data": [
                {"weight": "100", "reps": "6", "actual_rpe": "8"}
            ]}]}),
        ]
        metrics = compute_exercise_metrics(weeks)
        assert metrics["Squat"]["stagnation_weeks"] == 0
        assert metrics["Squat"]["flag"] == "progressing"

    def test_missing_feedback_skipped(self):
        """Weeks without feedback data are gracefully skipped."""
        weeks = [
            _make_week(1, {"Day 1": [{"name": "Bench Press", "sets_data": [
                {"weight": "70", "reps": "8", "actual_rpe": "7"}
            ]}]}),
            {"week": 2, "program": {}, "feedback": {}},  # no data
        ]
        metrics = compute_exercise_metrics(weeks)
        assert metrics["Bench Press"]["stagnation_weeks"] == 0
