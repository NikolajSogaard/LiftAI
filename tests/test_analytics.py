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


from agent_system.analytics import compute_global_metrics, decide_review_type


class TestComputeGlobalMetrics:
    def test_rising_rpe_trend(self):
        exercise_metrics = {
            "Bench Press": {"stagnation_weeks": 0, "rpe_trend": "rising", "flag": "progressing"},
            "Squat": {"stagnation_weeks": 0, "rpe_trend": "rising", "flag": "progressing"},
        }
        weeks = [
            _make_week(1, {"Day 1": [
                {"name": "Bench Press", "sets_data": [{"weight": "70", "reps": "8", "actual_rpe": "6"}]},
                {"name": "Squat", "sets_data": [{"weight": "100", "reps": "5", "actual_rpe": "6"}]},
            ]}),
            _make_week(2, {"Day 1": [
                {"name": "Bench Press", "sets_data": [{"weight": "72.5", "reps": "8", "actual_rpe": "8"}]},
                {"name": "Squat", "sets_data": [{"weight": "102.5", "reps": "5", "actual_rpe": "9"}]},
            ]}),
        ]
        gm = compute_global_metrics(exercise_metrics, weeks)
        assert gm["avg_rpe_trend"] == "rising"
        assert 0.0 <= gm["fatigue_score"] <= 1.0

    def test_no_stalls_low_fatigue(self):
        exercise_metrics = {
            "Bench Press": {"stagnation_weeks": 0, "rpe_trend": "stable", "flag": "progressing"},
        }
        weeks = [
            _make_week(1, {"Day 1": [
                {"name": "Bench Press", "sets_data": [{"weight": "70", "reps": "8", "actual_rpe": "7"}]},
            ]}),
        ]
        gm = compute_global_metrics(exercise_metrics, weeks)
        assert gm["fatigue_score"] < 0.7
        assert gm["stalled_exercise_ratio"] == 0.0


class TestDecideReviewType:
    def test_normal_week(self):
        global_metrics = {
            "avg_rpe_trend": "stable",
            "fatigue_score": 0.2,
            "stalled_exercise_ratio": 0.0,
            "mesocycle_position": 0.5,
        }
        result = decide_review_type(global_metrics, week_in_mesocycle=2, mesocycle_length=4)
        assert result["review_type"] == "normal"
        assert result["triggers"] == []

    def test_end_of_mesocycle(self):
        global_metrics = {
            "avg_rpe_trend": "stable",
            "fatigue_score": 0.3,
            "stalled_exercise_ratio": 0.1,
            "mesocycle_position": 1.0,
        }
        result = decide_review_type(global_metrics, week_in_mesocycle=4, mesocycle_length=4)
        assert result["review_type"] == "mesocycle_review"
        assert any("end of mesocycle" in t.lower() for t in result["triggers"])

    def test_high_fatigue_triggers_deload(self):
        global_metrics = {
            "avg_rpe_trend": "rising",
            "fatigue_score": 0.8,
            "stalled_exercise_ratio": 0.2,
            "mesocycle_position": 0.5,
        }
        result = decide_review_type(global_metrics, week_in_mesocycle=2, mesocycle_length=4)
        assert result["review_type"] == "deload"

    def test_high_stall_ratio_triggers_review(self):
        global_metrics = {
            "avg_rpe_trend": "stable",
            "fatigue_score": 0.3,
            "stalled_exercise_ratio": 0.6,
            "mesocycle_position": 0.75,
        }
        result = decide_review_type(global_metrics, week_in_mesocycle=3, mesocycle_length=4)
        assert result["review_type"] == "mesocycle_review"

    def test_deload_takes_priority_over_review(self):
        """When both fatigue and stall thresholds are exceeded, deload wins."""
        global_metrics = {
            "avg_rpe_trend": "rising",
            "fatigue_score": 0.85,
            "stalled_exercise_ratio": 0.6,
            "mesocycle_position": 1.0,
        }
        result = decide_review_type(global_metrics, week_in_mesocycle=4, mesocycle_length=4)
        assert result["review_type"] == "deload"
