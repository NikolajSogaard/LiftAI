# Autoregulation & Mesocycle Progression — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add intelligent autoregulation to LiftAI — detect stagnation/fatigue from performance history, trigger program reviews at mesocycle boundaries, and let an Analyst agent recommend exercise swaps, volume changes, and deloads with user approval.

**Architecture:** Python analytics module computes deterministic metrics (stagnation, fatigue, RPE trends). An Analyst LLM agent interprets those metrics into a decision document. The existing Writer → Critic → Editor pipeline executes the approved changes. Conditional LangGraph routing selects the path based on review type (normal/deload/mesocycle_review).

**Tech Stack:** Python, LangGraph, Google Gemini API (google-genai SDK), Flask, SSE, FAISS RAG

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `agent_system/analytics.py` | Pure-Python metrics: per-exercise stagnation, RPE trends, fatigue score, review-type decision |
| `agent_system/agents/analyst.py` | LLM agent that interprets analytics into a decision document (swaps, deload, volume) |
| `prompts/analyst_prompts.py` | Analyst role, mesocycle review task template, deload task template |
| `tests/test_analytics.py` | Unit tests for the analytics module |

### Modified Files

| File | What changes |
|------|-------------|
| `config.py` | Add 5 mesocycle/autoregulation constants |
| `prompts/writer_prompts.py` | Add `TASK_DELOAD_WRITER`, `TASK_NEW_BLOCK`, `DELOAD_WRITER_ROLE`, `NEW_BLOCK_ROLE` + register in `WRITER_PROMPT_SETTINGS` |
| `prompts/__init__.py` | Re-export new analyst prompt symbols |
| `agent_system/agents/__init__.py` | Re-export `Analyst` |
| `agent_system/__init__.py` | Re-export `Analyst` |
| `agent_system/agents/writer.py` | Add `deload` and `new_block` writer types in `__call__`, `write_deload()`, `write_new_block()` methods |
| `agent_system/generator.py` | New `ProgressionProgramGenerator` class with Analytics → Analyst → Approval Gate → Writer → Critic → Editor workflow |
| `app.py` | New session fields, updated `/next_week` to use `ProgressionProgramGenerator`, new `/approve_review` endpoint, block summary generation |
| `templates/index.html` | Review UI overlay with analyst reasoning + recommendations + Approve/Edit/Skip |

---

### Task 1: Config Constants

**Files:**
- Modify: `config.py:23` (after LESSON_MAX_CHARS)

- [ ] **Step 1: Add mesocycle/autoregulation constants to config.py**

Open `config.py` and add after the `# ── Reflexion` section (after line 23):

```python
# ── Mesocycle & autoregulation ──────────────────────────────────────────────
DEFAULT_MESOCYCLE_LENGTH = 4         # weeks per training block
STAGNATION_THRESHOLD_WEEKS = 2      # consecutive weeks with no progress → flagged
FATIGUE_SCORE_DELOAD_TRIGGER = 0.7  # fatigue above this → deload
STALL_RATIO_REVIEW_TRIGGER = 0.5    # fraction of exercises stalled → mesocycle review
DELOAD_VOLUME_REDUCTION = 0.5       # reduce sets by ~50% during deload
```

- [ ] **Step 2: Commit**

```bash
git add config.py
git commit -m "feat: add mesocycle and autoregulation config constants"
```

---

### Task 2: Analytics Module — Core Metrics

**Files:**
- Create: `agent_system/analytics.py`
- Create: `tests/test_analytics.py`

- [ ] **Step 1: Create tests/test_analytics.py with test for per-exercise stagnation**

Create `tests/__init__.py` (empty) if it doesn't exist, then create `tests/test_analytics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analytics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_system.analytics'`

- [ ] **Step 3: Create agent_system/analytics.py with compute_exercise_metrics**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analytics.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_system/analytics.py tests/__init__.py tests/test_analytics.py
git commit -m "feat: add analytics module with per-exercise stagnation and RPE metrics"
```

---

### Task 3: Analytics Module — Global Metrics and Review Decision

**Files:**
- Modify: `agent_system/analytics.py`
- Modify: `tests/test_analytics.py`

- [ ] **Step 1: Add tests for global metrics and review decision**

Append to `tests/test_analytics.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_analytics.py -v`
Expected: New tests FAIL — `ImportError: cannot import name 'compute_global_metrics'`

- [ ] **Step 3: Add compute_global_metrics and decide_review_type to analytics.py**

Append to `agent_system/analytics.py`:

```python
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
        The weekly records (used for per-week average RPE calculation).

    Returns
    -------
    dict
        {avg_rpe_trend, fatigue_score, stalled_exercise_ratio, mesocycle_position}
    """
    # Per-week average RPE across all exercises
    weekly_rpes: list[float] = []
    for record in weeks:
        feedback = record.get("feedback") or {}
        all_rpes: list[float] = []
        for day_exercises in feedback.values():
            for ex in day_exercises:
                for s in ex.get("sets_data", []):
                    try:
                        rpe = float(s.get("actual_rpe", 0) or 0)
                        if rpe > 0:
                            all_rpes.append(rpe)
                    except (ValueError, TypeError):
                        pass
        if all_rpes:
            weekly_rpes.append(sum(all_rpes) / len(all_rpes))

    # Average RPE trend
    if len(weekly_rpes) >= 2:
        first = sum(weekly_rpes[: len(weekly_rpes) // 2]) / (len(weekly_rpes) // 2)
        second = sum(weekly_rpes[len(weekly_rpes) // 2 :]) / (len(weekly_rpes) - len(weekly_rpes) // 2)
        rpe_diff = second - first
        if rpe_diff > 0.5:
            avg_rpe_trend = "rising"
        elif rpe_diff < -0.5:
            avg_rpe_trend = "falling"
        else:
            avg_rpe_trend = "stable"
    else:
        avg_rpe_trend = "stable"

    # Stalled exercise ratio
    total = len(exercise_metrics)
    stalled = sum(1 for m in exercise_metrics.values() if m["flag"] == "stalled")
    stalled_ratio = stalled / total if total > 0 else 0.0

    # RPE rise component (0..1): how much average RPE rose, capped at 2.0 RPE increase
    rpe_rise = 0.0
    if len(weekly_rpes) >= 2:
        rpe_rise = min(max(rpe_diff, 0.0) / 2.0, 1.0)

    # Rep decline component (0..1): placeholder based on stall ratio
    # A true rep-decline metric would compare prescribed vs actual reps,
    # but the current data model stores reps as strings ("8-12") making
    # comparison impractical. We approximate with stalled_ratio.
    rep_decline = stalled_ratio

    # Fatigue score composite
    fatigue = (0.4 * rpe_rise) + (0.3 * rep_decline) + (0.3 * stalled_ratio)
    fatigue = min(fatigue, 1.0)

    return {
        "avg_rpe_trend": avg_rpe_trend,
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
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_analytics.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_system/analytics.py tests/test_analytics.py
git commit -m "feat: add global metrics and review-type decision to analytics module"
```

---

### Task 4: Analytics Module — Top-Level analyze_training_history Function

**Files:**
- Modify: `agent_system/analytics.py`
- Modify: `tests/test_analytics.py`

- [ ] **Step 1: Add integration test for analyze_training_history**

Append to `tests/test_analytics.py`:

```python
from agent_system.analytics import analyze_training_history


class TestAnalyzeTrainingHistory:
    def test_full_pipeline_normal(self):
        """3 weeks of progressing data, mid-mesocycle → normal."""
        weeks = [
            _make_week(w, {"Day 1": [{"name": "Bench Press", "sets_data": [
                {"weight": str(70 + w * 2.5), "reps": "8", "actual_rpe": "7"}
            ]}]})
            for w in range(1, 4)
        ]
        result = analyze_training_history(weeks, week_in_mesocycle=3, mesocycle_length=4)
        assert result["review_type"] == "normal"
        assert "Bench Press" in result["exercise_flags"]
        assert result["exercise_flags"]["Bench Press"]["flag"] == "progressing"
        assert "global_metrics" in result

    def test_full_pipeline_end_of_mesocycle(self):
        """4 weeks, at end of mesocycle → mesocycle_review."""
        weeks = [
            _make_week(w, {"Day 1": [{"name": "Bench Press", "sets_data": [
                {"weight": str(70 + w * 2.5), "reps": "8", "actual_rpe": "7"}
            ]}]})
            for w in range(1, 5)
        ]
        result = analyze_training_history(weeks, week_in_mesocycle=4, mesocycle_length=4)
        assert result["review_type"] == "mesocycle_review"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analytics.py::TestAnalyzeTrainingHistory -v`
Expected: FAIL — `ImportError: cannot import name 'analyze_training_history'`

- [ ] **Step 3: Add analyze_training_history to analytics.py**

Append to `agent_system/analytics.py`:

```python
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
```

- [ ] **Step 4: Run all analytics tests**

Run: `python -m pytest tests/test_analytics.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_system/analytics.py tests/test_analytics.py
git commit -m "feat: add analyze_training_history top-level entry point"
```

---

### Task 5: Analyst Prompts

**Files:**
- Create: `prompts/analyst_prompts.py`
- Modify: `prompts/__init__.py`

- [ ] **Step 1: Create prompts/analyst_prompts.py**

```python
"""Prompt templates for the Analyst agent — mesocycle reviews and deloads."""

ANALYST_ROLE = {
    'role': 'system',
    'content': (
        'You are a strength training analyst specializing in program periodization and autoregulation. '
        'You receive computed performance metrics and training history, and produce specific, '
        'evidence-based recommendations for program adjustments. '
        'Your recommendations must be concrete — name exact exercises, exact volume changes, '
        'and explain your reasoning based on the data provided. '
        'You preserve the overall split structure unless the data strongly warrants a change.'
    )
}

ANALYST_DECISION_STRUCTURE = '''{
    "review_type": "mesocycle_review",
    "reasoning": "Free-text explanation of what patterns the data shows and why changes are needed.",
    "recommendations": [
        {
            "type": "swap",
            "exercise": "Name of exercise to replace",
            "replacement": "Name of replacement exercise",
            "reason": "Why this swap — reference the stagnation data"
        },
        {
            "type": "adjust_volume",
            "muscle_group": "Muscle group name",
            "change": "+2 sets/week or -3 sets/week",
            "reason": "Why this volume change"
        }
    ],
    "deload": null,
    "next_mesocycle_length": 4
}'''

ANALYST_DELOAD_STRUCTURE = '''{
    "review_type": "deload",
    "reasoning": "Free-text explanation of fatigue signals observed.",
    "recommendations": [],
    "deload": {
        "volume_reduction": 0.5,
        "intensity": "moderate",
        "duration_weeks": 1
    },
    "next_mesocycle_length": null
}'''

TASK_MESOCYCLE_REVIEW = '''Analyze this training block and recommend changes for the next mesocycle.

Performance analytics:
{}

Current mesocycle history:
{}

Previous block summaries:
{}

User profile:
{}

Based on the analytics data, provide:
1. Your reasoning — what patterns do you see in the data?
2. Exercise swap recommendations — only for stalled or problematic exercises.
   For each swap, name the replacement and explain why it is a good variation.
3. Volume adjustments — any muscle groups that need more or less work.
4. Recommended mesocycle length for the next block.

Rules:
- Keep the same split structure (e.g., Upper/Lower stays Upper/Lower)
- Only swap exercises that have clear stagnation signals (2+ weeks no progress)
- Main compound lifts should only get variations, not completely different movements
- Accessories can be swapped more freely
- Do not swap exercises that are still progressing

Respond in JSON following this structure:
{}
'''

TASK_DELOAD = '''Design a deload week based on the following fatigue indicators.

Performance analytics:
{}

Current mesocycle history:
{}

User profile:
{}

Provide:
1. Your reasoning — what fatigue signals are you seeing?
2. Volume reduction plan — which exercises reduce sets and by how much
3. Intensity guidance — keep weight the same but reduce target RPE by how much

Rules:
- Reduce total weekly sets by approximately 40-50%
- Maintain exercise selection — do not swap exercises during a deload
- Keep weights the same or slightly lower, reduce RPE targets by 1-2
- Prioritize reducing volume on exercises showing the highest fatigue signals

Respond in JSON following this structure:
{}
'''
```

- [ ] **Step 2: Update prompts/__init__.py to export analyst symbols**

Replace the contents of `prompts/__init__.py` with:

```python
from .writer_prompts import WriterPromptSettings, WRITER_PROMPT_SETTINGS
from .critic_prompts import CriticPromptSettings, CRITIC_PROMPT_SETTINGS
from .analyst_prompts import ANALYST_ROLE, TASK_MESOCYCLE_REVIEW, TASK_DELOAD
```

- [ ] **Step 3: Commit**

```bash
git add prompts/analyst_prompts.py prompts/__init__.py
git commit -m "feat: add analyst prompt templates for mesocycle review and deload"
```

---

### Task 6: Analyst Agent

**Files:**
- Create: `agent_system/agents/analyst.py`
- Modify: `agent_system/agents/__init__.py`
- Modify: `agent_system/__init__.py`

- [ ] **Step 1: Create agent_system/agents/analyst.py**

```python
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
from agent_system.utils import parse_json_draft
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
            Must contain: analytics, current_mesocycle_history, user_input.
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
```

- [ ] **Step 2: Update agent_system/agents/__init__.py**

Replace contents with:

```python
from .writer import Writer
from .critic import Critic
from .editor import Editor
from .analyst import Analyst
```

- [ ] **Step 3: Update agent_system/__init__.py**

Replace contents with:

```python
from .generator import ProgramGenerator
from .setup_api import setup_llm
from .chatbot import ProgramChatbot
from .agents import (
    Writer,
    Critic,
    Editor,
    Analyst,
)
```

- [ ] **Step 4: Verify import works**

Run: `python -c "from agent_system import Analyst; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add agent_system/agents/analyst.py agent_system/agents/__init__.py agent_system/__init__.py
git commit -m "feat: add Analyst agent for mesocycle review and deload decisions"
```

---

### Task 7: New Writer Prompts and Writer Types

**Files:**
- Modify: `prompts/writer_prompts.py`
- Modify: `agent_system/agents/writer.py`

- [ ] **Step 1: Add deload and new-block prompts to writer_prompts.py**

Add after the existing `TASK_PROGRESSION` definition (after line 100) in `prompts/writer_prompts.py`:

```python
TASK_DELOAD_WRITER = '''Generate a deload week program based on:

1) The previous week's program:
{}

2) The analyst's deload recommendations:
{}

IMPORTANT:
- Keep all exercises identical to the previous week
- Reduce sets per the analyst's volume reduction plan (approximately 40-50% fewer sets)
- Maintain weights but lower target RPE by 1-2 points
- This is a recovery week — the goal is reduced fatigue, not progression
- Do NOT add any new exercises or modify exercise order

Follow this JSON structure as a guide for your response:
{}
'''

TASK_NEW_BLOCK = '''Generate Week 1 of a new training block based on:

1) The previous block's final program:
{}

2) The analyst's recommendations for the new block:
{}

3) Previous block summaries:
{}

IMPORTANT:
- Implement all approved exercise swaps from the analyst recommendations
- Apply volume adjustments as recommended
- Maintain the same training split structure
- Set initial weights conservatively — use the last working weights for retained exercises,
  start 10-15% lighter for newly swapped exercises
- Reset RPE targets to moderate levels (6-8 for compounds, 7-9 for isolation)
- Fill in the "cues" field with specific coaching notes for any new exercises

Follow this JSON structure as a guide for your response:
{}
'''

DELOAD_WRITER_ROLE = {
    'role': 'system',
    'content': (
        'You are an AI system specialized in generating deload training weeks. '
        'Your task is to reduce training volume while preserving exercise selection and movement patterns. '
        'Keep the program structure identical to the previous week but reduce sets by the recommended amount. '
        'Provide clear, CONCISE output with no additional commentary.'
    )
}

NEW_BLOCK_ROLE = {
    'role': 'system',
    'content': (
        'You are an AI system specialized in creating the first week of a new training mesocycle. '
        'You implement specific exercise swaps and volume adjustments recommended by an analyst, '
        'while preserving the overall training split structure. '
        'Set conservative initial loads for new exercises and moderate RPE targets across the board. '
        'Provide clear, CONCISE output with no additional commentary.'
    )
}
```

Then add the new settings registrations at the end of the file (before the `# Add the original v1` alias block, which is around line 194):

```python
WRITER_PROMPT_SETTINGS['deload'] = WriterPromptSettings(
    role=DELOAD_WRITER_ROLE,
    task=TASK_DELOAD_WRITER,
    structure=PROGRAM_STRUCTURE_WEEK1,
)

WRITER_PROMPT_SETTINGS['new_block'] = WriterPromptSettings(
    role=NEW_BLOCK_ROLE,
    task=TASK_NEW_BLOCK,
    structure=PROGRAM_STRUCTURE_WEEK1,
)
```

- [ ] **Step 2: Add write_deload and write_new_block methods to Writer**

In `agent_system/agents/writer.py`, add two new methods after the existing `revise` method (before the `# --- Helper methods` comment around line 222). Also update the `__call__` method and `specialized_instructions`:

First, add to `specialized_instructions` dict in `__init__` (around line 33):

```python
            "deload": "Focus on deload week programming. Retrieve guidance on how to reduce training volume while maintaining movement patterns for recovery.",
            "new_block": "Focus on mesocycle transitions and exercise rotation. Retrieve guidance on how to set up the first week of a new training block with fresh exercises.",
```

Then add the two new methods before `# --- Helper methods for revision post-processing ---`:

```python
    def write_deload(self, program: dict) -> dict:
        """Generate a deload week — same exercises, reduced volume.

        Parameters
        ----------
        program:
            LangGraph state dict. Must contain 'draft' (previous week's program)
            and 'analyst_decision' with the deload plan.

        Returns
        -------
        dict
            The deload program draft.
        """
        if not self.task:
            raise ValueError("Writer 'deload' has no task template")

        analyst_decision = program.get("analyst_decision", {})
        previous_program = self.format_previous_week_program(program)

        query = self.get_retrieval_query(program)
        enhanced_task = self.task
        if query:
            self._emit("Retrieving context for deload programming...")
            try:
                result, _ = self.retrieval_fn(query, self.specialized_instructions.get("deload", ""))
                enhanced_task += f"\nRelevant context from training literature:\n{result}\n"
            except Exception:
                logger.warning("RAG retrieval failed for deload writer", exc_info=True)

        prompt = self._build_prompt([
            self.role,
            {'role': 'user', 'content': enhanced_task.format(
                previous_program,
                json.dumps(analyst_decision, indent=2),
                self.structure,
            )},
        ])

        self._emit("Generating deload week program...")
        draft = self.model(prompt)
        if isinstance(draft, str):
            draft = self._parse_string_draft(draft)
        return draft

    def write_new_block(self, program: dict) -> dict:
        """Generate Week 1 of a new mesocycle based on analyst recommendations.

        Parameters
        ----------
        program:
            LangGraph state dict. Must contain 'draft', 'analyst_decision',
            and optionally 'previous_block_summaries'.

        Returns
        -------
        dict
            The new block's Week 1 program draft.
        """
        if not self.task:
            raise ValueError("Writer 'new_block' has no task template")

        analyst_decision = program.get("analyst_decision", {})
        previous_program = self.format_previous_week_program(program)
        block_summaries = json.dumps(program.get("previous_block_summaries", []), indent=2)

        query = self.get_retrieval_query(program)
        enhanced_task = self.task
        if query:
            self._emit("Retrieving context for new training block...")
            try:
                result, _ = self.retrieval_fn(query, self.specialized_instructions.get("new_block", ""))
                enhanced_task += f"\nRelevant context from training literature:\n{result}\n"
            except Exception:
                logger.warning("RAG retrieval failed for new block writer", exc_info=True)

        prompt = self._build_prompt([
            self.role,
            {'role': 'user', 'content': enhanced_task.format(
                previous_program,
                json.dumps(analyst_decision, indent=2),
                block_summaries,
                self.structure,
            )},
        ])

        self._emit("Generating new training block Week 1...")
        draft = self.model(prompt)
        if isinstance(draft, str):
            draft = self._parse_string_draft(draft)
        return draft
```

Finally, update the `__call__` method (around line 326) to handle the new types. Replace the existing `__call__`:

```python
    def __call__(self, program: dict[str, str | None]) -> dict[str, str | None]:
        if self.writer_type == "deload":
            logger.info("Deload writer")
            draft = self.write_deload(program)
        elif self.writer_type == "new_block":
            logger.info("New block writer (mesocycle transition)")
            draft = self.write_new_block(program)
        elif self.writer_type == "progression" and 'feedback' in program:
            logger.info("Progression writer (Week 2+)")
            draft = self.revise(program, override_type="progression")
        elif program.get('draft') is None:
            logger.info("Initial program creation")
            draft = self.write(program)
        elif 'feedback' in program:
            logger.info("Revising based on feedback")
            draft = self.revise(program, override_type="revision")
        else:
            logger.info("Fallback: initial write")
            draft = self.write(program)

        program['draft'] = draft
        return program
```

Also update `get_retrieval_query` to handle new types (around line 46):

```python
    def get_retrieval_query(self, program: dict[str, str | None]) -> str:
        user_input = program.get('user-input', '')
        if self.writer_type == "initial":
            return f"Best practices for designing a strength training program for someone with these goals and preferences: {user_input}"
        if self.writer_type == "progression":
            return f"Progressive overload principles and autoregulation strategies for strength training: {user_input}"
        if self.writer_type == "revision":
            return f"How to revise and improve a strength training program based on feedback: {user_input}"
        if self.writer_type == "deload":
            return f"How to program deload weeks in strength training — volume and intensity reduction protocols"
        if self.writer_type == "new_block":
            return f"How to structure mesocycle transitions, exercise rotation and fresh training blocks for continued progression: {user_input}"
        return f"Strength training program design best practices: {user_input}"
```

- [ ] **Step 3: Verify imports work**

Run: `python -c "from prompts.writer_prompts import WRITER_PROMPT_SETTINGS; print(list(WRITER_PROMPT_SETTINGS.keys()))"`
Expected: Output includes `'deload'` and `'new_block'`

- [ ] **Step 4: Commit**

```bash
git add prompts/writer_prompts.py agent_system/agents/writer.py
git commit -m "feat: add deload and new-block writer types with prompt templates"
```

---

### Task 8: ProgressionProgramGenerator — Updated LangGraph Workflow

**Files:**
- Modify: `agent_system/generator.py`

- [ ] **Step 1: Add ProgressionProgramGenerator to generator.py**

Add the following class after the existing `ProgramGenerator` class in `agent_system/generator.py`. Also add the new imports at the top of the file:

Add to imports (after existing imports at top):

```python
from .analytics import analyze_training_history
from .agents import Analyst
```

Then add the new class after `ProgramGenerator`:

```python
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

    def run_analytics(self, state: dict) -> dict:
        """Run the Python analytics module and store results in state."""
        self._propagate_status()
        if self.on_status:
            self.on_status({"step": "analytics", "message": "Analyzing training history..."})

        weeks = state.get("current_mesocycle_history", [])
        week_in_meso = state.get("week_in_mesocycle", 1)

        analytics = analyze_training_history(
            weeks=weeks,
            week_in_mesocycle=week_in_meso,
            mesocycle_length=self.mesocycle_length,
        )
        state["analytics"] = analytics

        review_type = analytics["review_type"]
        if self.on_status:
            self.on_status({
                "step": "analytics",
                "message": f"Analysis complete — review type: {review_type}",
            })
        logger.info("Analytics result: review_type=%s, triggers=%s", review_type, analytics["triggers"])
        return state

    def run_analyst(self, state: dict) -> dict:
        """Run the Analyst agent to produce a decision document."""
        return self.analyst(state)

    def run_writer(self, state: dict) -> dict:
        """Run the Writer agent."""
        return self.writer(state)

    def run_critic(self, state: dict) -> dict:
        """Run the Critic agent."""
        return self.critic(state)

    def run_editor(self, state: dict) -> dict:
        """Run the Editor agent."""
        return self.editor(state)

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
        state = self.run_analytics(state)
        review_type = state["analytics"]["review_type"]

        # Inject exercise flags into state for the writer
        state["exercise_flags"] = state["analytics"].get("exercise_flags", {})

        if review_type == "normal":
            # Normal progression — straight to Writer → Critic → Editor
            state = self.run_writer(state)
            state = self.run_critic(state)
            state = self.run_editor(state)
            return state

        # Phase 2: Analyst (for deload or mesocycle_review)
        state = self.run_analyst(state)
        state["needs_approval"] = True
        return state

    def continue_after_approval(self, state: dict) -> dict:
        """Resume the workflow after user approves the analyst decision.

        Call this after the user reviews and approves the analyst_decision.
        Sets up the appropriate Writer type and runs Writer → [Critic] → Editor.

        Parameters
        ----------
        state:
            The state dict returned by create_program() with needs_approval=True.
            The caller should set state['user_approved'] = True.

        Returns
        -------
        dict
            Final state with 'formatted' containing the program.
        """
        self._propagate_status()
        state.pop("needs_approval", None)

        review_type = state["analytics"]["review_type"]

        # Writer phase
        state = self.run_writer(state)

        # Critic phase — skip for deloads
        if review_type != "deload":
            state = self.run_critic(state)

        # Editor phase
        state = self.run_editor(state)
        return state
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from agent_system.generator import ProgressionProgramGenerator; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent_system/generator.py
git commit -m "feat: add ProgressionProgramGenerator with analytics-driven conditional workflow"
```

---

### Task 9: App.py — Session Fields, Block Summary, and Updated /next_week

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add new config imports and session initialization**

In `app.py`, add to the config imports (around line 37):

```python
from config import (
    DEFAULT_MODEL,
    DEFAULT_CRITIC_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_WRITER_TEMPERATURE,
    DEFAULT_WRITER_TOP_P,
    DEFAULT_MAX_ITERATIONS,
    MAX_USER_INPUT_CHARS,
    QUEUE_TIMEOUT,
    DEFAULT_MESOCYCLE_LENGTH,
)
```

Add `Analyst` to the agent_system import (around line 19):

```python
from agent_system import (
    setup_llm,
    ProgramGenerator,
    ProgramChatbot,
    Writer,
    Critic,
    Editor,
    Analyst,
)
```

Add the new import for the progression generator:

```python
from agent_system.generator import ProgressionProgramGenerator
```

- [ ] **Step 2: Update generation_complete to initialize mesocycle session fields**

In the `generation_complete` function (around line 346), update the session initialization. Replace the existing session assignments with:

```python
    session['program'] = result['program']
    session['raw_program'] = result['raw_program']
    session['user_input'] = result['user_input']
    session['persona'] = result['persona']
    session['feedback'] = {}
    session['current_week'] = 1
    session['mesocycle'] = 1
    session['week_in_mesocycle'] = 1
    session['mesocycle_length'] = DEFAULT_MESOCYCLE_LENGTH
    session['block_summaries'] = []
    session['pending_review'] = None
    session['all_programs'] = [{'week': 1, 'mesocycle': 1, 'week_in_mesocycle': 1, 'type': 'normal', 'program': result['program']}]
```

- [ ] **Step 3: Add helper function for building block summaries**

Add this function before the routes (e.g., after `get_program_generator`):

```python
def _build_block_summary(all_programs: list[dict], mesocycle: int) -> dict:
    """Build a condensed summary of a completed mesocycle.

    Parameters
    ----------
    all_programs:
        The full list of weekly records.
    mesocycle:
        The mesocycle number to summarize.

    Returns
    -------
    dict
        Condensed block summary with key lift trends and exercises used.
    """
    block_weeks = [w for w in all_programs if w.get("mesocycle") == mesocycle]
    if not block_weeks:
        return {}

    # Track per-exercise first/last performance
    exercise_data: dict[str, dict] = {}
    exercises_used: list[str] = []

    for week_record in block_weeks:
        feedback = week_record.get("feedback") or {}
        for day, exercises in feedback.items():
            for ex in exercises:
                name = ex.get("name", "")
                if not name:
                    continue
                if name not in exercises_used:
                    exercises_used.append(name)

                sets = ex.get("sets_data", [])
                if not sets:
                    continue

                # Get best set (highest weight)
                best_weight, best_reps = 0, 0
                for s in sets:
                    try:
                        w = float(s.get("weight", 0) or 0)
                        r = float(s.get("reps", 0) or 0)
                        if w > best_weight:
                            best_weight, best_reps = w, r
                    except (ValueError, TypeError):
                        pass

                if best_weight == 0:
                    continue

                if name not in exercise_data:
                    exercise_data[name] = {
                        "start_weight": best_weight,
                        "start_reps": best_reps,
                        "end_weight": best_weight,
                        "end_reps": best_reps,
                    }
                else:
                    exercise_data[name]["end_weight"] = best_weight
                    exercise_data[name]["end_reps"] = best_reps

    key_lifts = {}
    for name, data in exercise_data.items():
        if data["end_weight"] > data["start_weight"]:
            trend = "progressing"
        elif data["end_weight"] < data["start_weight"]:
            trend = "regressing"
        else:
            trend = "stalled" if data["end_reps"] <= data["start_reps"] else "progressing"

        key_lifts[name] = {
            "start": f"{data['start_weight']}kg x {int(data['start_reps'])}",
            "end": f"{data['end_weight']}kg x {int(data['end_reps'])}",
            "trend": trend,
        }

    return {
        "mesocycle": mesocycle,
        "weeks": len(block_weeks),
        "key_lifts": key_lifts,
        "exercises_used": exercises_used,
    }
```

- [ ] **Step 4: Add get_progression_generator helper**

Add after `get_program_generator`:

```python
def get_progression_generator(config: dict, writer_type: str = "progression") -> ProgressionProgramGenerator:
    """Build a ProgressionProgramGenerator for week 2+ generation."""
    writer_settings_key = writer_type
    writer_prompt_settings = WRITER_PROMPT_SETTINGS.get(writer_settings_key, WRITER_PROMPT_SETTINGS['progression'])

    critic_setting_key = 'week1' if writer_type == 'new_block' else 'progression'
    critic_prompt_settings = CRITIC_PROMPT_SETTINGS[critic_setting_key]

    llm_writer = setup_llm(
        model=config['model'],
        respond_as_json=True,
        temperature=config['writer_temperature'],
        top_p=config['writer_top_p'],
        thinking_budget=config.get('thinking_budget'),
    )
    llm_critic = setup_llm(
        model=config.get('critic_model', config['model']),
        max_tokens=config['max_tokens'],
        respond_as_json=False,
    )
    llm_analyst = setup_llm(
        model=config.get('critic_model', config['model']),
        respond_as_json=True,
    )

    task_revision = writer_prompt_settings.task_revision
    if not task_revision and 'revision' in WRITER_PROMPT_SETTINGS:
        task_revision = WRITER_PROMPT_SETTINGS['revision'].task_revision

    writer = Writer(
        model=llm_writer,
        role=writer_prompt_settings.role,
        structure=writer_prompt_settings.structure or WRITER_PROMPT_SETTINGS['initial'].structure,
        task=writer_prompt_settings.task,
        task_revision=task_revision,
        task_progression=getattr(writer_prompt_settings, 'task_progression', None),
        writer_type=writer_type,
        retrieval_fn=retrieve_and_generate,
    )
    critic = Critic(
        model=llm_critic,
        role=critic_prompt_settings.role,
        tasks=getattr(critic_prompt_settings, 'tasks', None),
        retrieval_fn=retrieve_and_generate,
    )
    editor = Editor()
    analyst = Analyst(model=llm_analyst, retrieval_fn=retrieve_and_generate)

    return ProgressionProgramGenerator(
        writer=writer, critic=critic, editor=editor, analyst=analyst,
        mesocycle_length=config.get('mesocycle_length', DEFAULT_MESOCYCLE_LENGTH),
        max_iterations=config.get('max_iterations', 1),
    )
```

- [ ] **Step 5: Rewrite the /next_week route**

Replace the entire `/next_week` route with the new version that uses the progression generator and SSE streaming (matches the existing `/generate` pattern):

```python
@app.route('/next_week', methods=['GET', 'POST'])
def next_week():
    if 'program' not in session:
        flash("No program available to generate next week's program")
        return redirect(url_for('generate_program'))

    program = session.get('program', {})
    current_week = session.get('current_week', 1)
    feedback_data = _parse_feedback_form(program, request.form, key_prefix=f"{current_week}_")
    session['feedback'] = feedback_data

    # Enrich current week's record with feedback
    all_programs = session.get('all_programs', [])
    for wp in all_programs:
        if wp.get('week') == current_week:
            wp['feedback'] = feedback_data
            break

    current_program = session['raw_program']
    if 'formatted' in current_program and isinstance(current_program['formatted'], dict):
        if 'weekly_program' in current_program['formatted']:
            if isinstance(current_program, dict) and 'weekly_program' not in current_program:
                current_program['weekly_program'] = current_program['formatted']['weekly_program']

    mesocycle = session.get('mesocycle', 1)
    week_in_mesocycle = session.get('week_in_mesocycle', current_week)
    mesocycle_length = session.get('mesocycle_length', DEFAULT_MESOCYCLE_LENGTH)
    block_summaries = session.get('block_summaries', [])

    # Build mesocycle history (only weeks in current mesocycle)
    current_mesocycle_history = [w for w in all_programs if w.get('mesocycle', 1) == mesocycle]

    user_input = session.get('user_input', '')
    if session.get('persona'):
        selected_persona = _get_personas().get(session['persona'])
        if selected_persona:
            user_input += f"\nTarget Persona: {selected_persona}"

    config = DEFAULT_CONFIG.copy()
    config['mesocycle_length'] = mesocycle_length

    # Create SSE job
    job_id = uuid.uuid4().hex[:12]
    q = queue.Queue()
    _generation_queues[job_id] = q

    @copy_current_request_context
    def _run_progression():
        try:
            q.put({"step": "analytics", "message": "Analyzing training history..."})

            # Start with progression type; the generator will determine if
            # a deload or mesocycle review is needed
            prog_gen = get_progression_generator(config, writer_type="progression")
            prog_gen.on_status = lambda msg: q.put(msg)

            state = prog_gen.create_program(
                user_input=user_input,
                current_mesocycle_history=current_mesocycle_history,
                week_in_mesocycle=week_in_mesocycle,
                previous_block_summaries=block_summaries,
                feedback=feedback_data,
                previous_draft=current_program.get('formatted') or current_program,
            )

            if state.get("needs_approval"):
                # Analyst produced a decision — pause for user approval
                _generation_results[job_id] = {
                    "state": state,
                    "config": config,
                    "user_input": user_input,
                    "current_mesocycle_history": current_mesocycle_history,
                    "block_summaries": block_summaries,
                }
                q.put({
                    "step": "review",
                    "message": "Program review ready",
                    "analyst_decision": state.get("analyst_decision", {}),
                    "analytics": state.get("analytics", {}),
                    "job_id": job_id,
                })
            else:
                # Normal progression — full pipeline already ran
                parsed_program = parse_program(state.get('formatted'))
                new_week = current_week + 1
                _generation_results[job_id] = {
                    "program": parsed_program,
                    "raw_program": state,
                    "new_week": new_week,
                    "review_type": "normal",
                    "mesocycle": mesocycle,
                    "week_in_mesocycle": week_in_mesocycle + 1,
                }
                q.put({"step": "done", "message": "Program generated successfully!", "job_id": job_id})

        except Exception as e:
            logger.exception("Progression generation failed")
            q.put({"step": "error", "message": str(e)})
        finally:
            _generation_queues.pop(job_id, None)

    thread = threading.Thread(target=_run_progression, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})
```

- [ ] **Step 6: Add /approve_review endpoint**

Add after the `/next_week` route:

```python
@app.route('/approve_review', methods=['POST'])
def approve_review():
    """Resume generation after user approves an analyst decision."""
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')
    if not job_id or job_id not in _generation_results:
        return jsonify({'success': False, 'message': 'Review session not found'}), 404

    stored = _generation_results.pop(job_id)
    state = stored['state']
    config = stored['config']

    review_type = state['analytics']['review_type']

    # Determine the correct writer type for the continuation
    writer_type = 'deload' if review_type == 'deload' else 'new_block'

    job_id_2 = uuid.uuid4().hex[:12]
    q = queue.Queue()
    _generation_queues[job_id_2] = q

    @copy_current_request_context
    def _run_continuation():
        try:
            q.put({"step": "writer", "message": f"Generating {review_type.replace('_', ' ')} program..."})

            prog_gen = get_progression_generator(config, writer_type=writer_type)
            prog_gen.on_status = lambda msg: q.put(msg)

            state['user_approved'] = True
            result = prog_gen.continue_after_approval(state)

            parsed_program = parse_program(result.get('formatted'))
            current_week = session.get('current_week', 1)
            mesocycle = session.get('mesocycle', 1)
            week_in_meso = session.get('week_in_mesocycle', 1)

            if review_type == 'deload':
                # Deload is an inserted week — does not advance mesocycle position
                new_week = current_week + 1
                new_mesocycle = mesocycle
                new_week_in_meso = week_in_meso  # stays the same
            else:
                # New mesocycle block
                new_week = current_week + 1
                new_mesocycle = mesocycle + 1
                new_week_in_meso = 1

            _generation_results[job_id_2] = {
                "program": parsed_program,
                "raw_program": result,
                "new_week": new_week,
                "review_type": review_type,
                "mesocycle": new_mesocycle,
                "week_in_mesocycle": new_week_in_meso,
                "analyst_decision": state.get("analyst_decision"),
            }
            q.put({"step": "done", "message": "Program generated successfully!", "job_id": job_id_2})

        except Exception as e:
            logger.exception("Post-approval generation failed")
            q.put({"step": "error", "message": str(e)})
        finally:
            _generation_queues.pop(job_id_2, None)

    thread = threading.Thread(target=_run_continuation, daemon=True)
    thread.start()
    return jsonify({"job_id": job_id_2})
```

- [ ] **Step 7: Add /next_week/complete endpoint**

Add a completion endpoint for the next_week SSE flow (similar to `/generate/complete/<job_id>`):

```python
@app.route('/next_week/complete/<job_id>')
def next_week_complete(job_id):
    """Load next-week generation result into the session and redirect to index."""
    result = _generation_results.pop(job_id, None)
    if not result:
        flash("Generation result expired or not found.")
        return redirect(url_for('index'))

    parsed_program = result['program']
    new_week = result['new_week']
    review_type = result['review_type']
    new_mesocycle = result['mesocycle']
    new_week_in_meso = result['week_in_mesocycle']

    # If starting a new mesocycle, build summary of the completed one
    old_mesocycle = session.get('mesocycle', 1)
    if new_mesocycle > old_mesocycle:
        all_programs = session.get('all_programs', [])
        summary = _build_block_summary(all_programs, old_mesocycle)
        if summary:
            block_summaries = session.get('block_summaries', [])
            block_summaries.append(summary)
            session['block_summaries'] = block_summaries

    session['program'] = parsed_program
    session['raw_program'] = result['raw_program']
    session['feedback'] = {}
    session['current_week'] = new_week
    session['mesocycle'] = new_mesocycle
    session['week_in_mesocycle'] = new_week_in_meso

    all_programs = session.get('all_programs', [])
    week_record = {
        'week': new_week,
        'mesocycle': new_mesocycle,
        'week_in_mesocycle': new_week_in_meso,
        'type': review_type,
        'program': parsed_program,
    }
    if result.get('analyst_decision'):
        week_record['analyst_decision'] = result['analyst_decision']
    all_programs.append(week_record)
    session['all_programs'] = all_programs

    flash(f"Week {new_week} program generated successfully!")
    return redirect(url_for('index'))
```

- [ ] **Step 8: Update /save_program and /load_program for new session fields**

In `save_program` (around line 496), add the new fields to `save_data`:

```python
        save_data = {
            'program_name': program_name,
            'date_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_input': session.get('user_input', ''),
            'persona': session.get('persona', ''),
            'current_week': session.get('current_week', 1),
            'mesocycle': session.get('mesocycle', 1),
            'week_in_mesocycle': session.get('week_in_mesocycle', 1),
            'mesocycle_length': session.get('mesocycle_length', DEFAULT_MESOCYCLE_LENGTH),
            'raw_program': session.get('raw_program', {}),
            'all_programs': session.get('all_programs', []),
            'block_summaries': session.get('block_summaries', []),
        }
```

In `load_program` (around line 555), restore the new fields with backward-compatible defaults:

```python
        # Restore session
        session['program'] = data.get('all_programs', [])[-1].get('program', {}) if data.get('all_programs') else {}
        session['raw_program'] = data.get('raw_program', {})
        session['user_input'] = data.get('user_input', '')
        session['persona'] = data.get('persona', '')
        session['current_week'] = data.get('current_week', 1)
        session['all_programs'] = data.get('all_programs', [])
        session['feedback'] = {}
        # New mesocycle fields — backward-compatible defaults
        session['mesocycle'] = data.get('mesocycle', 1)
        session['week_in_mesocycle'] = data.get('week_in_mesocycle', data.get('current_week', 1))
        session['mesocycle_length'] = data.get('mesocycle_length', DEFAULT_MESOCYCLE_LENGTH)
        session['block_summaries'] = data.get('block_summaries', [])
        session['pending_review'] = None
```

- [ ] **Step 9: Verify app imports work**

Run: `python -c "import app; print('OK')"`
Expected: `OK` (or import succeeds — it may warn about missing env vars but shouldn't error on import structure)

- [ ] **Step 10: Commit**

```bash
git add app.py
git commit -m "feat: integrate progression generator with SSE streaming and approval flow in app.py"
```

---

### Task 10: Frontend — Review UI in index.html

**Files:**
- Modify: `templates/index.html`

This task adds the review overlay that appears when the analyst produces a decision document. The SSE stream from `/next_week` can now emit a `step: "review"` event with the analyst's reasoning and recommendations. The frontend shows an overlay with the analysis and Approve/Skip buttons.

- [ ] **Step 1: Read index.html to find the right insertion points**

Read the current `templates/index.html` to identify:
1. Where the "next week" form/button is
2. Where SSE handling JS lives (if any — the next_week flow may not have SSE yet since it was synchronous before)

The existing `/next_week` route was a synchronous POST. We're converting it to async (SSE) like `/generate`. The frontend JS needs to:
1. POST to `/next_week` to get a `job_id`
2. Open an SSE connection to `/generate/stream/<job_id>` (reuse the same stream endpoint)
3. Handle `step: "review"` events by showing the approval overlay
4. Handle `step: "done"` events by redirecting to `/next_week/complete/<job_id>`

- [ ] **Step 2: Add the review overlay HTML**

Add this HTML block inside the `<body>` of `index.html`, before the closing `</body>` tag:

```html
<!-- Review Overlay -->
<div id="review-overlay" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.85); z-index:9999; overflow-y:auto; padding:2rem;">
  <div style="max-width:700px; margin:2rem auto; background:#1a1a1a; border-radius:16px; padding:2rem; border:1px solid #333;">
    <h2 style="color:#FF6B2B; font-family:'Barlow Condensed',sans-serif; font-size:1.6rem; margin-bottom:1rem;" id="review-title">Program Review</h2>
    <div id="review-reasoning" style="color:#ccc; font-family:'DM Sans',sans-serif; font-size:0.95rem; line-height:1.6; margin-bottom:1.5rem; padding:1rem; background:#111; border-radius:8px;"></div>
    <div id="review-recommendations" style="margin-bottom:1.5rem;"></div>
    <div style="display:flex; gap:1rem; justify-content:flex-end;">
      <button id="review-skip-btn" style="padding:0.6rem 1.5rem; border-radius:8px; border:1px solid #555; background:transparent; color:#ccc; font-family:'Barlow Condensed',sans-serif; font-size:1rem; cursor:pointer;">Skip Review</button>
      <button id="review-approve-btn" style="padding:0.6rem 1.5rem; border-radius:8px; border:none; background:#FF6B2B; color:#fff; font-family:'Barlow Condensed',sans-serif; font-size:1rem; cursor:pointer;">Approve & Generate</button>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Add the review overlay JavaScript**

Add this `<script>` block before the closing `</body>` tag (after the overlay HTML):

```html
<script>
(function() {
  const overlay = document.getElementById('review-overlay');
  const titleEl = document.getElementById('review-title');
  const reasoningEl = document.getElementById('review-reasoning');
  const recsEl = document.getElementById('review-recommendations');
  const approveBtn = document.getElementById('review-approve-btn');
  const skipBtn = document.getElementById('review-skip-btn');

  let pendingJobId = null;

  function showReview(data) {
    const decision = data.analyst_decision || {};
    const analytics = data.analytics || {};
    const reviewType = decision.review_type || analytics.review_type || 'review';

    titleEl.textContent = reviewType === 'deload'
      ? 'Deload Week Recommended'
      : 'Mesocycle Review — End of Block';

    reasoningEl.textContent = decision.reasoning || 'No detailed reasoning provided.';

    recsEl.innerHTML = '';
    const recs = decision.recommendations || [];
    if (recs.length > 0) {
      const heading = document.createElement('h3');
      heading.textContent = 'Proposed Changes';
      heading.style.cssText = "color:#FF6B2B; font-family:'Barlow Condensed',sans-serif; font-size:1.2rem; margin-bottom:0.75rem;";
      recsEl.appendChild(heading);

      recs.forEach(function(rec) {
        const div = document.createElement('div');
        div.style.cssText = "padding:0.75rem 1rem; background:#111; border-radius:8px; margin-bottom:0.5rem; border-left:3px solid #FF6B2B;";
        if (rec.type === 'swap') {
          div.innerHTML = '<span style="color:#38BDF8; font-family:\'DM Mono\',monospace;">' + rec.exercise + '</span>'
            + ' → <span style="color:#38BDF8; font-family:\'DM Mono\',monospace;">' + rec.replacement + '</span>'
            + '<br><span style="color:#999; font-size:0.85rem;">' + rec.reason + '</span>';
        } else if (rec.type === 'adjust_volume') {
          div.innerHTML = '<span style="color:#38BDF8; font-family:\'DM Mono\',monospace;">' + rec.muscle_group + '</span>'
            + ': ' + rec.change
            + '<br><span style="color:#999; font-size:0.85rem;">' + rec.reason + '</span>';
        }
        recsEl.appendChild(div);
      });
    }

    if (decision.deload) {
      const deloadDiv = document.createElement('div');
      deloadDiv.style.cssText = "padding:1rem; background:#111; border-radius:8px; margin-bottom:0.5rem;";
      deloadDiv.innerHTML = '<span style="color:#FF6B2B; font-weight:600;">Deload Plan:</span> '
        + 'Reduce volume by ~' + Math.round((decision.deload.volume_reduction || 0.5) * 100) + '%, '
        + 'intensity: ' + (decision.deload.intensity || 'moderate');
      recsEl.appendChild(deloadDiv);
    }

    pendingJobId = data.job_id;
    overlay.style.display = 'block';
  }

  approveBtn.addEventListener('click', function() {
    if (!pendingJobId) return;
    overlay.style.display = 'none';
    approveBtn.disabled = true;

    fetch('/approve_review', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({job_id: pendingJobId})
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.job_id) {
        listenToStream(data.job_id);
      }
    })
    .catch(function(err) { console.error('Approve failed:', err); });
  });

  skipBtn.addEventListener('click', function() {
    overlay.style.display = 'none';
    pendingJobId = null;
  });

  function listenToStream(jobId) {
    const es = new EventSource('/generate/stream/' + jobId);
    es.onmessage = function(e) {
      const msg = JSON.parse(e.data);
      if (msg.step === 'done') {
        es.close();
        window.location.href = '/next_week/complete/' + msg.job_id;
      } else if (msg.step === 'error') {
        es.close();
        alert('Error: ' + msg.message);
      }
      // Could show progress updates here
    };
    es.onerror = function() { es.close(); };
  }

  // Expose for the next_week SSE handler
  window._liftai_showReview = showReview;
  window._liftai_listenToStream = listenToStream;
})();
</script>
```

- [ ] **Step 4: Update the next-week form submission to use SSE**

Find the existing next-week form submit handler in the `<script>` section of `index.html`. It currently does a synchronous form POST. Replace it with an async fetch + SSE pattern. The exact code depends on how the form is currently structured, but the pattern is:

```javascript
// In the existing next-week form submit handler, replace the synchronous submit with:
function submitNextWeek(formElement) {
  const formData = new FormData(formElement);
  fetch('/next_week', {
    method: 'POST',
    body: formData,
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.job_id) {
      const es = new EventSource('/generate/stream/' + data.job_id);
      es.onmessage = function(e) {
        const msg = JSON.parse(e.data);
        if (msg.step === 'review') {
          es.close();
          window._liftai_showReview(msg);
        } else if (msg.step === 'done') {
          es.close();
          window.location.href = '/next_week/complete/' + msg.job_id;
        } else if (msg.step === 'error') {
          es.close();
          alert('Error: ' + msg.message);
        }
      };
      es.onerror = function() { es.close(); };
    }
  })
  .catch(function(err) { alert('Error: ' + err.message); });
}
```

NOTE: The exact selectors and form IDs depend on the current `index.html` structure. Read the file first and adapt this code to match the existing form element and submit button. Intercept the form's submit event with `e.preventDefault()` and call `submitNextWeek(formElement)`.

- [ ] **Step 5: Commit**

```bash
git add templates/index.html
git commit -m "feat: add review overlay UI and SSE-based next-week generation flow"
```

---

### Task 11: Verify End-to-End Integration

**Files:** None (testing only)

- [ ] **Step 1: Run all analytics tests**

Run: `python -m pytest tests/test_analytics.py -v`
Expected: All tests PASS

- [ ] **Step 2: Verify all imports resolve**

Run: `python -c "from agent_system import ProgramGenerator, Analyst, Writer, Critic, Editor; from agent_system.generator import ProgressionProgramGenerator; from agent_system.analytics import analyze_training_history; from prompts import ANALYST_ROLE, TASK_MESOCYCLE_REVIEW, TASK_DELOAD; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Start the Flask app and verify it boots**

Run: `python app.py`
Expected: Flask development server starts without import errors. (It will need a valid API key to actually generate programs, but boot should succeed.)

- [ ] **Step 4: Commit any fixes needed**

If any issues were found, fix and commit.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete autoregulation and mesocycle progression system"
```
