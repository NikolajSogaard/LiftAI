# Autoregulation & Mesocycle Progression System — Design Spec

## Overview

Extend LiftAI's week-to-week progression with intelligent autoregulation: detect stagnation and fatigue from accumulated performance data, trigger program reviews at mesocycle boundaries (or early when warranted), and let a dedicated Analyst agent recommend exercise swaps, volume adjustments, and deload weeks — all subject to user approval before execution.

## Goals

- **Steady long-term progression**: automatically detect when the current program stops working and recommend evidence-based changes.
- **Transparent decision-making**: every recommendation is backed by computed metrics and LLM reasoning the user can inspect.
- **User stays in control**: the system suggests, the user approves/edits/skips.

## Architecture Summary

Two new components added to the existing pipeline:

1. **Python Analytics Module** (`agent_system/analytics.py`) — deterministic metric computation (stagnation, fatigue, RPE trends). Decides the review type: `normal`, `deload`, or `mesocycle_review`.
2. **Analyst Agent** (`agent_system/agents/analyst.py`) — LLM-powered agent that interprets analytics output and produces a structured decision document with exercise swap recommendations, volume adjustments, and deload plans.

These feed into the existing Writer -> Critic -> Editor pipeline via an updated LangGraph workflow with conditional routing.

---

## Section 1: Data Model — Training History

### Per-Week Record

Each entry in `all_programs` is enriched to:

```python
{
    "week": 3,
    "mesocycle": 1,
    "week_in_mesocycle": 3,
    "type": "normal",            # "normal" | "deload" | "mesocycle_review"
    "program": { ... },          # the weekly_program dict
    "feedback": {                # user-submitted performance data
        "Day 1": [
            {
                "name": "Bench Press",
                "sets_data": [
                    {"weight": "80", "reps": "8", "actual_rpe": "7"},
                    {"weight": "80", "reps": "8", "actual_rpe": "8"},
                    {"weight": "80", "reps": "7", "actual_rpe": "9"}
                ],
                "overall_feedback": ""
            }
        ]
    },
    "analyst_decision": { ... }  # only present on review/deload weeks
}
```

### Condensed Block Summary

Generated at the end of each mesocycle and carried forward:

```python
{
    "mesocycle": 1,
    "weeks": 4,
    "split": "Upper/Lower 4x/week",
    "key_lifts": {
        "Bench Press": {"start": "70kg x 8", "end": "80kg x 8", "trend": "progressing"},
        "Squat": {"start": "100kg x 5", "end": "105kg x 6", "trend": "progressing"},
        "RDL": {"start": "80kg x 10", "end": "80kg x 10", "trend": "stalled"}
    },
    "exercises_used": ["Bench Press", "Squat", "RDL", "Lat Pulldown", ...],
    "avg_rpe_trend": "rising",
    "volume_per_muscle_group": { ... },
    "notes": "RDL stalled at week 3. Deload triggered at week 4."
}
```

### Tiered Context for LLM

| Layer | Data | When used |
|-------|------|-----------|
| Recent week | Full set-by-set feedback | Every week |
| Current mesocycle | All weekly records (program + feedback) | Every week |
| Previous blocks | Condensed summaries only | Mesocycle reviews |

---

## Section 2: Python Analytics Module

New file: `agent_system/analytics.py`. Pure Python, no LLM calls.

### Inputs

- Current mesocycle's weekly records (program + feedback)
- Mesocycle config (default length from `config.py`)

### Per-Exercise Metrics

- **Stagnation score**: Consecutive weeks with no weight or rep increase. Flagged at 2+ weeks.
- **RPE trend**: Linear direction over the mesocycle (`rising`, `stable`, `falling`). Computed from average actual RPE per exercise per week.
- **Load progression rate**: Total weight increase (kg) over the mesocycle.
- **Rep completion rate**: Percentage of sets where actual reps met or exceeded target.

### Global Metrics

- **Average RPE trend**: Weighted average RPE across all exercises per week.
- **Fatigue score**: Composite signal:
  ```
  fatigue = (0.4 * rpe_rise) + (0.3 * rep_decline) + (0.3 * stall_ratio)
  ```
  Each component normalized to 0.0–1.0.
- **Mesocycle position**: `week_in_mesocycle / mesocycle_length`.

### Decision Output

```python
{
    "review_type": "normal",  # "normal" | "deload" | "mesocycle_review"
    "triggers": [],           # list of reason strings if not "normal"
    "exercise_flags": {
        "RDL": {"stagnation_weeks": 3, "rpe_trend": "rising", "flag": "stalled"},
        "Bench Press": {"stagnation_weeks": 0, "rpe_trend": "stable", "flag": "progressing"}
    },
    "global_metrics": {
        "avg_rpe_trend": "rising",
        "fatigue_score": 0.72,
        "stalled_exercise_ratio": 0.25,
        "mesocycle_position": 0.75
    }
}
```

### Trigger Rules

| Condition | Review Type |
|-----------|------------|
| End of mesocycle (`week_in_mesocycle == mesocycle_length`) | `mesocycle_review` |
| `fatigue_score > 0.7` at any point | `deload` (early trigger) |
| `stalled_exercise_ratio > 0.5` | `mesocycle_review` (early trigger) |
| None of the above | `normal` |

Thresholds are configurable constants in `config.py`.

**Deload and mesocycle counting:** A deload triggered mid-mesocycle is an *inserted* recovery week — it does not count toward the mesocycle length. After the deload, the mesocycle resumes from where it left off. For example, if a deload triggers at week 3 of a 4-week block, the sequence is: Week 3 -> Deload -> Week 4 (mesocycle review).

---

## Section 3: Analyst Agent

New file: `agent_system/agents/analyst.py`. LLM-powered agent with RAG access.

### When It Runs

- **Normal weeks**: Skipped. Writer gets exercise flags from analytics directly.
- **Deload weeks**: Runs, produces a deload plan.
- **Mesocycle review weeks**: Runs the full review with exercise swap and volume recommendations.

### Inputs

```python
{
    "review_type": "mesocycle_review",
    "analytics": { ... },
    "current_mesocycle_history": [ ... ],
    "previous_block_summaries": [ ... ],
    "user_input": "...",
    "rag_context": "..."
}
```

### Output — Decision Document

**Mesocycle review:**
```python
{
    "review_type": "mesocycle_review",
    "reasoning": "RPE has been trending upward for 3 weeks. RDL and Lateral Raise have stalled...",
    "recommendations": [
        {
            "type": "swap",
            "exercise": "RDL",
            "replacement": "Stiff-Leg Deadlift",
            "reason": "Stalled 3 weeks at 80kg. Variation targets same posterior chain with different stimulus."
        },
        {
            "type": "swap",
            "exercise": "Lateral Raise",
            "replacement": "Cable Lateral Raise",
            "reason": "Stalled 2 weeks. Cable variation provides more consistent tension curve."
        },
        {
            "type": "adjust_volume",
            "muscle_group": "Upper horizontal push",
            "change": "+2 sets/week",
            "reason": "Bench press progressing well — can handle more volume in new block."
        }
    ],
    "deload": null,
    "next_mesocycle_length": 4
}
```

**Deload:**
```python
{
    "review_type": "deload",
    "reasoning": "Fatigue score 0.78 — RPE rising across all compounds, rep completion declining...",
    "recommendations": [],
    "deload": {
        "volume_reduction": 0.5,
        "intensity": "moderate",
        "duration_weeks": 1
    },
    "next_mesocycle_length": null
}
```

### RAG Queries

- Stalled exercises: "When should you swap exercises in a strength program and what are good variations for [exercise]?"
- Deloads: "How to program deload weeks — volume and intensity recommendations"
- Mesocycle transitions: "How to structure mesocycle transitions and exercise rotation for continued progression"

### Change Scope Rules

- Preserve overall split structure (Upper/Lower stays Upper/Lower)
- Only swap exercises with clear stagnation signals (2+ weeks)
- Main compound lifts: suggest variations only (Bench -> Close-Grip Bench), not completely different movements
- Accessories: can be swapped more freely
- Reasoning must be transparent and data-driven

---

## Section 4: Updated LangGraph Workflow

### Week 1 (unchanged)
```
START -> Writer -> Critic -> [Editor | Reflector -> Writer]
```

### Week 2+ Normal Progression
```
START -> Analytics -> Writer(progression) -> Critic(progression) -> Editor
```
Analytics returns `review_type: "normal"`. Analyst is skipped. Writer receives exercise flags for smarter per-exercise decisions.

### Week 2+ Deload or Mesocycle Review
```
START -> Analytics -> Analyst -> [User Approval Gate] -> Writer -> Critic -> Editor
```

### Conditional Routing

```python
def route_after_analytics(state):
    review_type = state["analytics"]["review_type"]
    if review_type == "normal":
        return "writer"
    return "analyst"

def route_after_analyst(state):
    return "approval_gate"

def route_after_approval(state):
    if state.get("user_approved"):
        return "writer"
    return END
```

### User Approval Gate

1. Analytics + Analyst run in the background thread (existing SSE pattern)
2. SSE pushes decision document to frontend with `step: "review"`
3. Frontend shows review UI (reasoning + recommendations + Approve/Edit/Skip)
4. User response sent via new `POST /approve_review` endpoint
5. Endpoint resumes generation thread -> Writer -> Critic -> Editor

### Writer Context by Review Type

| Review type | Writer receives |
|-------------|----------------|
| `normal` | Previous program + feedback + exercise flags from analytics |
| `deload` | Previous program + analyst deload plan |
| `mesocycle_review` | Previous program + analyst decision document + block summaries |

### Critic Behavior by Review Type

| Review type | Critic mode |
|-------------|-------------|
| `normal` | `progression` (single task — same as today) |
| `deload` | Skipped (deload programs are simple, no critique needed) |
| `mesocycle_review` | `week1` (full 5-task critique — new block is essentially a new program) |

---

## Section 5: Storage & Session Changes

### New Session Fields

```python
session['mesocycle'] = 1
session['week_in_mesocycle'] = 3
session['mesocycle_length'] = 4
session['block_summaries'] = []
session['pending_review'] = None   # analyst decision awaiting approval
```

### Saved Program Format

Extends current format with:

```python
{
    # ... existing fields ...
    "mesocycle": 2,
    "week_in_mesocycle": 4,
    "mesocycle_length": 4,
    "block_summaries": [ ... ],
}
```

### Backward Compatibility

Loading old saves without mesocycle data defaults to: `mesocycle: 1`, `week_in_mesocycle: current_week`, `block_summaries: []`.

### Config Additions (`config.py`)

```python
DEFAULT_MESOCYCLE_LENGTH = 4
STAGNATION_THRESHOLD_WEEKS = 2
FATIGUE_SCORE_DELOAD_TRIGGER = 0.7
STALL_RATIO_REVIEW_TRIGGER = 0.5
DELOAD_VOLUME_REDUCTION = 0.5
```

---

## Section 6: New Prompts

### Analyst Prompts (`prompts/analyst_prompts.py`)

**Role:**
```python
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
```

**Mesocycle Review Task:** Receives analytics, mesocycle history, block summaries, user profile. Outputs reasoning + recommendations (swaps, volume adjustments, next block length).

**Deload Task:** Receives analytics, mesocycle history, user profile. Outputs reasoning + volume reduction plan + intensity guidance.

### New Writer Tasks (`prompts/writer_prompts.py`)

**TASK_DELOAD_WRITER:** Generates a deload week — keeps all exercises, reduces sets per analyst plan, lowers RPE targets.

**TASK_NEW_BLOCK:** Generates Week 1 of new mesocycle — implements approved swaps, applies volume adjustments, sets conservative initial weights (last working weight for retained exercises, 10-15% lighter for new exercises), resets RPE to moderate levels.

---

## Section 7: File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `agent_system/analytics.py` | Python analytics — stagnation, fatigue, RPE trends |
| `agent_system/agents/analyst.py` | Analyst agent — decision documents |
| `prompts/analyst_prompts.py` | Analyst role + task templates |

### Modified Files

| File | Changes |
|------|---------|
| `config.py` | Mesocycle/autoregulation constants |
| `agent_system/generator.py` | Updated LangGraph workflow with conditional routing |
| `agent_system/agents/writer.py` | New writer types (deload, new_block) + enriched progression with exercise flags |
| `prompts/writer_prompts.py` | Add TASK_DELOAD_WRITER, TASK_NEW_BLOCK |
| `app.py` | New session fields, updated /next_week, new /approve_review endpoint, block summary generation |
| `templates/index.html` | Review UI with Approve/Edit/Skip |

### Unchanged

- Week 1 generation flow
- RAG pipeline, FAISS, embeddings
- Chatbot
- Save/Load (backward-compatible)
- Core Writer -> Critic -> Editor pattern
