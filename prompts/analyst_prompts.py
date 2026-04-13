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
