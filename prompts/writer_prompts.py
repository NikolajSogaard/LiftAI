import dataclasses
from typing import Optional


@dataclasses.dataclass
class WriterPromptSettings:
    role: dict[str, str]
    structure: str
    task: Optional[str] = None
    task_revision: Optional[str] = None
    task_progression: Optional[str] = None

    def save(self, fname: str):
        raise NotImplementedError

    def load(self, fname: str):
        raise NotImplementedError


# Initial program creation task
TASK_INITIAL = '''
Create the best strength training program for this individual:
{}

STEP 1 — PLANNING (fill in the "reasoning" field):
Before writing the program, briefly think through:
- What training split and frequency fits this person's schedule, goals, and experience level?
- Which exercises and movement patterns are most appropriate and why?
- What weekly set volume is right for their level across all major muscle groups?

STEP 2 — PROGRAM (fill in "weekly_program"):
Output the complete program following this JSON structure exactly:
{}

Guidelines:
- Match the program precisely to the user's experience level, goals, available training days, equipment, and any injuries or limitations they mentioned.
- Ensure sufficient weekly frequency for every major muscle group (chest, back, shoulders, arms, quads, posterior chain).
- Order exercises sensibly: compound movements first, isolation work later.
- Keep weekly set volumes appropriate for the experience level.
- Make the cues field genuinely useful: a brief, specific coaching note on form or intent for that exercise.
'''

# Revision task based on critic feedback
TASK_REVISION = '''
Revise the program below:
{}

Based on feedback from your colleague below:
{}

IMPORTANT: 
- Address all the feedback provided in the critique and adjust the program based on those suggestions.
- Make sure to adjust the program according to the critique from: frequency and split, exercise selection, set volume, rep ranges, and RIR (Reps In Reserve).
- You MUST always directly implement all the suggested changes in the program itself from the critique, not just in the suggestion field.
- Maintain the same number of training days unless the feedback explicitly suggests changing it.
Follow this JSON structure as a guide for your response:
{}
'''


# Progressive overload task for Week 2+
TASK_PROGRESSION = '''
Create the next week's training program based on:

1) The previous week's program:
{}

2) The detailed feedback and performance data:
{}

IMPORTANT:
- ONLY modify the "AI Progression" field - keep all exercises, sets, rep ranges and rest periods identical
- YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT FOR LOAD ADJUSTMENTS WITH NO VARIATION:
  Set 1:(8 reps @ 80kg, RIR 3)
  Set 2:(8 reps @ 80kg, RIR 2)
  Set 3:(7 reps @ 80kg, RIR 1)
        75kg ↓

- YOUR RESPONSE MUST FOLLOW EXACTLY THIS FORMAT FOR REP ADJUSTMENTS WITH NO VARIATION:
  Set 1:(8 reps @ 80kg, RIR 3)
  Set 2:(8 reps @ 80kg, RIR 2)
  Set 3:(7 reps @ 80kg, RIR 1)
        10 reps ↑

- First line must be "Set 1:" followed by performance data in parentheses "(reps @ weight, RIR score)"
- Include ONE line per set showing the actual performance data from last week
- Then provide ONE line with ONLY the adjustment with arrow symbol - nothing else
- Use "↑" for increases and "↓" for decreases
- For weight changes: "85kg ↑" or "75kg ↓"
- For rep changes: "10 reps ↑" or "8 reps ↓"
- DO NOT include any other explanatory text whatsoever
- DO NOT include phrases like "Based on your performance" or "Aim for" or "Target RIR"
- DO NOT include any recommendations about RIR targets
- If no performance data is available, leave the suggestion field empty

IMPORTANT: Your AI Progression field must contain ONLY the set data and adjustment line as shown above.

Follow this JSON structure as a guide for your response:
{}
'''

PROGRAM_STRUCTURE_WEEK1 = '''
{
  "reasoning": "Brief planning notes: why this split, why these exercises, why this volume for this person.",
  "weekly_program": {
    "Day 1": [
      {
        "name": "Exercise name",
        "sets": 3,
        "reps": "8-12",
        "target_rir": "2-3",
        "rest": "60-90 seconds",
        "cues": "Specific coaching note on form, focus, or purpose for this exercise"
      },
      {
        "name": "Exercise name",
        "sets": 3,
        "reps": "8-12",
        "target_rir": "2-3",
        "rest": "2 minutes",
        "cues": "Specific coaching note on form, focus, or purpose for this exercise"
      }
    ],
    "Day 2": [
      {
        "name": "Exercise name",
        "sets": 4,
        "reps": "5-8",
        "target_rir": "1-2",
        "rest": "2-3 minutes",
        "cues": "Specific coaching note on form, focus points, or exercise purpose"
      }
    ]
  }
}
(continue for all training days)
'''

# Specific role for initial program creation
INITIAL_WRITER_ROLE = {
    'role': 'system',
    'content': 'You are an AI system specialized in creating initial strength training programs, with expertise in exercise science.' 
                'Your task is to create effective and evidence-based strength training programs tailored to the user’s needs, goals, and experience level. '
                'Focus on establishing the right training frequency, training split, weekly set volume, and exercise selection for the users exerpience level '
                'Provide clear CONCISE, actionable instructions that are appropriate for the specified experience level and don’t go outside the scope of your tasks.'
}

# Specific role for program revision
REVISION_WRITER_ROLE = {
    'role': 'system',
    'content': 'You are an AI system specialized in revising strength training programs based on feedback, with expertise in exercise science.' 
                'Your task is to implement specific feedback and improvements to existing training program.'
                'Focus on addressing weaknesses identified by critics while maintaining program coherence. '
                'Always ensure changes are evidence-based and maintain the program\'s overall structure unless explicitly required to change it.'
                'Provide clear CONCISE adjustments that directly address the feedback given.' 
}


# Enhanced role for progression writer
PROGRESSION_ROLE = {
    'role': 'system',
    'content': 'You are an AI system specialized adjustion strength training programs based on previous weeks performance' 
                'Your task is to analyze previous performance data and provide specific progression recommendations. '
                'ONLY provide specific weight and effort suggestions in the "AI Progression" field. KEEP the rest of the program identical to the previous week.'
                'Analyze the actual performance (weights, reps achieved, RIR reported) to make data-driven decisions. '
                'Be specific with weight in kg or rep recommendations and explain your reasoning briefly.'
}

# Dictionary to store all prompt settings
WRITER_PROMPT_SETTINGS: dict[str, WriterPromptSettings] = {}

# Initial program creation settings - only has TASK_INITIAL with its own role
WRITER_PROMPT_SETTINGS['initial'] = WriterPromptSettings(
    role=INITIAL_WRITER_ROLE,
    task=TASK_INITIAL,
    structure=PROGRAM_STRUCTURE_WEEK1,
)

# Revision based on critic feedback - only has TASK_REVISION with its own role
WRITER_PROMPT_SETTINGS['revision'] = WriterPromptSettings(
    role=REVISION_WRITER_ROLE,
    task_revision=TASK_REVISION,
    structure=PROGRAM_STRUCTURE_WEEK1,
)

# Week 2+ progression - should use task_progression, not task_revision
WRITER_PROMPT_SETTINGS['progression'] = WriterPromptSettings(
    role=PROGRESSION_ROLE,
    task_progression=TASK_PROGRESSION,
    structure=None,
)

# Deload writer
TASK_DELOAD_WRITER = '''Generate a deload week program based on:

1) The previous week's program:
{}

2) The analyst's deload recommendations:
{}

IMPORTANT:
- Keep all exercises identical to the previous week
- Reduce sets per the analyst's volume reduction plan (approximately 40-50% fewer sets)
- Maintain weights but raise target RIR by 1-2 points (leave more reps in reserve)
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
- Reset RIR targets to moderate levels (2-4 for compounds, 1-3 for isolation)
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
        'Set conservative initial loads for new exercises and moderate RIR targets across the board. '
        'Provide clear, CONCISE output with no additional commentary.'
    )
}

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

# Add the original v1 as an alias to initial for backward compatibility
WRITER_PROMPT_SETTINGS['v1'] = WRITER_PROMPT_SETTINGS['initial']
