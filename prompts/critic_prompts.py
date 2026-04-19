import dataclasses
from typing import Dict, Optional, List

# Define reusable prompt components
COMMON_RESPONSE_FORMAT = '''
IMPORTANT:
1. Provide SPECIFIC, CONCRETE suggestions - don't just identify problems
2. For each change, specify exactly what to modify and how
3. If nothing needs improvement, return "None"
'''

# Shared critic response constraints — appended to role so they're sent ONCE
# per critic LLM (not duplicated into all 5 task templates).
COMMON_RESPONSE_CONSTRAINTS = '''
RESPONSE CONSTRAINTS (apply to every critique):
- Keep responses under ~200 words. Bullet points only. No preamble, no meta-commentary.
- Provide SPECIFIC, CONCRETE suggestions: exact exercise, exact set count, exact rep range, exact RPE, exact day.
- If nothing meaningful needs improvement, return exactly "None".
'''

# Critic operating principles — applied across all tasks
COMMON_CRITIC_PRINCIPLES = '''
OPERATING PRINCIPLES (apply to every critique):

1. THINK BEFORE CRITIQUING
   - State the user attribute you're leaning on (goal, experience, days/week) when a call could go either way.
   - If the user's input supports multiple valid training approaches, name the tradeoff instead of silently picking one.
   - If something in the input is genuinely ambiguous, flag it rather than guessing.

2. MINIMUM NECESSARY CHANGES
   - Suggest only changes that materially improve the program for THIS user.
   - No speculative "nice-to-haves." No tweaks that are just stylistic preference.
   - If three changes matter and seven are possible, return three.

3. STAY IN YOUR SCOPE
   - Every task has an explicit focus (frequency, exercises, volume, reps, or RPE). Do not drift outside it.
   - Do not re-litigate decisions owned by a sibling task. Trust the pipeline.
   - If a cross-cutting concern appears, note it briefly but do not act on it.

4. CONCRETE, VERIFIABLE OUTPUT
   - Every suggestion must be specific enough for the Writer to apply without interpretation: exact exercise, exact set count, exact rep range, exact RPE, exact day.
   - Vague directives ("increase volume a bit", "make it harder") are not acceptable output.
   - If no concrete change is warranted, return "None".
'''

@dataclasses.dataclass
class PromptComponent:
    intro: str
    evaluation_criteria: List[str]
    guidelines: Dict[str, List[str]]
    action_instructions: List[str]
    response_format: str = COMMON_RESPONSE_FORMAT

    def format_for_task(self, task_type: str) -> str:
        """Format the component for a specific task type"""
        criteria = "\n".join([f"{i+1}. {c}" for i, c in enumerate(self.evaluation_criteria)])
        guidelines_text = ""
        if task_type in self.guidelines:
            guidelines = self.guidelines[task_type]
            guidelines_text = "\nGuidelines:\n" + "\n".join([f"- {g}" for g in guidelines])
        
        actions = "\n".join([f"- {a}" for a in self.action_instructions])
        
        return f"{self.intro}\n\nEvaluate whether:\n{criteria}\n{guidelines_text}\n\nIMPORTANT ACTIONS:\n{actions}\n{self.response_format}"

@dataclasses.dataclass
class CriticPromptSettings:
    role: dict[str, str]
    tasks: Optional[Dict[str, str]] = None

# All task templates for the critic
TASK_FREQUENCY_AND_SPLIT = '''
Program:
{}
User input:
{}

Focus ONLY on TRAINING FREQUENCY and SPLIT SELECTION. Do not comment on exercise selection, rep ranges, or RPE.

Evaluate:
- Sufficient frequency for each major muscle group
- Split fits the user's available days and experience level
- Split fits the user's goal (hypertrophy, strength, powerlifting, general fitness)

Reference splits:
- Hypertrophy: Full body 2x; Hybrid FB 3x; Upper/Lower 4x; Hybrid 5d (e.g. PPL+UL); PPL 6x.
- Strength/Powerlifting: Full body 3x; Upper/Lower 4x; Dedicated-lift 4-5x; movement-based; main lift + assistance.
- Specialized days (e.g. posterior-chain focus) are fine if weekly volume stays balanced across muscle groups.
- For strength/PL: prioritize the user's main lifts; main lifts can run 2-4x/week with load management; accessories less frequent.

If changes are needed, give the exact split structure (day-by-day).
'''

TASK_EXERCISE_SELECTION = '''
Program:
{}
User input:
{}

Focus ONLY on EXERCISE SELECTION. Do not comment on frequency, split, rep ranges, or RPE.

Evaluate:
- Exercises fit the user's goal, experience, and preferences
- No day exceeds 8 exercises
- Specialized days (e.g. posterior-chain focus) are balanced by complementary work elsewhere in the week

Goal-specific guidance:
- Beginner: prioritize compounds (squat/row/press); balance major muscle groups; mix free-weight and machine.
- Hypertrophy: 50-70% compound / 30-50% isolation; vary angles; ~50/50 free-weight vs machine.
- Strength/Powerlifting: exercises must complement the user's main lifts (or their preferred variants); main lifts can sit on unrelated days if it serves a purpose and recovery isn't compromised; pick accessories that target supporting muscle groups.

If changes are needed, list exactly which exercises to replace and what to replace them with (per day).
'''

TASK_SET_VOLUME = '''
Program:
{}
User input:
{}

Focus ONLY on WEEKLY SET VOLUME per movement pattern. Do not comment on frequency, exercise specifics, rep ranges, or RPE.

Steps:
1) Tally current weekly sets for: Upper Horizontal Push (chest), Upper Horizontal Pull (rows), Upper Vertical Push (overhead), Upper Vertical Pull (pull-ups/lats), Lower Anterior Chain (quads), Lower Posterior Chain (glutes/hams). Compounds may count for multiple patterns.
2) Compare to the volume guidelines in the reference data below; flag patterns above/below range.
3) Recommend concrete fixes: which day, which exercise, change set count or add/remove. Per-exercise set count must stay 2-5. For ranges, start at the LOWER end (especially for beginners). Distribute volume across days, don't concentrate on one.
- Strength/PL: ensure main lifts get sufficient volume; accessories complement without overloading.
- Other goals: balance volume across all patterns, spread across the week.

If volume is already balanced, return "None".
'''

TASK_REP_RANGES = '''
Program:
{}
User input:
{}

Focus ONLY on REP RANGES. Do not comment on frequency, split, exercise selection, or RPE.

Goal-specific rep ranges:
- Hypertrophy: compounds 5-12; isolation 8-20 (12-20 OK for cables/machines). Do NOT drop hypertrophy compounds to 1-5.
- Strength/Powerlifting: main lifts 1-6; accessories 4-8; isolation accessories 8-12.
- General fitness/beginner: compounds 8-15; isolation 10-20.
- Mixed (strength+hypertrophy): main lifts 4-8; secondary compounds 6-12; isolation 10-15.

Rules: never use AMRAP. Give exact rep ranges per exercise that needs adjustment.
'''

TASK_RPE = '''
Program:
{}
User input:
{}

Focus ONLY on RPE TARGETS. Do not comment on frequency, split, exercise selection, or rep ranges.

Guidelines:
- Isolation exercises: higher RPE, 8-10.
- Compound movements: slightly lower RPE.
- Low-stability exercises (machines, cable flies): high RPE OK (8-10).
- Always express RPE as a RANGE (e.g. 8-9, 9-10), never a single number.
- Match RPE to the user's experience level (powerlifter, bodybuilder, beginner).

Give exact RPE ranges per exercise that needs adjustment.
'''

TASK_PROGRESSION = '''
Your colleague has written the following Week {week_number} training program:
{}


The individual's original program input was:
{}


Performance data from the previous week:
{}


Focus solely on the progression strategy in the "AI Progression" field.

Evaluate whether the progression recommendations appropriately adjust for progressive overload based on the performance data. Specifically, check that:
- Adjustments are made conservatively (e.g., weight increases of 2.5-5kg for upper body or 5-10kg for lower body; or rep changes by whole numbers).
- Each set is modified using either a load change OR a rep change—but never both.
- The changes are clear and actionable, with exact numbers provided (e.g., "85kg ↑" or "10 reps ↑").

CRITERIA FOR REP VS WEIGHT ADJUSTMENTS:
- RECOMMEND WEIGHT INCREASE WHEN:
  * RPE is consistently below target range (e.g., RPE 5-6 when target is 7-8)
  * User is in middle-to-upper end of the rep range AND RPE is below target
  * Exercise is a compound movement focused on strength development

- RECOMMEND WEIGHT DECREASE WHEN:
  * RPE is consistently above target range (e.g., RPE 9-10 when target is 7-8)
  * User is below target reps, and RPE is very high or over target range

- RECOMMEND REP INCREASE WHEN:
  * User is at the LOWER END of the rep range (e.g., 6 reps when range is 6-10)
  * RPE is within target range but reps have room to increase within range
  * Exercise is isolation or hypertrophy-focused
  * Adding 1-2 reps would still keep user within the prescribed rep range



Provide concise, concrete feedback with a single adjustment per set (either rep or load), using the following format:
  - One line per set showing the performance data exactly as provided.
  - A single subsequent line with ONLY the adjustment (e.g., "85kg ↑" or "10 reps ↑").

If the progression strategy is already optimal, simply return "None" with no further text.
'''
#- RECOMMEND NO CHANGE WHEN:
#  * RPE is already at upper end of target range (8-9)
#  * User failed to complete all prescribed reps with good form
#  * Performance was inconsistent between sets

# Dictionary of specialized critic settings for different evaluation tasks
CRITIC_PROMPT_SETTINGS: dict[str, CriticPromptSettings] = {}

# Setting for Week 1 with all tasks
CRITIC_PROMPT_SETTINGS['week1'] = CriticPromptSettings(
    role={
        'role': 'system',
        'content': (
            'You are an experienced strength training coach with deep expertise in exercise science and program design.'
            'Your job is to critically evaluate the training program provided above, for the task provided'
            'Provide clear, actionable, feedback to help improve the program.'
            'If the program meets all criteria, simply return "None".'
            + COMMON_CRITIC_PRINCIPLES
            + COMMON_RESPONSE_CONSTRAINTS
        ),
    },
    tasks={
        'frequency_and_split': TASK_FREQUENCY_AND_SPLIT,
        'exercise_selection': TASK_EXERCISE_SELECTION,
        'set_volume': TASK_SET_VOLUME,
        'rep_ranges': TASK_REP_RANGES,
        'rpe': TASK_RPE,
    },
)

# Setting for Week 2+ with progression focus only
CRITIC_PROMPT_SETTINGS['progression'] = CriticPromptSettings(
    role={
        'role': 'system',
        'content': (
            'You are an experienced strength and conditioning coach with deep expertise in exercise science, program design, and progressive overload principles. '
            'Your task is to analyze the training program and previous week\'s performance data to ensure effective progression and proper autoregulation. '
            'Provide specific, actionable feedback on weight selection, rep ranges, RPE targets, and progression rates. '
            'Make precise recommendations for adjustments to optimize the program for continued progress. '
            'If the program meets all criteria for optimal progression, simply return "None".'
            + COMMON_CRITIC_PRINCIPLES
            + COMMON_RESPONSE_CONSTRAINTS
        ),
    },
    tasks={
        'progression': TASK_PROGRESSION,  # Use the single progression task
    },
)

for setting_key in ['frequency_and_split', 'exercise_selection', 'set_volume', 'rep_ranges', 'rpe']:
    if setting_key in CRITIC_PROMPT_SETTINGS:
        task_var_name = f"TASK_{setting_key.upper()}"
        task_template = locals().get(task_var_name, globals().get(task_var_name))
        CRITIC_PROMPT_SETTINGS[setting_key].tasks = {
            'frequency_and_split': TASK_FREQUENCY_AND_SPLIT,
            'exercise_selection': TASK_EXERCISE_SELECTION,
            'set_volume': TASK_SET_VOLUME,
            'rep_ranges': TASK_REP_RANGES,
            'rpe': TASK_RPE,
        }

