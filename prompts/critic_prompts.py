import dataclasses
from typing import Dict, Optional, List

# Define reusable prompt components
COMMON_RESPONSE_FORMAT = '''
IMPORTANT: 
1. Provide SPECIFIC, CONCRETE suggestions - don't just identify problems
2. For each change, specify exactly what to modify and how
3. If nothing needs improvement, return "None"
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
Your colleague has written the following training program:
{}
For an individual who provided the following input:
{}
Focus ONLY on the TRAINING FREQUENCY and SPLIT SELECTION. Do NOT comment on exercise selection, rep ranges, or RPE.
Specifically, evaluate whether:
1. The program provides sufficient frequency for each major muscle group
2. The training split effectively utilizes the user's available training days and experience level. 
3. The split is appropriate for the user's goals (strength, powerlifting, hypertrophy, etc.)

For hypertrophy-focused goals, consider these common split options:
- Full body (2x/week): Trains all muscle groups each session
- Hybrid full body (3x/week): Alternates between full body and upper/lower splits
- Upper/Lower (4x/week): Alternates between upper and lower body days
- Hybrid splits (5 days): Such as Push, Pull, Legs, Upper, Lower
- Push/Pull/Legs (6x/week): One day each for pushing movements, pulling movements, and leg exercises

For powerlifting and strength-focused goals, consider these specialized split options:
- Full-Body Split (3 Days/Week): Each session covers all three main lifts.
- Upper/Lower Split (4 Days/Week): Alternates between upper and lower body days - frequency 2-3 times a week.
- Dedicated Lifts Split (4-5 Days/Week): Each session focuses on one or two main lifts.
- Movement-based splits: Organize training around the individual's main lifts.
- Main lift + supplemental work: Each day features one main compound lift followed by related assistance work

For strength and powerlifting goals, consider the following recommendations:
- Prioritize Main Lifts: Focus on the individual's preferred main exercises—these might be traditional lifts like squat, bench press, and deadlift, or could be alternatives depending on the individual's preferences.
- Higher Frequency for Main Lifts: The individual's main lifts and their variations can be trained more than 2-4 times per week with proper load management.
- Accessory Work: Although main lifts receive higher frequency, accessory exercises should be performed less frequently.
- Tailored Adjustments: If the current training frequency or split doesn't align with the user's available training days or specific goals, adjust the plan to better match those needs.
- Specialized Training Days: It's appropriate to have specialized days (like "posterior chain focus") as long as overall weekly volume remains balanced across muscle groups/movement patterns.

IMPORTANT: If changes are needed, provide SPECIFIC, CONCRETE suggestions - specify exactly what split structure you recommend with a clear layout of training days.
If nothing needs improvement, return "None".
'''

TASK_EXERCISE_SELECTION = '''
Your colleague has written the following training program:
{}
 
The individual has provided these details:
{}
 
Focus ONLY on EXERCISE SELECTION. Do NOT comment on frequency, split structure, rep ranges, or RPE.
 
Evaluate whether:
- The chosen exercises align with the individual's goals, experience level, and personal preferences
- NOTE: If the frequency_and_split critique suggested different training splits or structure, consider how exercises would fit into that revised structure rather than the original
- Each training day should contain NO MORE THAN 8 exercises to ensure workout want be too long.

For beginners:
- Prioritize compound movements (e.g., squats, rows, presses) to maximize training efficiency.
- Maintain a balanced routine targeting all major muscle groups each week.
- Think about the mix of free-weight and machine exercises for the specific user.
- Keep workouts short and focused—quality over quantity.

For bodybuilding/hypertrophy-focused goals:
- Maintain a balanced mix of compound (50-70%) and isolation (30-50%) exercises
- Include a variety of exercises to target each muscle group from different angles
- Aim for a 50-50 mix of free-weight and machine exercises
 
For strength/powerlifting-focused goals:
- Choose exercises that directly complement the individual's main lifts or their well-suited variations
- Main lifts should be based on the individual's preferences and goals, not necessarily traditional powerlifts
- For powerlifting programs, structure each day around main compound movements followed by relevant accessories
- It's acceptable to place certain "main" lifts on seemingly unrelated days (e.g., pull-ups on leg day) if:
  1. It serves a specific purpose (e.g., back development for improved bracing on squats)
  2. It doesn't interfere with recovery for the primary focus of that day
  3. The overall weekly volume and frequency for that movement pattern is appropriate
- Identify accessory movements that target the supporting muscle groups impacting the main lifts
- For specialized days (like posterior chain focus), ensure that across the week, all muscle groups and movement patterns receive balanced attention
 
Balance considerations for specialized training days:
- When creating a specialized day (e.g., posterior chain focus), compensate with complementary work on other days
- Ensure appropriate opposing muscle group work within the week (e.g., if emphasizing hamstrings heavily, ensure adequate quad work elsewhere)
- Main lifts should be placed where recovery will be optimal for performance
 
IMPORTANT: If changes are needed, provide SPECIFIC, CONCRETE suggestions - list exactly which exercises to replace and what to replace them with.
If nothing needs improvement, return "None".
'''

TASK_SET_VOLUME = '''
Your colleague has written the following training program:
{}

The individual has provided these details:
{}

Focus ONLY on evaluating the WEEKLY SET VOLUME for each major movement pattern and suggesting concrete adjustments. Do NOT comment on frequency, split structure, exercise selection specifics, rep ranges, or RPE.

NOTE: Consider any changes suggested by previous critiques (frequency_and_split, exercise_selection) when evaluating set volume.

1) First, calculate the current weekly set volume for these major movement patterns:
   - Upper: Horizontal Push (Chest/pressing)
   - Upper: Horizontal Pull (Rows/rear back)
   - Upper: Vertical Push (Overhead/shoulders) 
   - Upper: Vertical Pull (Pull-ups/lats)
   - Lower: Anterior Chain (Quads)
   - Lower: Posterior Chain (Glutes/Hams)

2) Then, evaluate if the volume needs to be changed. Use the specific volume guidelines provided in the reference data section below.

3) IMPORTANT: Make CONCRETE ADJUSTMENTS to the program while ensuring proper volume distribution throughout the week:
   - If volume is too high: Either reduce the number of sets for specific exercises OR completely remove certain exercises. Specify exactly which exercises to modify and from which days
   - If volume is too low: Either increase the number of sets for existing exercises OR add new exercises. Specify exactly which exercises to modify or add, with specific set/rep recommendations
   - If movement pattern balance is off: Identify which patterns are overemphasized or underemphasized, then suggest specific exercises to add, remove, or modify numbers of sets to create proper balance. Specify exactly which adjustments to make on which training days
        - FOR POWERLIFTING or Strength focused goal: Ensure the main lifts are trained with sufficient volume and frequency, and that accessory work complements the main lifts without overloading them
        - FOR other goals: ensure the volume is balanced across all muscle groups and movement patterns are are spread evenly throughout the week
   - Ensure volume for each movement pattern is distributed appropriately across multiple training days rather than concentrated on a single day
   - IMPORTANT: 
        - When suggesting a range of sets (e.g., 10-12 sets per week), ALWAYS start at the lower end of the range, especially for a beginner.
        - The number of sets of an exercise should be between 2-5 sets
        - Compound exercises can be a part of the volume of multiple movement patterns

Do not just analyze - actually modify the program to create optimal training volume with proper distribution throughout the week.
If the volume distribution is already optimal, return "None".
'''

TASK_REP_RANGES = '''
Your colleague has written the following training program:
{}

The individual has provided these details:
{}

Focus ONLY on the REP RANGES. Do NOT comment on frequency, split structure, exercise selection, or RPE.

Evaluate whether the rep ranges align with the individual's PRIMARY GOAL. Use the goal-specific guidelines below.

GOAL-SPECIFIC REP RANGE GUIDELINES:

For HYPERTROPHY / BODYBUILDING goals:
- Compound movements (squat, deadlift, press, row): 5-12 reps is the ideal range. Both 6-8 and 8-12 are appropriate.
- Isolation exercises (curls, flies, raises, extensions): 8-20 reps. Higher reps (12-20) are fine for cables and machines.
- Do NOT push hypertrophy compounds down to powerlifting-style rep ranges (1-5).

For STRENGTH / POWERLIFTING goals:
- Main lifts (squat, bench, deadlift and close variants): 1-6 reps. Accessories in 4-8 range.
- Isolation accessory work: 8-12 reps.

For GENERAL FITNESS / BEGINNER goals:
- Compounds: 8-15 reps.
- Isolation: 10-20 reps.

For MIXED (strength + hypertrophy) goals:
- Main lifts: 4-8 reps.
- Secondary compounds: 6-12 reps.
- Isolation: 10-15 reps.

Additional rules:
- Do not include AMRAP (As Many Reps As Possible).
- NOTE: Consider any changes suggested by previous critiques (frequency_and_split, exercise_selection, set_volume) when evaluating rep ranges.

IMPORTANT: If changes are needed, provide SPECIFIC, CONCRETE suggestions - specify exact rep ranges for each exercise that needs adjustment.
If nothing needs improvement, return "None".
'''

TASK_RPE = '''
Your colleague has written the following training program:
{}

The individual has provided these details:
{}

Focus ONLY on the RPE (Rating of Perceived Exertion) Targets. Do NOT comment on frequency, split structure, exercise selection, or rep ranges.

Evaluate whether:
- Evaluate if the user is a powerlifter, bodybuilder, or beginner, and adjust the rep ranges accordingly.
- The RPE values are appropriate for the individual's experience level
- NOTE: Consider any changes suggested by ALL previous critiques (frequency_and_split, exercise_selection, set_volume, rep_ranges) when evaluating RPE targets
- Your RPE recommendations should be compatible with the training frequency, exercise selection, set volume, and rep ranges previously recommended

Guidelines:
- Isolation exercises should have a higher Target RPE 8–10
- Compound movements should have a slightly lower Target RPE
- Always provide the RPE in a range (e.g., 8-9, 9-10)
- Exercises that require lower stability (like machine exercises, or something like cable flies) could use a high RPE (8-10)

IMPORTANT: If changes are needed, provide SPECIFIC, CONCRETE suggestions - specify exact RPE targets for each exercise that needs adjustment.
If nothing needs improvement, return "None".
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

