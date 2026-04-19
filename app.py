from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, copy_current_request_context
from flask_session import Session
import json
import logging
import os
import argparse
import tempfile
from datetime import datetime, timedelta
import uuid
import threading
import queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

from agent_system import (
    setup_llm,
    ProgramGenerator,
    ProgramChatbot,
    Writer,
    Critic,
    Editor,
    Analyst,
)
from agent_system.generator import ProgressionProgramGenerator

from prompts import (
    WriterPromptSettings,
    CriticPromptSettings,
    WRITER_PROMPT_SETTINGS,
    CRITIC_PROMPT_SETTINGS,
)

from rag_retrieval import retrieve_and_generate

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

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET_KEY") or os.urandom(24)

# Configure server-side sessions
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(tempfile.gettempdir(), "flask_session_files")
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)
app.config["SESSION_USE_SIGNER"] = True
Session(app) # initialize session management

# SSE message queues keyed by job_id
_generation_queues: dict[str, queue.Queue] = {}
_generation_results: dict[str, dict] = {}

_personas_cache = None

def _get_personas() -> dict:
    """Load personas JSON once and cache for the process lifetime.

    Returns
    -------
    dict
        Mapping of persona name → persona description string.
        Returns an empty dict if the file cannot be loaded.
    """
    global _personas_cache
    if _personas_cache is None:
        try:
            with open('Data/personas/personas_vers2.json') as f:
                _personas_cache = json.load(f)["Personas"]
        except Exception:
            logger.warning("Failed to load personas file", exc_info=True)
            _personas_cache = {}
    return _personas_cache

def _parse_feedback_form(program: dict, form, key_prefix: str = "") -> dict:
    """Parse raw form POST data into a structured feedback dict.

    Returns
    -------
    dict
        ``{day: [{"exercise": str, "feedback": str}]}`` mapping.

    Notes
    -----
    key_prefix is prepended to the day key, e.g. pass "{week}_" for next_week.
    """
    feedback_data = {}
    for day, exercises in program.items():
        day_key = day.replace(' ', '')
        prefix = f"{key_prefix}{day_key}" if key_prefix else day_key
        feedback_data[day] = []
        for i, exercise in enumerate(exercises):
            exercise_feedback = {
                'name': exercise['name'],
                'sets_data': [],
                'overall_feedback': form.get(f"{prefix}_ex{i}_feedback", "")
            }
            for j in range(exercise.get('sets', 0)):
                exercise_feedback['sets_data'].append({
                    'weight': form.get(f"{prefix}_ex{i}_set{j}_weight"),
                    'reps': form.get(f"{prefix}_ex{i}_set{j}_reps"),
                    'actual_rir': form.get(f"{prefix}_ex{i}_set{j}_actual_rir")
                })
                # Superset A2 fields (only present when exercise is a superset)
                a2_weight = form.get(f"{prefix}_ex{i}_a2set{j}_weight")
                if a2_weight is not None:
                    exercise_feedback.setdefault('a2_sets_data', []).append({
                        'weight': a2_weight,
                        'reps': form.get(f"{prefix}_ex{i}_a2set{j}_reps"),
                        'actual_rir': form.get(f"{prefix}_ex{i}_a2set{j}_actual_rir")
                    })
            feedback_data[day].append(exercise_feedback)
    return feedback_data

DEFAULT_CONFIG = {
    'model': DEFAULT_MODEL,
    'critic_model': DEFAULT_CRITIC_MODEL,
    'max_tokens': DEFAULT_MAX_TOKENS,
    'critic_max_tokens': 1500,                # Critic feedback is short; cap keeps outputs terse
    'progression_writer_max_tokens': 2000,    # Progression drafts are a few lines per exercise
    'writer_temperature': DEFAULT_WRITER_TEMPERATURE,
    'writer_top_p': DEFAULT_WRITER_TOP_P,
    'writer_prompt_settings': 'v1',
    'critic_prompt_settings': 'week1',
    'max_iterations': DEFAULT_MAX_ITERATIONS,
    'thinking_budget': None,                  # Set to e.g. 5000 if model supports ThinkingConfig
}

def get_program_generator(config: dict | None = None) -> "ProgramGenerator":
    """Build a ProgramGenerator from the given (or default) config."""
    if config is None:
        config = DEFAULT_CONFIG
    
    week_number = config.get('week_number', 1)
    is_revision = config.get('is_revision', False)
    
    if week_number > 1:
        writer_type = "progression"
    elif is_revision:
        writer_type = "revision"
    else:
        writer_type = "initial"

    writer_prompt_settings = WRITER_PROMPT_SETTINGS[writer_type]
    critic_setting_key = 'progression' if week_number > 1 else 'week1'
    critic_prompt_settings = CRITIC_PROMPT_SETTINGS[critic_setting_key]
    
    # LLMs
    llm_writer = setup_llm(
        model=config['model'],
        respond_as_json=True,
        temperature=config['writer_temperature'],
        top_p=config['writer_top_p'],
        thinking_budget=config.get('thinking_budget'),
    )
    llm_critic = setup_llm(
        model=config.get('critic_model', config['model']),
        max_tokens=config.get('critic_max_tokens', config['max_tokens']),
        respond_as_json=False,
    )

    # Resolve revision task
    task_revision = writer_prompt_settings.task_revision
    if not task_revision and 'revision' in WRITER_PROMPT_SETTINGS:
        task_revision = WRITER_PROMPT_SETTINGS['revision'].task_revision
    
    # Agents
    writer = Writer(
        model=llm_writer,
        role=writer_prompt_settings.role,
        structure=writer_prompt_settings.structure,
        task=writer_prompt_settings.task,
        task_revision=task_revision,
        task_progression=getattr(writer_prompt_settings, 'task_progression', None),
        writer_type=writer_type,
        retrieval_fn=retrieve_and_generate
    )
    critic = Critic(
        model=llm_critic,
        role=critic_prompt_settings.role,
        tasks=getattr(critic_prompt_settings, 'tasks', None),
        retrieval_fn=retrieve_and_generate
    )
    editor = Editor()

    return ProgramGenerator(
        writer=writer, critic=critic, editor=editor,
        max_iterations=config.get('max_iterations', 2)
    )

def get_progression_generator(config: dict, writer_type: str = "progression") -> ProgressionProgramGenerator:
    """Build a ProgressionProgramGenerator for week 2+ generation."""
    writer_settings_key = writer_type
    writer_prompt_settings = WRITER_PROMPT_SETTINGS.get(writer_settings_key, WRITER_PROMPT_SETTINGS['progression'])

    critic_setting_key = 'week1' if writer_type == 'new_block' else 'progression'
    critic_prompt_settings = CRITIC_PROMPT_SETTINGS[critic_setting_key]

    # Progression writer emits a few lines per exercise; full 8k cap is overkill.
    writer_max = (
        config.get('progression_writer_max_tokens', config['max_tokens'])
        if writer_type == 'progression'
        else config['max_tokens']
    )
    llm_writer = setup_llm(
        model=config['model'],
        max_tokens=writer_max,
        respond_as_json=True,
        temperature=config['writer_temperature'],
        top_p=config['writer_top_p'],
        thinking_budget=config.get('thinking_budget'),
    )
    llm_critic = setup_llm(
        model=config.get('critic_model', config['model']),
        max_tokens=config.get('critic_max_tokens', config['max_tokens']),
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


def _build_block_summary(all_programs: list[dict], mesocycle: int) -> dict:
    """Build a condensed summary of a completed mesocycle."""
    block_weeks = [w for w in all_programs if w.get("mesocycle") == mesocycle]
    if not block_weeks:
        return {}

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
                        "start_weight": best_weight, "start_reps": best_reps,
                        "end_weight": best_weight, "end_reps": best_reps,
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


def parse_program(program_output):
    """Extract weekly_program from various nested output formats."""
    try:
        if isinstance(program_output, str):
            try:
                program_output = json.loads(program_output)
            except json.JSONDecodeError:
                pass

        if isinstance(program_output, dict):
            # Direct weekly_program
            if 'weekly_program' in program_output:
                return program_output['weekly_program']

            # Nested in formatted field
            if 'formatted' in program_output:
                formatted = program_output['formatted']
                if isinstance(formatted, str):
                    try:
                        parsed = json.loads(formatted)
                        return parsed.get('weekly_program', parsed)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(formatted, dict):
                    return formatted.get('weekly_program', formatted)

            # Embedded in message field
            if 'message' in program_output and isinstance(program_output['message'], str):
                msg = program_output['message']
                try:
                    if "```json" in msg:
                        msg = msg.split("```json", 1)[1].split("```", 1)[0].strip()
                    if msg.strip().startswith("{") and msg.strip().endswith("}"):
                        parsed = json.loads(msg)
                        if isinstance(parsed, dict):
                            return parsed.get('weekly_program', parsed)
                except (json.JSONDecodeError, IndexError):
                    pass

            # Fallback to draft field
            if 'draft' in program_output:
                draft = program_output['draft']
                if isinstance(draft, dict):
                    return draft.get('weekly_program', draft)

        # Nothing found
        return {"Day 1": [{"name": "No program data found", "sets": 0, "reps": "0",
                           "target_rir": 0, "rest": "N/A",
                           "cues": "Please try generating a new program."}]}

    except Exception as e:
        logger.exception("Error parsing program")
        return {"Day 1": [{"name": "Error parsing program", "sets": 0, "reps": "0",
                           "target_rir": 0, "rest": "N/A", "cues": str(e)}]}

@app.route('/')
def index():
    if 'program' not in session:
        return redirect(url_for('generate_program'))
    programs = session.get('all_programs', [])
    current_week = session.get('current_week', 1)
    return render_template('index.html', programs=programs, current_week=current_week)

@app.route('/generate', methods=['GET', 'POST'])
def generate_program():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        persona = request.form.get('persona', '')

        if not user_input:
            user_input = "Generate a strength training program for the selected persona."
        elif len(user_input) > MAX_USER_INPUT_CHARS:
            user_input = user_input[:MAX_USER_INPUT_CHARS]
            logger.warning("/generate: user_input truncated to %d chars", MAX_USER_INPUT_CHARS)

        # Validate persona against known personas
        if persona:
            known_personas = _get_personas()
            if known_personas and persona not in known_personas:
                logger.warning("/generate: unknown persona %r ignored", persona)
                persona = ''

        config = DEFAULT_CONFIG.copy()
        
        program_input = user_input
        if persona:
            selected = _get_personas().get(persona)
            if selected:
                program_input = f"{user_input}\nTarget Persona: {selected}"
        
        # Create a job id and queue for SSE streaming
        job_id = uuid.uuid4().hex[:12]
        q = queue.Queue()
        _generation_queues[job_id] = q

        @copy_current_request_context
        def _run_generation():
            try:
                q.put({"step": "writer", "message": "Setting up agents..."})
                program_generator = get_program_generator(config)
                program_generator.on_status = lambda msg: q.put(msg)
                
                program_result = program_generator.create_program(user_input=program_input)
                
                parsed_program = parse_program(program_result.get('formatted'))
                
                # Store result server-side for the completion endpoint to pick up
                _generation_results[job_id] = {
                    'program': parsed_program,
                    'raw_program': program_result,
                    'user_input': user_input,
                    'persona': persona,
                }
                
                q.put({"step": "done", "message": "Program generated successfully!", "job_id": job_id})
            except Exception as e:
                logger.exception("Program generation failed")
                q.put({"step": "error", "message": str(e)})
            finally:
                _generation_queues.pop(job_id, None)

        thread = threading.Thread(target=_run_generation, daemon=True)
        thread.start()
        
        return jsonify({"job_id": job_id})
    
    return render_template('generate.html')


@app.route('/generate/stream/<job_id>')
def generate_stream(job_id):
    def _event_stream():
        q = _generation_queues.get(job_id)
        if not q:
            yield f"data: {json.dumps({'step': 'error', 'message': 'Job not found'})}\n\n"
            return
        elapsed = 0
        poll_interval = 15  # seconds between keepalive pings
        while elapsed < QUEUE_TIMEOUT:
            try:
                msg = q.get(timeout=poll_interval)
                elapsed = 0  # reset on real message
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("step") in ("done", "error"):
                    return
            except queue.Empty:
                elapsed += poll_interval
                yield ": keepalive\n\n"  # SSE comment — keeps connection alive
        yield f"data: {json.dumps({'step': 'timeout', 'message': 'Generation timed out'})}\n\n"
    return Response(_event_stream(), mimetype='text/event-stream')


@app.route('/generate/complete/<job_id>')
def generation_complete(job_id):
    """Load generation result into the session and redirect to index."""
    result = _generation_results.pop(job_id, None)
    if not result:
        flash("Generation result expired or not found.")
        return redirect(url_for('generate_program'))
    
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
    
    return redirect(url_for('index'))


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if 'program' not in session:
        flash("No active program to provide feedback for.")
        return redirect(url_for('index'))
    program = session.get('program', {})
    feedback_data = _parse_feedback_form(program, request.form)
    session['feedback'] = feedback_data
    flash("Feedback submitted successfully!")
    return redirect(url_for('index'))

def create_next_week_prompt(program: dict, feedback: dict, user_input: str = "", current_week: int = 1, persona=None) -> str:
    """Build the user-input string for a week-N+1 program generation request.

    Parameters
    ----------
    program:
        The current week's formatted program dict.
    feedback:
        Structured feedback dict from ``_parse_feedback_form``.

    Returns
    -------
    str
        A prompt string ready to pass to ``ProgramGenerator.create_program``.
    """
    prompt = f"""
    Original User Input: {user_input}
    Previous Program: {json.dumps(program)}
    User Feedback: {json.dumps(feedback)}
    Please generate Week {current_week + 1} program considering the feedback provided.
    Autoregulate the training loads based on the actual performance data.
    """
    if persona:
        prompt += f"\nTarget Persona: {persona}"
    return prompt

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
                new_week = current_week + 1
                new_mesocycle = mesocycle
                new_week_in_meso = week_in_meso  # deload doesn't advance mesocycle position
            else:
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

# Ensure SavedPrograms directory exists
SAVED_PROGRAMS_DIR = os.path.join('Data', 'SavedPrograms')
os.makedirs(SAVED_PROGRAMS_DIR, exist_ok=True)
_SAFE_PROGRAMS_DIR = os.path.realpath(SAVED_PROGRAMS_DIR)

@app.route('/save_program', methods=['POST'])
def save_program():
    if 'program' not in session:
        return jsonify({'success': False, 'message': 'No active program to save'})
    
    try:
        program_name = request.form.get('program_name', '') or f"Program_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        safe_name = "_".join(
            "".join(c for c in program_name if c.isalnum() or c in ' _-').strip().split()
        )
        filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.json"
        filepath = os.path.join(SAVED_PROGRAMS_DIR, filename)
        
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
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({'success': True, 'message': f'Program saved as {program_name}'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error saving program: {str(e)}'})

@app.route('/list_saved_programs', methods=['GET'])
def list_saved_programs():
    try:
        programs = []
        for fname in os.listdir(SAVED_PROGRAMS_DIR):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(SAVED_PROGRAMS_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    programs.append({
                        'filename': fname,
                        'name': data.get('program_name', fname),
                        'date': data.get('date_saved', ''),
                        'weeks': len(data.get('all_programs', [])),
                        'current_week': data.get('current_week', 1)
                    })
            except Exception as e:
                logger.warning("Error reading saved program %s: %s", fname, e)
        programs.sort(key=lambda x: x.get('date', ''), reverse=True)
        return jsonify({'success': True, 'programs': programs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error listing programs: {e}'})

@app.route('/load_program', methods=['POST'])
def load_program():
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'No program selected'})

        filepath = os.path.realpath(os.path.join(SAVED_PROGRAMS_DIR, filename))
        if not filepath.startswith(_SAFE_PROGRAMS_DIR + os.sep):
            return jsonify({'success': False, 'message': 'Invalid filename'})

        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Program file not found'})
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
        
        return jsonify({'success': True, 'redirect': url_for('index')})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading program: {e}'})

_chatbot: ProgramChatbot | None = None

def _get_chatbot() -> ProgramChatbot:
    global _chatbot
    if _chatbot is None:
        from agent_system.setup_api import _get_client
        client = _get_client()
        _chatbot = ProgramChatbot(model_name=DEFAULT_CONFIG['model'], client=client)
    return _chatbot


@app.route('/chat', methods=['POST'])
def chat():
    if 'program' not in session:
        return jsonify({'error': 'No active program. Generate a program first.'}), 400

    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message.'}), 400

    # Frontend sends back the current (possibly already edited) program
    program = data.get('program') or session.get('program', {})
    history = data.get('history', [])

    try:
        chatbot = _get_chatbot()
        result = chatbot.chat(message=message, program=program, history=history)
    except Exception as e:
        logger.exception("Chat request failed")
        return jsonify({'error': str(e)}), 500

    # Persist any edits back to the session
    if result.get('updated_program'):
        session['program'] = result['updated_program']
        # Keep all_programs in sync — update the current week entry
        all_programs = session.get('all_programs', [])
        current_week = session.get('current_week', 1)
        for wp in all_programs:
            if wp.get('week') == current_week:
                wp['program'] = result['updated_program']
                break
        session['all_programs'] = all_programs

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
