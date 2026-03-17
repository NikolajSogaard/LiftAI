from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, copy_current_request_context
from flask_session import Session
import json
import os
import argparse
import tempfile
from datetime import datetime, timedelta
import uuid
import threading
import queue

from agent_system import (
    setup_llm,
    ProgramGenerator,
    Writer,
    Critic,
    Editor,
)

from prompts import (
    WriterPromptSettings,
    CriticPromptSettings,
    WRITER_PROMPT_SETTINGS,
    CRITIC_PROMPT_SETTINGS,
)

from rag_retrieval import retrieve_and_generate

app = Flask(__name__)
app.secret_key = os.urandom(24)

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
    """Load personas JSON once and cache for the process lifetime."""
    global _personas_cache
    if _personas_cache is None:
        try:
            with open('Data/personas/personas_vers2.json') as f:
                _personas_cache = json.load(f)["Personas"]
        except Exception:
            _personas_cache = {}
    return _personas_cache

def _parse_feedback_form(program: dict, form, key_prefix: str = "") -> dict:
    """Parse set/rep/RPE form data into a feedback_data dict.

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
                    'actual_rpe': form.get(f"{prefix}_ex{i}_set{j}_actual_rpe")
                })
                # Superset A2 fields (only present when exercise is a superset)
                a2_weight = form.get(f"{prefix}_ex{i}_a2set{j}_weight")
                if a2_weight is not None:
                    exercise_feedback.setdefault('a2_sets_data', []).append({
                        'weight': a2_weight,
                        'reps': form.get(f"{prefix}_ex{i}_a2set{j}_reps"),
                        'actual_rpe': form.get(f"{prefix}_ex{i}_a2set{j}_actual_rpe")
                    })
            feedback_data[day].append(exercise_feedback)
    return feedback_data

DEFAULT_CONFIG = {
    'model': 'gemini-3-flash-preview',        # Writer & Editor
    'critic_model': 'gemini-3-flash-preview', # Critic — 5 parallel tasks
    'max_tokens': 8000,
    'writer_temperature': 0.4,
    'writer_top_p': 0.9,
    'writer_prompt_settings': 'v1',
    'critic_prompt_settings': 'week1',
    'max_iterations': 1,
    'thinking_budget': None,                  # Set to e.g. 5000 if model supports ThinkingConfig
}

def get_program_generator(config=None):
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
        max_tokens=config['max_tokens'],
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
                           "target_rpe": 0, "rest": "N/A",
                           "cues": "Please try generating a new program."}]}

    except Exception as e:
        print(f"Error parsing program: {e}")
        return {"Day 1": [{"name": "Error parsing program", "sets": 0, "reps": "0",
                           "target_rpe": 0, "rest": "N/A", "cues": str(e)}]}

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
        while True:
            try:
                msg = q.get(timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("step") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'step': 'timeout', 'message': 'Generation timed out'})}\n\n"
                break
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
    session['all_programs'] = [{'week': 1, 'program': result['program']}]
    
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

def create_next_week_prompt(user_input, current_program, feedback_data, current_week, persona=None):
    prompt = f"""
    Original User Input: {user_input}
    Previous Program: {json.dumps(current_program)}
    User Feedback: {json.dumps(feedback_data)}
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
    current_program = session['raw_program']

    if 'formatted' in current_program and isinstance(current_program['formatted'], dict):
        if 'weekly_program' in current_program['formatted']:
            if isinstance(current_program, dict) and 'weekly_program' not in current_program:
                current_program['weekly_program'] = current_program['formatted']['weekly_program']
    
    current_week = session.get('current_week', 1)
    if current_week == 1:
        session['original_program_structure'] = session['program']

    current_program['week_number'] = current_week
    current_program['feedback'] = feedback_data

    next_week_input = create_next_week_prompt(
        user_input=session.get('user_input', ''),
        current_program=current_program,
        feedback_data=feedback_data,
        current_week=current_week,
        persona=session.get('persona_data') if session.get('persona') else None
    )

    if session.get('persona'):
        selected_persona = _get_personas().get(session['persona'])
        if selected_persona:
            next_week_input += f"\nTarget Persona: {selected_persona}"

    new_week = current_week + 1
    config = DEFAULT_CONFIG.copy()
    config['critic_prompt_settings'] = 'week2plus'
    config['week_number'] = new_week
    
    program_generator = get_program_generator(config)
    program_result = program_generator.create_program(user_input=next_week_input)

    if new_week > 1 and 'original_program_structure' in session:
        original_structure = session['original_program_structure']
        new_program = parse_program(program_result.get('formatted'))

        merged_program = {}
        for day, exercises in original_structure.items():
            merged_program[day] = []
            for i, exercise in enumerate(exercises):
                preserved_exercise = exercise.copy()
                if day in new_program and i < len(new_program[day]):
                    new_exercise = new_program[day][i]
                    preserved_exercise['suggestion'] = new_exercise.get('suggestion', new_exercise.get('AI Progression'))
                merged_program[day].append(preserved_exercise)

        parsed_program = merged_program
    else:
        parsed_program = parse_program(program_result.get('formatted'))

    session['program'] = parsed_program
    session['raw_program'] = program_result
    session['feedback'] = {}
    session['current_week'] = new_week

    all_programs = session.get('all_programs', [])
    all_programs.append({'week': new_week, 'program': parsed_program})
    session['all_programs'] = all_programs

    flash(f"Week {new_week} program generated successfully!")
    return redirect(url_for('index'))

# Ensure SavedPrograms directory exists
SAVED_PROGRAMS_DIR = os.path.join('Data', 'SavedPrograms')
os.makedirs(SAVED_PROGRAMS_DIR, exist_ok=True)

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
            'raw_program': session.get('raw_program', {}),
            'all_programs': session.get('all_programs', []),
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
                print(f"Error reading {fname}: {e}")
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
        
        filepath = os.path.join(SAVED_PROGRAMS_DIR, filename)
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
        
        return jsonify({'success': True, 'redirect': url_for('index')})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading program: {e}'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
