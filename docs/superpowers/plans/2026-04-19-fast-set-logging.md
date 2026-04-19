# Fast Set-Logging UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3-typed-inputs-per-set UI with a phone-first "prefilled pill + one-tap ✓ confirm" logger that accepts overrides, auto-advances with a pulse, and persists each confirmed set immediately.

**Architecture:** Template swaps the per-set inputs in [templates/index.html](../../../templates/index.html) for a row of pills (kg / reps / rir) plus a ✓ button; hidden inputs preserve the existing form-POST contract to `/next_week`. A new `POST /log_set` endpoint upserts each confirmed set into `session['set_log']`, which is read back at render time to prefill kg for later weeks. All JS/CSS is inline in the template; no build step.

**Tech Stack:** Flask + Jinja2, Flask-Session (filesystem), vanilla JS/CSS inline in `index.html`, pytest for backend.

**Out of scope:** Superset A2 inputs (the dynamic `_a2set{j}_*` fields built by JS around [templates/index.html:1303-1312](../../../templates/index.html#L1303-L1312)) — follow-up plan. Gemini 3 sampling defaults — separate experiment.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| [templates/index.html](../../../templates/index.html) | Set-row markup, pill CSS, client JS for prefill / confirm / undo / pulse | Modify (lines 867-896 + inline `<style>` + inline `<script>`) |
| [app.py](../../../app.py) | New `/log_set` endpoint; extend `/` route to pass `set_log` to template | Modify |
| [tests/test_log_set.py](../../../tests/test_log_set.py) | Pytest coverage for `/log_set` endpoint | Create |

---

## Task 1: Backend — `POST /log_set` endpoint (TDD)

**Files:**
- Create: `tests/test_log_set.py`
- Modify: `app.py` (add route near existing routes, ~line 393)

- [ ] **Step 1: Write the failing test**

Create `tests/test_log_set.py`:

```python
"""Tests for POST /log_set endpoint — per-set persistence."""
import json
import pytest

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def _seed_session(client):
    """Give the client a minimal session so /log_set has something to write to."""
    with client.session_transaction() as sess:
        sess['program'] = {'Day 1': [{'name': 'Bench', 'sets': 3, 'reps': '8-12', 'target_rir': '2'}]}
        sess['current_week'] = 1


def test_log_set_persists_values(client):
    _seed_session(client)
    resp = client.post('/log_set', json={
        'week': 1, 'day': 'Day 1', 'exercise_index': 0, 'set_index': 0,
        'weight': '80', 'reps': '10', 'actual_rir': '2',
    })
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True}
    with client.session_transaction() as sess:
        key = '1|Day 1|0|0'
        assert sess['set_log'][key] == {
            'weight': '80', 'reps': '10', 'actual_rir': '2',
        }


def test_log_set_upserts_same_key(client):
    _seed_session(client)
    client.post('/log_set', json={
        'week': 1, 'day': 'Day 1', 'exercise_index': 0, 'set_index': 0,
        'weight': '80', 'reps': '10', 'actual_rir': '2',
    })
    client.post('/log_set', json={
        'week': 1, 'day': 'Day 1', 'exercise_index': 0, 'set_index': 0,
        'weight': '82.5', 'reps': '9', 'actual_rir': '1',
    })
    with client.session_transaction() as sess:
        assert sess['set_log']['1|Day 1|0|0']['weight'] == '82.5'


def test_log_set_rejects_missing_fields(client):
    _seed_session(client)
    resp = client.post('/log_set', json={'week': 1, 'day': 'Day 1'})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_log_set.py -v`
Expected: FAIL — `404 Not Found` (endpoint doesn't exist yet).

- [ ] **Step 3: Add the endpoint to `app.py`**

Insert this route immediately after the `index()` route at `app.py:393`:

```python
@app.route('/log_set', methods=['POST'])
def log_set():
    """Persist a single confirmed set to session['set_log']."""
    data = request.get_json(silent=True) or {}
    required = ('week', 'day', 'exercise_index', 'set_index')
    if not all(k in data for k in required):
        return jsonify({'ok': False, 'error': 'missing fields'}), 400

    key = f"{data['week']}|{data['day']}|{data['exercise_index']}|{data['set_index']}"
    log = session.get('set_log', {})
    log[key] = {
        'weight': data.get('weight'),
        'reps': data.get('reps'),
        'actual_rir': data.get('actual_rir'),
    }
    session['set_log'] = log
    return jsonify({'ok': True})
```

Confirm `jsonify` and `session` are already imported at the top of `app.py` (they are — grep to verify if unsure).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_log_set.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_log_set.py app.py
git commit -m "feat: add /log_set endpoint for per-set persistence"
```

---

## Task 2: Backend — pass `set_log` to the index template

**Files:**
- Modify: `app.py:387-393`

- [ ] **Step 1: Extend the index route**

Replace the body of `def index()` at `app.py:387-393`:

```python
@app.route('/')
def index():
    if 'program' not in session:
        return redirect(url_for('generate_program'))
    programs = session.get('all_programs', [])
    current_week = session.get('current_week', 1)
    set_log = session.get('set_log', {})
    return render_template(
        'index.html',
        programs=programs,
        current_week=current_week,
        set_log=set_log,
    )
```

- [ ] **Step 2: Sanity check route still works**

Run: `python app.py` and load `http://localhost:5000/` (after generating a program).
Expected: page renders, no template errors. `set_log` is an empty dict on first load — no visible change yet.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: pass set_log from session into index template"
```

---

## Task 3: Template — pill markup + hidden inputs

**Files:**
- Modify: `templates/index.html:866-896`

- [ ] **Step 1: Replace the `<div class="sets-section">…</div>` block**

Replace lines 866-896 (the entire `<!-- Per-set inputs -->` section) with:

```html
              <!-- Per-set inputs (pill UI + hidden form inputs) -->
              <div class="sets-section">
                {% set day_key = day.replace(' ', '') %}
                {% for i in range(exercise.sets|default(3, true)) %}
                  {% set prior_key = (week_program.week - 1) ~ '|' ~ day ~ '|' ~ exercise_index ~ '|' ~ i %}
                  {% set prior_weight = set_log.get(prior_key, {}).get('weight', '') if set_log else '' %}
                  {% set is_editable = (week_program.week == current_week) %}
                  <div class="set-row"
                       data-week="{{ week_program.week }}"
                       data-day="{{ day }}"
                       data-exercise-index="{{ exercise_index }}"
                       data-set-index="{{ i }}"
                       data-reps-target="{{ exercise.reps|default('', true) }}"
                       data-rir-target="{{ exercise.target_rir|default('', true) }}"
                       data-prior-weight="{{ prior_weight }}">
                    <span class="set-label">S{{ i+1 }}</span>
                    <div class="set-fields-row">
                      <div class="set-pill" data-field="weight">
                        <span class="pill-value" data-placeholder="kg">{{ prior_weight }}</span>
                      </div>
                      <div class="set-pill" data-field="reps">
                        <span class="pill-value" data-placeholder="reps"></span>
                      </div>
                      <div class="set-pill" data-field="actual_rir">
                        <span class="pill-value" data-placeholder="RIR"></span>
                      </div>
                      <button type="button" class="set-confirm" aria-label="Confirm set"
                              {% if not is_editable %}disabled{% endif %}>✓</button>
                    </div>

                    <!-- Hidden inputs keep the existing /next_week form contract intact -->
                    <input type="hidden"
                           name="{{ week_program.week }}_{{ day_key }}_ex{{ exercise_index }}_set{{ i }}_weight"
                           value="{{ prior_weight }}">
                    <input type="hidden"
                           name="{{ week_program.week }}_{{ day_key }}_ex{{ exercise_index }}_set{{ i }}_reps"
                           value="">
                    <input type="hidden"
                           name="{{ week_program.week }}_{{ day_key }}_ex{{ exercise_index }}_set{{ i }}_actual_rir"
                           value="">
                  </div>
                {% endfor %}
              </div>
```

Notes for the implementer:
- `data-reps-target` / `data-rir-target` carry the prescription strings (e.g. `"8-12"`, `"2-3"`). Client JS parses them in Task 5.
- `data-prior-weight` carries the kg value logged during the previous week's same slot (empty for week 1).
- Hidden inputs keep the exact name convention read by `_parse_feedback_form()` at [app.py:110-114](../../../app.py#L110-L114) — `/next_week` keeps working with zero changes.
- `pill-value` is empty for reps/rir initially. Task 5's JS will populate via the midpoint rule after DOMContentLoaded.

- [ ] **Step 2: Load the page and confirm no Jinja errors**

Run: `python app.py`; open `/` with a generated program.
Expected: rows render with empty reps/rir pills and a ✓ button per row. No interactivity yet.

- [ ] **Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: replace per-set inputs with pill markup + hidden inputs"
```

---

## Task 4: CSS — pill / done / pulse styles

**Files:**
- Modify: `templates/index.html` inline `<style>` block (find the existing `.set-input` / `.set-field-group` rules, add new rules nearby)

- [ ] **Step 1: Add the pill + state styles**

Find the existing `.set-input` rule (around [templates/index.html:450-465](../../../templates/index.html#L450-L465)) and insert these rules after it:

```css
    /* Fast-log pill UI */
    .set-fields-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 52px;
      gap: 6px;
      align-items: stretch;
    }
    .set-pill {
      background: var(--surface2, #1c1c1c);
      border: 1px dashed var(--line, #2a2a2a);
      border-radius: 8px;
      min-height: 44px;   /* thumb-friendly */
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'DM Mono', monospace;
      font-size: 15px;
      color: var(--muted, #888884);
      cursor: pointer;
      user-select: none;
      -webkit-tap-highlight-color: transparent;
    }
    .set-pill .pill-value:empty::before {
      content: attr(data-placeholder);
      color: var(--muted, #888884);
      opacity: 0.55;
      font-size: 12px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .set-pill .pill-value:not(:empty) {
      color: var(--muted, #888884);   /* ghost state before confirm */
    }
    .set-pill input.pill-input {
      width: 100%;
      height: 44px;
      background: var(--bg, #0D0D0D);
      border: 1px solid var(--orange, #FF6B2B);
      color: var(--blue, #38BDF8);
      font-family: 'DM Mono', monospace;
      font-size: 16px;     /* >=16px prevents iOS auto-zoom */
      text-align: center;
      border-radius: 8px;
      outline: none;
    }
    .set-confirm {
      background: var(--orange, #FF6B2B);
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 20px;
      font-weight: 700;
      min-height: 44px;
      cursor: pointer;
    }
    .set-confirm:disabled {
      background: var(--surface2, #1c1c1c);
      color: var(--muted, #888884);
      cursor: not-allowed;
    }
    /* Done state: row turns solid, pills read as blue numbers, ✓ turns green-outlined */
    .set-row.done .set-pill {
      border-style: solid;
      background: #111;
    }
    .set-row.done .set-pill .pill-value {
      color: var(--blue, #38BDF8);
    }
    .set-row.done .set-confirm {
      background: transparent;
      color: #4ade80;
      border: 1px solid #4ade80;
    }
    /* Pulse nudges the next un-confirmed ✓ */
    @keyframes set-pulse {
      0%   { box-shadow: 0 0 0 0 rgba(255, 107, 43, 0.55); }
      70%  { box-shadow: 0 0 0 10px rgba(255, 107, 43, 0); }
      100% { box-shadow: 0 0 0 0 rgba(255, 107, 43, 0); }
    }
    .set-confirm.pulse {
      animation: set-pulse 1.4s ease-out 2;
    }
```

If `--muted`, `--surface2`, `--line`, or `--bg` aren't declared as CSS variables in `:root`, fall back to the literal hex values shown (already included as the second arg to `var()`).

- [ ] **Step 2: Visual check**

Reload `/`. Empty pills show `KG` / `REPS` / `RIR` placeholder labels in small muted caps. The ✓ button is orange. Row height is comfortable for thumbs.

- [ ] **Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: add pill / done / pulse CSS for fast set logging"
```

---

## Task 5: JS — midpoint prefill, pill-to-input swap, confirm, undo, pulse

**Files:**
- Modify: `templates/index.html` inline `<script>` block (add a self-contained block near the end of the script section, before `</script>`)

- [ ] **Step 1: Add the client script**

Append this IIFE to the inline `<script>` block (just before its closing `</script>`):

```javascript
  /* ---- Fast set-logging pill interactions ---- */
  (function () {
    function midpoint(target, { integer }) {
      if (!target) return '';
      const s = String(target).trim();
      // Range like "8-12" or "2-3"
      const m = s.match(/^(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)$/);
      if (m) {
        const mid = (parseFloat(m[1]) + parseFloat(m[2])) / 2;
        if (integer) return String(Math.round(mid));
        return String(Math.round(mid * 2) / 2);   // 0.5 resolution
      }
      // Single number like "10" or "2"
      if (/^\d+(?:\.\d+)?$/.test(s)) return s;
      return '';
    }

    function hiddenInputFor(row, field) {
      const name = `${row.dataset.week}_${row.dataset.day.replace(/ /g,'')}_ex${row.dataset.exerciseIndex}_set${row.dataset.setIndex}_${field}`;
      return row.querySelector(`input[type=hidden][name="${name}"]`);
    }

    function updateConfirmEnabled(row) {
      const btn = row.querySelector('.set-confirm');
      if (!btn || btn.hasAttribute('disabled') && row.dataset.readonly === '1') return;
      const kgVal = row.querySelector('.set-pill[data-field="weight"] .pill-value').textContent.trim();
      const isReadonly = row.dataset.readonly === '1';
      btn.disabled = isReadonly || !kgVal;
    }

    function postLogSet(row) {
      const payload = {
        week: Number(row.dataset.week),
        day: row.dataset.day,
        exercise_index: Number(row.dataset.exerciseIndex),
        set_index: Number(row.dataset.setIndex),
        weight: hiddenInputFor(row, 'weight').value,
        reps: hiddenInputFor(row, 'reps').value,
        actual_rir: hiddenInputFor(row, 'actual_rir').value,
      };
      fetch('/log_set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).catch(() => { /* fire-and-forget; form POST still works as fallback */ });
    }

    function commitPill(pill, value) {
      const valSpan = pill.querySelector('.pill-value');
      valSpan.textContent = value;
      const row = pill.closest('.set-row');
      const field = pill.dataset.field;
      hiddenInputFor(row, field).value = value;
      updateConfirmEnabled(row);
    }

    function openEditor(pill) {
      if (pill.querySelector('input.pill-input')) return;
      const row = pill.closest('.set-row');
      if (row.dataset.readonly === '1') return;
      const valSpan = pill.querySelector('.pill-value');
      const current = valSpan.textContent.trim();
      const field = pill.dataset.field;
      const step = (field === 'reps') ? '1' : '0.5';
      valSpan.style.display = 'none';
      const input = document.createElement('input');
      input.type = 'number';
      input.inputMode = 'decimal';
      input.step = step;
      input.className = 'pill-input';
      input.value = current;
      pill.appendChild(input);
      input.focus();
      input.select();
      const finish = () => {
        const v = input.value.trim();
        input.remove();
        valSpan.style.display = '';
        commitPill(pill, v);
        // If this pill belonged to a confirmed row, mark row dirty again
        if (row.classList.contains('done')) {
          row.classList.remove('done');
          row.querySelector('.set-confirm').disabled = !hiddenInputFor(row, 'weight').value;
        }
      };
      input.addEventListener('blur', finish, { once: true });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      });
    }

    function confirmRow(row) {
      if (row.classList.contains('done')) {
        // Undo
        row.classList.remove('done');
        updateConfirmEnabled(row);
        return;
      }
      // Ensure hidden inputs are populated from current pill values
      ['weight', 'reps', 'actual_rir'].forEach((f) => {
        const v = row.querySelector(`.set-pill[data-field="${f}"] .pill-value`).textContent.trim();
        hiddenInputFor(row, f).value = v;
      });
      row.classList.add('done');
      postLogSet(row);
      // Pulse the next unconfirmed ✓ in this exercise
      let next = row.nextElementSibling;
      while (next && (!next.classList.contains('set-row') || next.classList.contains('done'))) {
        next = next.nextElementSibling;
      }
      if (next) {
        const nextBtn = next.querySelector('.set-confirm');
        if (nextBtn && !nextBtn.disabled) {
          nextBtn.classList.remove('pulse');
          void nextBtn.offsetWidth;   // restart animation
          nextBtn.classList.add('pulse');
          setTimeout(() => nextBtn.classList.remove('pulse'), 3000);
        }
      }
    }

    function initRow(row) {
      const btn = row.querySelector('.set-confirm');
      // Template renders ✓ with `disabled` attribute on past/future weeks — mirror to the row.
      if (btn && btn.hasAttribute('disabled')) {
        row.dataset.readonly = '1';
      }
      // Midpoint prefill for reps/rir (kg handled by server-side prior-week lookup)
      const repsPill = row.querySelector('.set-pill[data-field="reps"]');
      const rirPill  = row.querySelector('.set-pill[data-field="actual_rir"]');
      const repsMid = midpoint(row.dataset.repsTarget, { integer: true });
      const rirMid  = midpoint(row.dataset.rirTarget,  { integer: false });
      if (repsMid) commitPill(repsPill, repsMid);
      if (rirMid)  commitPill(rirPill,  rirMid);

      // Kg hidden input already carries prior_weight from Jinja; sync confirm button state.
      updateConfirmEnabled(row);

      // Bind handlers
      row.querySelectorAll('.set-pill').forEach((pill) => {
        pill.addEventListener('click', (e) => {
          if (row.dataset.readonly === '1') return;
          if (e.target.tagName === 'INPUT') return;   // already editing
          openEditor(pill);
        });
      });
      if (btn) {
        btn.addEventListener('click', () => {
          if (btn.disabled) return;
          confirmRow(row);
        });
      }
    }

    function initAll() {
      document.querySelectorAll('.set-row').forEach(initRow);
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initAll);
    } else {
      initAll();
    }
  })();
```

- [ ] **Step 2: Manual smoke test**

1. Reload `/` with a generated program.
2. For an on-plan exercise (e.g. reps `"8-12"`, RIR `"2-3"`): reps pill reads `10`, RIR pill reads `2.5`. Kg pill shows the `KG` placeholder. ✓ is disabled.
3. Tap kg → numeric input appears. Type `80`. Blur. Pill reads `80`. ✓ enables.
4. Tap ✓ → row turns `done` (solid borders, blue numbers, green ✓ outline). Next row's ✓ pulses. Network panel shows `POST /log_set` 200.
5. Tap any pill on a `done` row → editor opens, row un-dones, edit commits. Tap ✓ again → re-confirms.
6. Tap a `done` ✓ directly → row un-dones without opening any pill editor.
7. Scroll to a past/future week's card: pills are non-interactive, ✓ disabled, hidden inputs still present.

- [ ] **Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: add pill interaction JS for fast set logging"
```

---

## Task 6: End-to-end form submission regression check

**Files:** none modified — verification only.

- [ ] **Step 1: Run one persona end-to-end**

1. `python app.py`, visit `/generate`, pick a persona from `Data/personas/personas_vers2.json`, generate week 1.
2. On the week 1 view, confirm 2-3 sets using the ✓ flow; manually edit one set's kg to an off-plan value; leave one exercise entirely un-confirmed (to test the fallback path).
3. Fill the week-end feedback text area for one exercise.
4. Submit the end-of-week form (whichever button triggers `/next_week`).

- [ ] **Step 2: Inspect parsed feedback**

Add a temporary `print(feedback_data)` call inside `_parse_feedback_form` at [app.py:124](../../../app.py#L124) (right before the `return`). Re-run step 1. Check stdout for the following:

- Confirmed sets: `sets_data[i]` contains the values that appeared in the pills.
- Edited set: the overridden value flows through (not the original prefill).
- Un-confirmed exercise: `sets_data[i]` contains empty strings for `weight` / `reps` / `actual_rir` (no regression — same as before this change when the user didn't type anything).

Remove the temporary `print` before committing.

- [ ] **Step 3: Week-2 kg prefill check**

After the submission above, the page should re-render showing week 2. For the confirmed exercises, the kg pills on week 2 should be prefilled with the values you logged in week 1. For the un-confirmed exercise, kg pills are empty.

- [ ] **Step 4: Run the backend test suite**

Run: `pytest tests/ -v`
Expected: all existing tests pass; the 3 new `test_log_set` tests pass.

- [ ] **Step 5: Commit any small fixes surfaced during verification**

(If this step is a no-op, skip the commit.)

---

## Deferred follow-ups (do not implement in this plan)

- **Superset A2 inputs** — the JS at [templates/index.html:1303-1312](../../../templates/index.html#L1303-L1312) builds A2 inputs on the fly and they use the `_a2set{j}_*` naming consumed at [app.py:117-123](../../../app.py#L117-L123). Extending pill UI to A2 requires touching that dynamic construction and is out of scope.
- **Gemini 3 sampling A/B** — `DEFAULT_WRITER_TEMPERATURE` / `DEFAULT_WRITER_TOP_P` in [config.py:9-10](../../../config.py#L9-L10) vs Gemini 3 native defaults `1.0 / 0.95`. Separate experiment.
- **Hardcoded temperature=0.4** in [agent_system/chatbot.py:211,266](../../../agent_system/chatbot.py#L211-L266). Fold into the Gemini 3 experiment.
