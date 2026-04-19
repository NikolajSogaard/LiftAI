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
