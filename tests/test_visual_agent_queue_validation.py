import json
from tests.test_visual_agent_auto_recover import _setup_va
from tests.test_persistent_queue import _stub_deps


def test_recover_invalid_queue(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    va = _setup_va(monkeypatch, tmp_path)
    va.job_status['x'] = {'status': 'queued', 'prompt': 'p', 'branch': None}
    va.task_queue.append({'id': 'x', 'prompt': 'p', 'branch': None})
    va._persist_state()
    va._persist_state()  # create backups

    va.QUEUE_FILE.write_text(json.dumps({'id': 'x'}) + '\n')
    va.task_queue.clear()
    va.job_status.clear()

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()

    assert [item['id'] for item in va2.task_queue] == ['x']
    assert va2.job_status['x']['status'] == 'queued'
    line = va2.QUEUE_FILE.read_text().splitlines()[0]
    assert json.loads(line)['prompt'] == 'p'


def test_validate_job_status(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    va = _setup_va(monkeypatch, tmp_path)
    va.job_status['x'] = {'status': 'queued', 'prompt': 'p', 'branch': None}
    va.task_queue.append({'id': 'x', 'prompt': 'p', 'branch': None})
    va.job_status['y'] = {'status': 'completed', 'prompt': 'q', 'branch': None}
    va.task_queue.append({'id': 'y', 'prompt': 'q', 'branch': None})
    va._persist_state()

    import sqlite3
    with sqlite3.connect(tmp_path / 'visual_agent_queue.db') as conn:
        conn.execute('DELETE FROM tasks WHERE id=?', ('x',))
        conn.commit()

    va.job_status.clear()

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()

    q_status = va2.task_queue.get_status()
    assert 'x' in q_status
    assert q_status['x']['status'] == 'queued'
    assert va2.job_status['y']['status'] == 'queued'
    assert va2.job_status == q_status
