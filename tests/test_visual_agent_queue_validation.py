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
