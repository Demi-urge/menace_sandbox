import sys
import types

from tests.test_visual_agent_auto_recover import _setup_va


def _stub_deps(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")

    def _noop_deco(*a, **k):
        def wrap(func):
            return func
        return wrap

    class DummyApp:
        def __init__(self, *a, **k):
            pass

        def on_event(self, *a, **k):
            return _noop_deco

        def post(self, *a, **k):
            return _noop_deco

        def get(self, *a, **k):
            return _noop_deco

    fastapi_mod.FastAPI = DummyApp
    fastapi_mod.HTTPException = type("HTTPException", (Exception,), {})
    fastapi_mod.Header = lambda default=None: default
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = type("BaseModel", (), {})
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.Config = type("Config", (), {})
    uvicorn_mod.Server = type("Server", (), {})
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)

    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")


def test_queue_operations(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    va = _setup_va(monkeypatch, tmp_path)
    q = va.task_queue
    q.clear()

    q.append({'id': 'a'})
    q.appendleft({'id': 'b'})
    assert len(q) == 2
    assert [item['id'] for item in q] == ['b', 'a']

    popped = q.popleft()
    assert popped['id'] == 'b'
    assert len(q) == 1

    q2 = va.PersistentQueue(va.QUEUE_FILE)
    assert [item['id'] for item in q2] == ['a']


def test_queue_backup_rotation(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    va = _setup_va(monkeypatch, tmp_path)
    q = va.task_queue
    q.clear()

    for i in range(4):
        q.append({'id': i})

    backups = sorted(tmp_path.glob('visual_agent_queue.jsonl.bak*'))
    assert len(backups) == va.BACKUP_COUNT
    counts = [len(b.read_text().splitlines()) for b in backups]
    assert counts == [3, 2, 1]


def test_recover_queue_from_corruption(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    va = _setup_va(monkeypatch, tmp_path)
    q = va.task_queue
    q.clear()
    va.job_status.clear()

    va.job_status['x'] = {'status': 'queued', 'prompt': 'p', 'branch': None}
    q.append({'id': 'x', 'prompt': 'p', 'branch': None})
    va._persist_state()
    va._persist_state()  # create backups

    va.QUEUE_FILE.write_text('bad')
    va.STATE_FILE.write_text('{')
    va.HASH_FILE.write_text('bad')

    q.clear()
    va.job_status.clear()

    va._recover_queue_file_locked()

    assert [item['id'] for item in va.task_queue] == ['x']
    assert va.job_status['x']['status'] == 'queued'
