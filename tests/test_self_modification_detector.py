import types
import menace.self_modification_detector as smd


def _run_once(monkeypatch, remote, current):
    def fake_get(url, timeout=5):
        return types.SimpleNamespace(status_code=200, json=lambda: remote)

    monkeypatch.setattr(smd, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(smd, "generate_code_hashes", lambda d: current)
    monkeypatch.setattr(smd, "load_reference_hashes", lambda p: {})
    monkeypatch.setattr(smd, "save_reference_hashes", lambda d, p: None)

    called = []
    monkeypatch.setattr(smd, "trigger_lockdown", lambda files: called.append(files))

    class DummyEvent:
        def __init__(self):
            self.calls = 0
        def wait(self, _):
            self.calls += 1
            return self.calls > 1

    monkeypatch.setattr(smd, "_STOP_EVENT", DummyEvent())
    monkeypatch.setattr(smd, "_MONITOR_THREAD", None)

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
        def start(self):
            if self.target:
                self.target()
        def is_alive(self):
            return False

    monkeypatch.setattr(smd.threading, "Thread", DummyThread)

    smd.monitor_self_integrity(interval_seconds=0, reference_url="http://ref")
    return called


def test_lockdown_trigger_on_mismatch(monkeypatch):
    called = _run_once(monkeypatch, {"a.py": "1"}, {"a.py": "2"})
    assert called == [["a.py"]]


def test_no_lockdown_on_match(monkeypatch):
    called = _run_once(monkeypatch, {"a.py": "1"}, {"a.py": "1"})
    assert called == []


def test_save_reference_hashes_logs_warning(tmp_path, caplog):
    path = tmp_path / "sub" / "ref.json"
    caplog.set_level("ERROR")
    smd.save_reference_hashes({"a": "1"}, str(path))
    assert "failed writing reference hashes" in caplog.text
