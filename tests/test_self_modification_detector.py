import hashlib
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


def test_generate_code_hashes(tmp_path):
    root = tmp_path
    (root / "a.py").write_text("print('a')", encoding="utf-8")
    (root / "b.txt").write_text("nope", encoding="utf-8")
    log_dir = root / "logs"
    log_dir.mkdir()
    (log_dir / "c.py").write_text("print('c')", encoding="utf-8")
    hashes = smd.generate_code_hashes(str(root))
    assert set(hashes) == {"a.py"}
    assert hashes["a.py"] == hashlib.sha256(b"print('a')").hexdigest()


def test_monitor_start_and_stop(monkeypatch):
    class DummyEvent:
        def __init__(self):
            self.cleared = False
            self.set_called = False

        def wait(self, _):
            return True

        def clear(self):
            self.cleared = True

        def set(self):
            self.set_called = True

    event = DummyEvent()
    monkeypatch.setattr(smd, "_STOP_EVENT", event)
    monkeypatch.setattr(smd, "_REFERENCE_HASHES", {})
    monkeypatch.setattr(smd, "generate_code_hashes", lambda d: {})
    monkeypatch.setattr(smd, "load_reference_hashes", lambda p: {})
    monkeypatch.setattr(smd, "save_reference_hashes", lambda d, p: None)
    monkeypatch.setattr(smd, "detect_self_modification", lambda r, c: [])

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.join_called = False
            self.alive = True

        def start(self):
            if self.target:
                self.target()

        def join(self):
            self.join_called = True
            self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(smd.threading, "Thread", DummyThread)
    smd._MONITOR_THREAD = None

    smd.monitor_self_integrity(interval_seconds=0)
    thread = smd._MONITOR_THREAD
    assert isinstance(thread, DummyThread)
    assert event.cleared

    smd.stop_monitoring()
    assert event.set_called
    assert thread.join_called
    assert smd._MONITOR_THREAD is None
