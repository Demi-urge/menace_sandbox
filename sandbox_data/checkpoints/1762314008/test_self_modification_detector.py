import hashlib
from pathlib import Path

import pytest

import menace_sandbox.self_modification_detector as smd
from menace_sandbox.sandbox_settings import SandboxSettings


def _settings(tmp_path: Path, url: str | None = None) -> SandboxSettings:
    return SandboxSettings(
        sandbox_repo_path=str(tmp_path),
        self_mod_interval_seconds=0,
        self_mod_reference_path="ref.json",
        self_mod_reference_url=url,
        self_mod_lockdown_flag_path=str(tmp_path / "lock.flag"),
    )


def _run_once(monkeypatch, remote, current, tmp_path: Path):
    settings = _settings(tmp_path, url="http://ref")
    detector = smd.SelfModificationDetector(settings, base_dir=tmp_path)

    monkeypatch.setattr(
        smd.SelfModificationDetector,
        "_fetch_reference_hashes",
        staticmethod(lambda url: remote),
    )
    monkeypatch.setattr(
        smd.SelfModificationDetector,
        "generate_code_hashes",
        staticmethod(lambda d: current),
    )
    monkeypatch.setattr(
        smd.SelfModificationDetector,
        "load_reference_hashes",
        staticmethod(lambda p: {}),
    )
    monkeypatch.setattr(
        smd.SelfModificationDetector,
        "save_reference_hashes",
        staticmethod(lambda d, p: None),
    )

    called: list[list[str]] = []
    monkeypatch.setattr(detector, "trigger_lockdown", lambda files: called.append(files))

    class DummyEvent:
        def __init__(self):
            self.calls = 0

        def wait(self, _):
            self.calls += 1
            return self.calls > 1

        def clear(self):
            pass

    detector._stop_event = DummyEvent()

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            if self.target:
                self.target()

        def is_alive(self):
            return False

        def join(self):
            pass

    monkeypatch.setattr(smd.threading, "Thread", DummyThread)
    detector.start()
    return called


def test_lockdown_trigger_on_mismatch(monkeypatch, tmp_path):
    called = _run_once(monkeypatch, {"a.py": "1"}, {"a.py": "2"}, tmp_path)  # path-ignore
    assert called == [["a.py"]]  # path-ignore


def test_no_lockdown_on_match(monkeypatch, tmp_path):
    called = _run_once(monkeypatch, {"a.py": "1"}, {"a.py": "1"}, tmp_path)  # path-ignore
    assert called == []


def test_save_reference_hashes_logs_and_raises(tmp_path, caplog):
    path = tmp_path / "sub" / "ref.json"
    caplog.set_level("ERROR")
    with pytest.raises(OSError):
        smd.save_reference_hashes({"a": "1"}, str(path))
    assert "failed writing reference hashes" in caplog.text


def test_generate_code_hashes(tmp_path):
    root = tmp_path
    (root / "a.py").write_text("print('a')", encoding="utf-8")  # path-ignore
    (root / "b.txt").write_text("nope", encoding="utf-8")
    log_dir = root / "logs"
    log_dir.mkdir()
    (log_dir / "c.py").write_text("print('c')", encoding="utf-8")  # path-ignore
    hashes = smd.generate_code_hashes(str(root))
    assert set(hashes) == {"a.py"}  # path-ignore
    assert hashes["a.py"] == hashlib.sha256(b"print('a')").hexdigest()  # path-ignore


def test_thread_start_and_cleanup(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    detector = smd.SelfModificationDetector(settings, base_dir=tmp_path)
    monkeypatch.setattr(detector, "_load_reference", lambda: None)

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.join_called = False
            self.alive = True

        def start(self):
            self.alive = True

        def join(self):
            self.join_called = True
            self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(smd.threading, "Thread", DummyThread)
    detector.start()
    thread = detector._thread
    assert isinstance(thread, DummyThread)
    detector.stop()
    assert detector._thread is None
    assert detector._stop_event.is_set()
    assert thread.join_called


def test_reconfigure_updates_settings(tmp_path):
    settings = _settings(tmp_path)
    detector = smd.SelfModificationDetector(settings, base_dir=tmp_path)
    new = SandboxSettings(
        sandbox_repo_path=str(tmp_path),
        self_mod_interval_seconds=5,
        self_mod_reference_path="other.json",
        self_mod_reference_url="http://new",
        self_mod_lockdown_flag_path=str(tmp_path / "new.flag"),
    )
    detector.reconfigure(new)
    assert detector.interval_seconds == 5
    assert detector.reference_url == "http://new"
    assert detector.reference_path.name == "other.json"
    assert detector.lockdown_flag_path.name == "new.flag"
