import json
from menace import violation_logger as vl


def test_log_and_load(tmp_path, monkeypatch):
    log_path = tmp_path / "violation_log.jsonl"
    monkeypatch.setattr(vl, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vl, "LOG_PATH", str(log_path))

    vl.log_violation("abc", "policy", 3, {"detail": "bad"})
    vl.log_violation("def", "security", 5, {"detail": "worse"})

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["entry_id"] == "abc"

    recents = vl.load_recent_violations(1)
    assert recents and recents[0]["entry_id"] == "def"

    summary = vl.violation_summary("abc")
    assert "policy" in summary


class DummyBus:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


def test_alignment_warning_event_bus(tmp_path, monkeypatch):
    log_path = tmp_path / "violation_log.jsonl"
    monkeypatch.setattr(vl, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vl, "LOG_PATH", str(log_path))

    class ImmediateThread:
        def __init__(self, target, daemon=True):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(vl, "Thread", ImmediateThread)

    bus = DummyBus()
    vl.set_event_bus(bus)
    vl.log_violation("a1", "alignment", 1, {}, alignment_warning=True)
    assert bus.published[0][0] == "alignment:flag"
    vl.set_event_bus(None)


def test_cli_alignment_warnings(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    log_path = tmp_path / "violation_log.jsonl"
    monkeypatch.setattr(vl, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vl, "LOG_PATH", str(log_path))

    vl.log_violation("warn1", "alignment", 2, {}, alignment_warning=True)

    from sandbox_runner import cli

    cli.main(["--alignment-warnings"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data and data[0]["entry_id"] == "warn1"


def test_recent_alignment_warnings_alias(tmp_path, monkeypatch):
    log_path = tmp_path / "violation_log.jsonl"
    monkeypatch.setattr(vl, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(vl, "LOG_PATH", str(log_path))

    vl.log_violation("warn2", "alignment", 3, {}, alignment_warning=True)
    warnings = vl.recent_alignment_warnings(10)
    assert warnings and warnings[0]["entry_id"] == "warn2"
