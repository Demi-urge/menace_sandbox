from pathlib import Path

from tests.test_self_debugger_sandbox import sds

SelfDebuggerSandbox = sds.SelfDebuggerSandbox
TargetRegion = sds.TargetRegion
TelemetryEvent = sds.TelemetryEvent


class DummyEngine:
    def __init__(self):
        self.calls = []

    def apply_patch(self, path, description, **kwargs):  # pragma: no cover - stub
        self.calls.append((path, description, kwargs))
        return None, False, 0.0


class DummyLogger:
    def __init__(self):
        self.events = []

    def log(self, event, *a, **k):  # pragma: no cover - stub
        if isinstance(event, TelemetryEvent):
            self.events.append(event)
        return event


def make_sandbox():
    engine = DummyEngine()
    sandbox = SelfDebuggerSandbox(None, engine)
    sandbox.error_logger = DummyLogger()
    return sandbox, engine, sandbox.error_logger


def test_escalation_and_reset():
    sandbox, engine, err_logger = make_sandbox()
    region = TargetRegion(start_line=1, end_line=2, function="f", filename="mod.py")

    sandbox._record_region_failure(region)  # count 1
    assert err_logger.events == []
    sandbox._record_region_failure(region)  # count 2 -> level 1
    assert engine.calls[-1][0] == Path("mod.py")
    assert engine.calls[-1][2]["target_region"] == region
    assert err_logger.events[-1].root_cause == "escalation_level_1"

    sandbox._record_region_failure(region)  # count 3
    sandbox._record_region_failure(region)  # count 4 -> level 2
    assert engine.calls[-1][0] == Path("mod.py")
    assert engine.calls[-1][2].get("target_region") is None
    assert err_logger.events[-1].root_cause == "escalation_level_2"

    sandbox._reset_failure_counter(region)
    key = (region.filename, region.function, region.start_line)
    assert key not in sandbox._failure_counts


def test_engine_respects_target_region(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text(
        "def f():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    return a + b\n"
    )

    class RecordingEngine:
        def __init__(self):
            self.calls = []

        def apply_patch(self, p, desc, **kwargs):  # pragma: no cover - stub
            self.calls.append((p, kwargs.get("target_region")))
            region = kwargs.get("target_region")
            text = path.read_text().splitlines()
            if region:
                for i in range(region.start_line - 1, region.end_line):
                    text[i] = "# patched"
            else:
                text = ["# module patched"]
            path.write_text("\n".join(text) + "\n")
            return None, False, 0.0

    engine = RecordingEngine()
    sandbox = SelfDebuggerSandbox(None, engine)
    sandbox.error_logger = DummyLogger()
    region = TargetRegion(start_line=2, end_line=3, function="f", filename=str(path))

    sandbox._record_region_failure(region)
    sandbox._record_region_failure(region)  # triggers function rewrite
    assert engine.calls[0][1] == region
    lines = path.read_text().splitlines()
    assert lines[0] == "def f():"
    assert lines[1] == "# patched" and lines[2] == "# patched"
    assert lines[3] == "    return a + b"

    sandbox._record_region_failure(region)
    sandbox._record_region_failure(region)  # triggers module rewrite
    assert engine.calls[1][1] is None
    assert path.read_text() == "# module patched\n"
