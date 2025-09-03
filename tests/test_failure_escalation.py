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
    region = TargetRegion(path="mod.py", start_line=1, end_line=2, func_name="f")

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
    key = (region.path, region.func_name, region.start_line)
    assert key not in sandbox._failure_counts
