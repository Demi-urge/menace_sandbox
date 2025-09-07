from menace import quick_fix_engine, self_debugger_sandbox, human_alignment_agent, violation_logger
from tests.test_self_debugger_sandbox import DummyBuilder
from pathlib import Path


def test_generate_patch_logs_alignment_warning(tmp_path, monkeypatch):
    src = tmp_path / "mod.py"  # path-ignore
    src.write_text("def f():\n    return 1\n")

    class Engine:
        def apply_patch(self, path: Path, desc: str, **kwargs):
            text = path.read_text() + "\n" + "eval('2+2')\n"
            path.write_text(text)
            return 1, False, 0.0

    logs = []

    def fake_log_violation(*args, **kwargs):
        logs.append((args, kwargs))

    monkeypatch.setattr(violation_logger, "log_violation", fake_log_violation)
    monkeypatch.setattr(quick_fix_engine, "log_violation", fake_log_violation)
    monkeypatch.setattr(human_alignment_agent, "log_violation", fake_log_violation)

    class DummyBuilder:
        def refresh_db_weights(self):
            return None

        def build(self, *a, **k):
            return ""

    pid = quick_fix_engine.generate_patch(
        str(src), Engine(), context_builder=DummyBuilder()
    )
    assert pid == 1
    assert logs, "expected alignment warning logged"


def test_self_debugger_preemptive_patch_logs_warning(tmp_path, monkeypatch):
    src = tmp_path / "mod.py"  # path-ignore
    src.write_text("def f():\n    return 1\n")

    logs = []

    def fake_log_violation(*args, **kwargs):
        logs.append((args, kwargs))

    monkeypatch.setattr(violation_logger, "log_violation", fake_log_violation)
    monkeypatch.setattr(self_debugger_sandbox, "log_violation", fake_log_violation)
    monkeypatch.setattr(human_alignment_agent, "log_violation", fake_log_violation)
    monkeypatch.setattr(self_debugger_sandbox, "log_record", lambda **kw: {})

    def stub_generate_patch(module: str, engine, **kwargs):
        path = Path(module)
        if path.suffix == "":
            path = path.with_suffix(".py")  # path-ignore
        text = path.read_text() + "\n" + "eval('2+2')\n"
        path.write_text(text)
        return 1

    monkeypatch.setattr(self_debugger_sandbox, "generate_patch", stub_generate_patch)

    class Predictor:
        def predict_high_risk_modules(self, top_n=5):
            return [str(src)]

    sandbox = self_debugger_sandbox.SelfDebuggerSandbox(
        object(),
        object(),
        context_builder=DummyBuilder(),
        error_predictor=Predictor(),
    )
    sandbox.preemptive_fix_high_risk_modules(limit=1)
    assert logs, "expected alignment warning logged"
