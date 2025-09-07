from __future__ import annotations

from tests import test_self_coding_engine_chunking as setup
import types

sce = setup.sce
TargetRegion = sce.TargetRegion
SelfCodingEngine = sce.SelfCodingEngine


class DummyFP:
    def __init__(self, filename, function_name, stack_trace, error_message, prompt_text, timestamp=0.0):
        self.filename = filename
        self.function_name = function_name
        self.stack_trace = stack_trace
        self.error_message = error_message
        self.prompt_text = prompt_text
        self.embedding = [0.0]
        self.timestamp = timestamp

    @classmethod
    def from_failure(cls, filename, function_name, stack_trace, error_message, prompt_text):
        return cls(filename, function_name, stack_trace, error_message, prompt_text)


def test_escalation_metrics(monkeypatch, tmp_path):
    gauge_calls: dict[tuple[tuple[str, str], ...], int] = {}

    class DummyGauge:
        def __init__(self) -> None:
            self.events: list[tuple[tuple[str, str], ...]] = []

        def labels(self, **labels):
            key = tuple(sorted(labels.items()))

            def inc(amount: float = 1.0) -> None:
                gauge_calls[key] = gauge_calls.get(key, 0) + amount
                self.events.append(key)

            return types.SimpleNamespace(inc=inc)

    patch_attempts = DummyGauge()
    patch_escalations = DummyGauge()
    monkeypatch.setattr(sce, "_PATCH_ATTEMPTS", patch_attempts)
    monkeypatch.setattr(sce, "_PATCH_ESCALATIONS", patch_escalations)

    engine = SelfCodingEngine(
        code_db=object(),
        memory_mgr=object(),
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    engine.audit_trail = types.SimpleNamespace(record=lambda payload: None)
    engine._build_retry_context = lambda desc, rep: {}
    engine._failure_cache = types.SimpleNamespace(seen=lambda trace: False, add=lambda trace: None)
    engine.failure_similarity_tracker = types.SimpleNamespace(update=lambda **k: None, get=lambda key: 0.0, std=lambda key: 1.0)
    engine._save_state = lambda: None
    monkeypatch.setattr(sce, "check_similarity_and_warn", lambda *a, **k: (a[3], False, 0.0, [], ""))
    monkeypatch.setattr(sce, "FailureFingerprint", DummyFP)
    monkeypatch.setattr(sce, "log_fingerprint", lambda fp: None)
    monkeypatch.setattr(sce, "record_failure", lambda *a, **k: None)
    monkeypatch.setattr(sce, "parse_failure", lambda trace: types.SimpleNamespace(trace=trace, tags=set()))

    calls: list[TargetRegion | None] = []

    def fake_apply(self, path, description, context_meta=None, target_region=None, **kwargs):
        calls.append(target_region)
        self._last_retry_trace = (
            f'Traceback (most recent call last):\n'
            f'  File "{path}", line 1, in {target_region.function if target_region else "<module>"}\n'
            '    1/0\n'
            'ZeroDivisionError: division by zero'
        )
        return None, False, 0.0

    engine.apply_patch = types.MethodType(fake_apply, engine)

    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("def f():\n    a=1\n    b=2\n    return a+b\n")
    region = TargetRegion(start_line=2, end_line=2, function="f")

    engine.apply_patch_with_retry(path, "desc", max_attempts=5, target_region=region)

    assert len(calls) == 5
    assert calls[0].start_line == 2 and calls[0].end_line == 2
    assert calls[1].start_line == 2 and calls[1].end_line == 2
    assert calls[2].start_line == 1 and calls[2].end_line == 4
    assert calls[3].start_line == 1 and calls[3].end_line == 4
    assert calls[4] is None

    assert gauge_calls[(('scope', 'line'),)] == 2
    assert gauge_calls[(('scope', 'function'),)] == 2
    assert gauge_calls[(('scope', 'module'),)] == 1
    assert gauge_calls[(('level', 'function'),)] == 1
    assert gauge_calls[(('level', 'module'),)] == 1
    assert patch_escalations.events == [
        (('level', 'function'),),
        (('level', 'module'),),
    ]
