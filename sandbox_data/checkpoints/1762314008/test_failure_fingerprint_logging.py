import json
import types
from pathlib import Path

from .test_failure_fingerprint_retry import SAMPLE_TRACE
from . import test_self_coding_engine_chunking as setup

from failure_fingerprint import FailureFingerprint

sce = setup.sce


def _make_engine(tmp_path):
    eng = sce.SelfCodingEngine(
        code_db=object(),
        memory_mgr=object(),
        patch_db=None,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    eng._build_retry_context = lambda desc, rep: {}
    eng._run_ci = lambda path=None: None
    eng._failure_cache = types.SimpleNamespace(seen=lambda t: False, add=lambda f: None)
    eng.logger = types.SimpleNamespace(
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    eng.audit_trail = types.SimpleNamespace(record=lambda payload: None)
    eng.data_bot = None
    eng._last_prompt = "prompt!"
    return eng


def test_failed_patch_logs_fingerprint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    eng = _make_engine(tmp_path)

    def fail(self, path, description, context_meta=None, **kwargs):
        self._last_retry_trace = SAMPLE_TRACE
        return None, False, 0.0

    eng.apply_patch = types.MethodType(fail, eng)
    eng.apply_patch_with_retry(Path("mod.py"), "desc", max_attempts=1)  # path-ignore

    log_file = tmp_path / "failure_fingerprints.jsonl"
    data = json.loads(log_file.read_text().strip())
    assert data["filename"] == "mod.py"  # path-ignore
    assert data["function_name"] == "<module>"
    assert "ZeroDivisionError" in data["error_message"]
    assert data["prompt_text"] == "prompt!"
    assert len(data["embedding"]) > 0


def test_second_run_warns_on_similar_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    eng2 = _make_engine(tmp_path)
    calls: list[str] = []

    def fail_then_succeed(self, path, description, context_meta=None, **kwargs):
        calls.append(description)
        if len(calls) == 1:
            self._last_retry_trace = SAMPLE_TRACE
            return None, False, 0.0
        return 1, False, 0.0

    eng2.apply_patch = types.MethodType(fail_then_succeed, eng2)
    prior = FailureFingerprint.from_failure(
        "prev.py", "main", SAMPLE_TRACE, "Boom", "p"  # path-ignore
    )
    monkeypatch.setattr(sce, "find_similar", lambda emb, thresh: [prior])
    monkeypatch.setattr(
        sce,
        "check_similarity_and_warn",
        lambda *a, **k: (a[3], False, 0.0, [prior], "WARNING"),
    )
    pid, reverted, delta = eng2.apply_patch_with_retry(Path("mod.py"), "desc", max_attempts=2)  # path-ignore
    assert pid == 1 and not reverted
    assert any("WARNING" in d for d in calls[1:])


def test_second_run_skips_after_similarity_limit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    eng = _make_engine(tmp_path)
    calls: list[str] = []

    def fail_only(self, path, description, context_meta=None, **kwargs):
        calls.append(description)
        self._last_retry_trace = SAMPLE_TRACE
        return None, False, 0.0

    eng.apply_patch = types.MethodType(fail_only, eng)
    prior = FailureFingerprint.from_failure(
        "mod.py", "<module>", SAMPLE_TRACE, "ZeroDivisionError: division by zero", "prompt!"  # path-ignore
    )
    monkeypatch.setattr(sce, "find_similar", lambda emb, thresh: [prior, prior, prior])
    monkeypatch.setattr(
        sce,
        "check_similarity_and_warn",
        lambda *a, **k: (a[3], False, 0.0, [prior, prior, prior], "WARNING"),
    )
    pid, reverted, delta = eng.apply_patch_with_retry(Path("mod.py"), "desc", max_attempts=3)  # path-ignore
    assert pid is None and len(calls) == 1
