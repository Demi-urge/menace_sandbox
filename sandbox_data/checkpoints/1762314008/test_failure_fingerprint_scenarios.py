import json
import types
from dataclasses import asdict
from pathlib import Path

from .test_failure_fingerprint_logging import _make_engine, SAMPLE_TRACE, sce
from failure_fingerprint import FailureFingerprint
from prompt_optimizer import PromptOptimizer
import prompt_optimizer as po


def test_repeated_failures_record_fingerprints(monkeypatch, tmp_path):
    """Ensure multiple failing attempts write separate fingerprint entries."""
    monkeypatch.chdir(tmp_path)
    eng = _make_engine(tmp_path)

    def always_fail(self, path, description, context_meta=None, **kwargs):
        self._last_retry_trace = SAMPLE_TRACE
        return None, False, 0.0

    eng.apply_patch = types.MethodType(always_fail, eng)
    eng.apply_patch_with_retry(Path("mod.py"), "first", max_attempts=1)  # path-ignore
    eng.apply_patch_with_retry(Path("mod.py"), "second", max_attempts=1)  # path-ignore

    log_file = tmp_path / "failure_fingerprints.jsonl"
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2
    data = [json.loads(l) for l in lines]
    assert all(d["filename"] == "mod.py" for d in data)  # path-ignore


def test_retry_warns_on_matching_fingerprint(monkeypatch, tmp_path):
    """A retry with a known fingerprint injects a warning into the prompt."""
    monkeypatch.chdir(tmp_path)
    # First engine logs an initial failing fingerprint
    eng1 = _make_engine(tmp_path)

    def fail_once(self, path, description, context_meta=None, **kwargs):
        self._last_retry_trace = SAMPLE_TRACE
        return None, False, 0.0

    eng1.apply_patch = types.MethodType(fail_once, eng1)
    eng1.apply_patch_with_retry(Path("mod.py"), "first", max_attempts=1)  # path-ignore

    # Load the fingerprint that was written
    log_file = tmp_path / "failure_fingerprints.jsonl"
    entry = json.loads(log_file.read_text().strip())
    entry.pop("hash", None)
    prior = FailureFingerprint(**entry)

    # Second engine performs the retry and should inject a warning
    eng2 = _make_engine(tmp_path)
    calls: list[str] = []

    def fail_then_succeed(self, path, description, context_meta=None, **kwargs):
        calls.append(description)
        if len(calls) == 1:
            self._last_retry_trace = SAMPLE_TRACE
            return None, False, 0.0
        return 1, False, 0.0

    eng2.apply_patch = types.MethodType(fail_then_succeed, eng2)
    monkeypatch.setattr(
        sce,
        "find_similar",
        lambda emb, thresh, path="failure_fingerprints.jsonl": [prior],
    )
    monkeypatch.setattr(
        sce,
        "check_similarity_and_warn",
        lambda *a, **k: (a[3], False, 0.0, [prior], "WARNING"),
    )

    pid, reverted, _ = eng2.apply_patch_with_retry(Path("mod.py"), "second", max_attempts=2)  # path-ignore
    assert pid == 1 and not reverted
    assert any("WARNING" in d for d in calls[1:])


def test_retry_skips_after_similarity_limit(monkeypatch, tmp_path):
    """Too many similar fingerprints cause the retry loop to skip execution."""
    monkeypatch.chdir(tmp_path)
    # Log an initial failing fingerprint
    eng1 = _make_engine(tmp_path)

    def fail_once(self, path, description, context_meta=None, **kwargs):
        self._last_retry_trace = SAMPLE_TRACE
        return None, False, 0.0

    eng1.apply_patch = types.MethodType(fail_once, eng1)
    eng1.apply_patch_with_retry(Path("mod.py"), "first", max_attempts=1)  # path-ignore

    log_file = tmp_path / "failure_fingerprints.jsonl"
    entry = json.loads(log_file.read_text().strip())
    entry.pop("hash", None)
    prior = FailureFingerprint(**entry)

    # Second engine should skip due to multiple similar fingerprints
    eng2 = _make_engine(tmp_path)
    calls: list[str] = []

    def fail_only(self, path, description, context_meta=None, **kwargs):
        calls.append(description)
        self._last_retry_trace = SAMPLE_TRACE
        return None, False, 0.0

    eng2.apply_patch = types.MethodType(fail_only, eng2)
    monkeypatch.setattr(
        sce,
        "find_similar",
        lambda emb, thresh, path="failure_fingerprints.jsonl": [prior, prior, prior],
    )
    monkeypatch.setattr(
        sce,
        "check_similarity_and_warn",
        lambda *a, **k: (a[3], False, 0.0, [prior, prior, prior], "WARNING"),
    )

    pid, _, _ = eng2.apply_patch_with_retry(Path("mod.py"), "second", max_attempts=3)  # path-ignore
    assert pid is None and len(calls) == 1


def test_prompt_optimizer_penalizes_from_fingerprint_log(tmp_path):
    """Prompt optimiser lowers scores when fingerprints exceed threshold."""
    success = tmp_path / "success.jsonl"
    failure = tmp_path / "failure.jsonl"
    log_file = tmp_path / "failure_fingerprints.jsonl"

    success.write_text(
        json.dumps(
            {
                "module": "mod.py",  # path-ignore
                "action": "<module>",
                "prompt": "prompt!",
                "success": True,
                "roi": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failure.write_text("", encoding="utf-8")

    fp = FailureFingerprint("mod.py", "<module>", "err", "trace", "prompt!")  # path-ignore
    fp_data = asdict(fp)
    fp_data.pop("hash", None)
    log_file.write_text(
        "\n".join([json.dumps(fp_data), json.dumps(fp_data)]) + "\n",
        encoding="utf-8",
    )

    po.FailureFingerprint = FailureFingerprint
    opt = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "stats.json",
        failure_fingerprints_path=log_file,
        fingerprint_threshold=1,
    )
    key = (
        "mod.py",  # path-ignore
        "<module>",
        "neutral",
        (),
        "none",
        False,
        False,
        False,
    )
    assert opt.stats[key].penalty_factor < 1.0
