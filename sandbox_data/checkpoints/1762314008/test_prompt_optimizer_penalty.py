import json
import sys

from failure_fingerprint import FailureFingerprint
from failure_fingerprint_store import FailureFingerprintStore

sys.modules.pop("prompt_optimizer", None)
from prompt_optimizer import PromptOptimizer


class _DummyVectorStore:
    def add(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        return []


class _DummyVectorService:
    def __init__(self) -> None:
        self.vector_store = _DummyVectorStore()

    def vectorise(self, kind, record):  # pragma: no cover - trivial
        text = record.get("text", "")
        return [float(len(text))]


def _make_store() -> FailureFingerprintStore:
    svc = _DummyVectorService()
    return FailureFingerprintStore(path=None, vector_service=svc, similarity_threshold=0.0)


def test_failure_fingerprints_penalize(tmp_path):
    success = tmp_path / "success.jsonl"
    failure = tmp_path / "failure.jsonl"
    success.write_text(
        json.dumps(
            {
                "module": "m",
                "action": "a",
                "prompt": "# H\nExample",
                "success": True,
                "roi": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failure.write_text("", encoding="utf-8")

    store = _make_store()
    fp1 = FailureFingerprint("m", "a", "err", "trace", "# H\nExample")
    fp2 = FailureFingerprint("m", "a", "err", "trace", "# H\nExample")
    store.add(fp1)
    store.add(fp2)

    opt = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "stats.json",
        failure_store=store,
        fingerprint_threshold=1,
    )
    key = (
        "m",
        "a",
        "neutral",
        ("H",),
        "start",
        False,
        False,
        False,
    )
    stat = opt.stats[key]
    assert stat.penalty_factor < 1.0


def test_failure_fingerprints_reduce_score(tmp_path):
    success = tmp_path / "success.jsonl"
    failure = tmp_path / "failure.jsonl"
    success.write_text(
        json.dumps(
            {
                "module": "m",
                "action": "a",
                "prompt": "# H\nExample",
                "success": True,
                "roi": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failure.write_text("", encoding="utf-8")
    key = (
        "m",
        "a",
        "neutral",
        ("H",),
        "start",
        False,
        False,
        False,
    )
    opt_base = PromptOptimizer(success, failure, stats_path=tmp_path / "s1.json")
    score_base = opt_base.stats[key].score()

    store = _make_store()
    fp1 = FailureFingerprint("m", "a", "err", "trace", "# H\nExample")
    fp2 = FailureFingerprint("m", "a", "err", "trace", "# H\nExample")
    store.add(fp1)
    store.add(fp2)

    opt_pen = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "s2.json",
        failure_store=store,
        fingerprint_threshold=1,
    )
    score_pen = opt_pen.stats[key].score()
    assert score_pen < score_base
