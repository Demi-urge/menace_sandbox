from failure_fingerprint_store import FailureFingerprint, FailureFingerprintStore
from failure_region_cli import cli
from .test_failure_fingerprint_store import DummyVectorService


def test_region_cli_counts(tmp_path, capsys):
    svc = DummyVectorService()
    store = FailureFingerprintStore(
        path=tmp_path / "fps.jsonl",
        vector_service=svc,
        similarity_threshold=0.9,
        compact_interval=0,
    )
    store.add(
        FailureFingerprint(
            "a.py",  # path-ignore
            "f",
            "e",
            "trace one",
            "p",
            target_region="us",
            escalation_level=1,
        )
    )
    store.add(
        FailureFingerprint(
            "b.py",  # path-ignore
            "g",
            "e",
            "trace two",
            "p",
            target_region="us",
            escalation_level=2,
        )
    )
    store.add(
        FailureFingerprint(
            "c.py",  # path-ignore
            "h",
            "e",
            "trace three",
            "p",
            target_region="eu",
            escalation_level=0,
        )
    )
    cli(["--path", str(store.path)])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "region\tcount\tescalation"
    assert out[1] == "us\t2\t2"
    assert out[2] == "eu\t1\t0"
