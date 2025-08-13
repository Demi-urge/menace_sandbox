from types import MethodType

import pytest

from menace.error_bot import ErrorDB
from menace.error_logger import TelemetryEvent


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_errordb_vector_search(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
    db = ErrorDB(
        path=tmp_path / "e.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"e.{backend}.index",
    )

    captured: list[str] = []

    def fake_embed(self, text: str):
        captured.append(text)
        if "alpha" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    ev1 = TelemetryEvent(root_cause="alpha", stack_trace="trace1")
    ev2 = TelemetryEvent(root_cause="beta", stack_trace="trace2")
    db.add_telemetry(ev1)
    db.add_telemetry(ev2)

    assert len(captured) == 2
    assert "alpha" in captured[0] and "trace1" in captured[0]

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0]["cause"] == "alpha"
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0]["cause"] == "beta"
