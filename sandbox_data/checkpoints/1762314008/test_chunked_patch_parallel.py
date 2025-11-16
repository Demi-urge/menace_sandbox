from __future__ import annotations

import threading
import types
from pathlib import Path

from tests.integration.test_chunked_patch_flow import _setup_engine, sce
from chunking import CodeChunk
import chunking as pc


def _prepare_engine(tmp_path: Path, monkeypatch):
    engine = _setup_engine(tmp_path, monkeypatch)
    path = tmp_path / "big.py"  # path-ignore
    path.write_text("def a():\n    pass\n\ndef b():\n    pass\n")

    monkeypatch.setattr(sce, "_count_tokens", lambda text: 1000)
    monkeypatch.setattr(
        sce,
        "split_into_chunks",
        lambda code, limit: [
            CodeChunk(1, 2, "def a():\n    pass", "h1", 5),
            CodeChunk(4, 5, "def b():\n    pass", "h2", 5),
        ],
    )
    monkeypatch.setattr(
        pc, "summarize_code", lambda text, llm, context_builder=None: text.splitlines()[0]
    )
    return engine, path


def test_chunk_prompts_dispatched_concurrently(tmp_path, monkeypatch):
    engine, path = _prepare_engine(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)

    def fake_generate_helper(desc, *a, **k):
        barrier.wait(timeout=2)
        return "# patch"

    monkeypatch.setattr(engine, "generate_helper", fake_generate_helper)
    monkeypatch.setattr(engine, "_run_ci", lambda p: types.SimpleNamespace(success=True))

    engine.apply_patch(path, "add patches")

    lines = path.read_text().splitlines()
    assert lines.count("# patch") == 2


def test_rollback_on_chunk_failure(tmp_path, monkeypatch):
    engine, path = _prepare_engine(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)

    def fake_generate_helper(desc, *a, **k):
        barrier.wait(timeout=2)
        return "# patch"

    monkeypatch.setattr(engine, "generate_helper", fake_generate_helper)

    ci_calls: list[int] = []

    def fake_run_ci(p):
        idx = len(ci_calls)
        ci_calls.append(idx)
        return types.SimpleNamespace(success=idx == 0)

    monkeypatch.setattr(engine, "_run_ci", fake_run_ci)

    engine.apply_patch(path, "add patches")

    lines = path.read_text().splitlines()
    assert lines.count("# patch") == 1
