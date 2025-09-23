from __future__ import annotations

from pathlib import Path
import types

from menace_sandbox.gpt_memory import GPTMemoryManager
from tests import test_self_coding_engine as sce_mod
from tests import test_self_improvement_logging as sil


def test_memory_logging_tags(monkeypatch):
    mem = GPTMemoryManager(":memory:")
    sie = sil._load_engine()
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    engine.gpt_memory = mem
    engine.logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)
    engine._memory_summaries = lambda module: ""
    engine.self_coding_engine = None
    monkeypatch.setattr(sie, "generate_patch", lambda m, e, **kw: 1)
    sie.SelfImprovementEngine._generate_patch_with_memory(engine, "mod", "act")
    rows = mem.conn.execute("SELECT tags FROM interactions").fetchall()
    all_tags = ",".join(r[0] for r in rows)
    assert "improvement_path" in all_tags
    assert "feedback" in all_tags

    mem2 = GPTMemoryManager(":memory:")
    coder_mod = sce_mod.sce
    coder = coder_mod.SelfCodingEngine.__new__(coder_mod.SelfCodingEngine)
    coder.gpt_memory = mem2
    coder.logger = types.SimpleNamespace(exception=lambda *a, **k: None)
    coder_mod.SelfCodingEngine._store_patch_memory(coder, Path("x.py"), "desc", "code", True, 0.5)  # path-ignore
    rows2 = mem2.conn.execute("SELECT response, tags FROM interactions").fetchall()
    tag_str = ",".join(r[1] for r in rows2)
    assert "error_fix" in tag_str
    assert "feedback" in tag_str
    assert any("status=success" in r[0] for r in rows2 if "feedback" in r[1])
