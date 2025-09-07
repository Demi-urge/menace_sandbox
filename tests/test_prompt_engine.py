from typing import Any, Dict, List

import json
import pytest
import sqlite3 as sq3
import sys
import types
import tempfile

# Stub heavy dependencies before importing the trainer
sys.modules.setdefault("gpt_memory", types.SimpleNamespace(GPTMemoryManager=object))
sys.modules.setdefault("code_database", types.SimpleNamespace(PatchHistoryDB=object))
sys.modules.setdefault("menace_sandbox", types.SimpleNamespace())

class _PMT:
    def __init__(self, *a, **k):
        pass

    def _extract_style(self, prompt: str):
        has_code = "```" in prompt
        has_bullets = "-" in prompt
        sections = [s.split(":")[0].lower() for s in prompt.splitlines() if ":" in s]
        return {"has_code": has_code, "has_bullets": has_bullets, "sections": sections[:2]}

    def train(self):
        return {"headers": {json.dumps(["H"]): 2 / 7}}

    def record(self, **_k):
        return True


sys.modules.setdefault(
    "prompt_memory_trainer", types.SimpleNamespace(PromptMemoryTrainer=_PMT)
)

from prompt_engine import (
    PromptEngine,
    DEFAULT_TEMPLATE,
    diff_within_target_region,
)  # noqa: E402
from self_improvement.target_region import TargetRegion  # noqa: E402
from prompt_memory_trainer import PromptMemoryTrainer  # noqa: E402

DUMMY_BUILDER = object()

class FallbackResult:
    def __init__(self, reason: str, patches: List[Dict[str, Any]], confidence: float = 0.0):
        self.reason = reason
        self.confidence = confidence

    def __iter__(self):
        return iter([])


class RoiTag:
    HIGH_ROI = types.SimpleNamespace(value="high-ROI")
    BUG_INTRODUCED = types.SimpleNamespace(value="bug-introduced")
    SUCCESS = types.SimpleNamespace(value="success")


class DummyRetriever:
    """Basic retriever returning predefined records."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def search(self, query: str, top_k: int):  # pragma: no cover - trivial
        return self.records[:top_k]


class DummyFallbackRetriever:
    """Retriever that always returns a fallback result."""

    def search(self, query: str, top_k: int):  # pragma: no cover - trivial
        return FallbackResult("low_confidence", [], confidence=0.1)


def _record(score: float, **meta: Any) -> Dict[str, Any]:
    """Helper to build retriever records with ``score`` and metadata."""

    return {"score": score, "metadata": meta}


def test_prompt_engine_returns_confidences_and_tags():
    records = [
        _record(
            0.9,
            summary="good",
            tests_passed=True,
            raroi=0.3,
            roi_tag=RoiTag.HIGH_ROI.value,
        ),
        _record(
            0.8,
            summary="bad",
            tests_passed=False,
            raroi=0.1,
            roi_tag=RoiTag.BUG_INTRODUCED.value,
        ),
    ]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        confidence_threshold=-1.0,
        context_builder=DUMMY_BUILDER,
    )
    prompt = engine.build_prompt("desc", context_builder=DUMMY_BUILDER)
    assert len(prompt.examples) == 2
    assert len(prompt.examples) == len(prompt.vector_confidences)
    assert RoiTag.HIGH_ROI.value in prompt.outcome_tags
    assert RoiTag.BUG_INTRODUCED.value in prompt.outcome_tags


def test_prompt_engine_sections_and_ranking():
    records = [
        _record(0.9, raroi=0.4, summary="low", tests_passed=True, ts=1),
        _record(0.8, raroi=0.9, summary="high", tests_passed=True, ts=2),
        _record(0.7, summary="fail", tests_passed=False, ts=2),
    ]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        top_n=3,
        confidence_threshold=0.0,
        context_builder=DUMMY_BUILDER,
        trainer=object(),
    )
    prompt = engine.build_prompt("desc", context_builder=DUMMY_BUILDER)

    text = str(prompt)
    # Sections from build_snippets are present
    assert "Given the following pattern:" in text
    assert "Avoid fail because it caused a failure:" in text

    # Ranking respects ROI values
    assert text.index("Code summary: high") < text.index("Code summary: low")
    # Failure snippet appears after successes
    assert text.rindex("Code summary: fail") > text.index(
        "Avoid fail because it caused a failure:"
    )


def test_prompt_engine_custom_headers():
    records = [
        _record(0.9, summary="good", tests_passed=True, ts=1),
        _record(0.8, summary="bad", tests_passed=False, ts=2),
    ]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        top_n=2,
        confidence_threshold=0.0,
        success_header="Correct example:",
        failure_header="Incorrect example:",
        context_builder=DUMMY_BUILDER,
        trainer=object(),
    )
    prompt = engine.build_prompt("desc", context_builder=DUMMY_BUILDER)
    text = str(prompt)
    assert "Correct example:" in text
    assert "Incorrect example:" in text
    assert "Given the following pattern:" not in text
    assert "Avoid bad because it caused a failure:" not in text


def test_prompt_engine_applies_trained_preferences(monkeypatch):
    records = [
        _record(0.9, summary="good", tests_passed=True, ts=1),
        _record(0.8, summary="bad", outcome="oops", tests_passed=False, ts=2),
    ]

    class StubTrainer:
        def __init__(self, *a, **k):
            pass

        def train(self):
            return {
                "headers": {
                    json.dumps(
                        ["Preferred header:", "Avoid {summary} because it caused {outcome}:"]
                    ): 0.9
                },
                "example_order": {json.dumps(["failure", "success"]): 0.9},
                "tone": {"excited": 0.9},
            }

    import prompt_engine as pe

    monkeypatch.setattr(pe, "PromptMemoryTrainer", StubTrainer)

    engine = pe.PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        context_builder=DUMMY_BUILDER,
        top_n=2,
        confidence_threshold=0.0,
    )
    prompt = engine.build_prompt("desc", context_builder=DUMMY_BUILDER)

    # learned success header is used
    assert "Preferred header:" in prompt
    # failure example appears before success example per learned order
    assert prompt.index("Avoid bad because it caused oops:") < prompt.index(
        "Preferred header:"
    )
    # tone preference applied
    assert engine.tone == "excited"
    assert engine.last_metadata["tone"] == "excited"


def test_prompt_engine_handles_retry_trace():
    records = [_record(1.0, summary="foo", tests_passed=True, raroi=0.5)]
    engine = PromptEngine(patch_retriever=DummyRetriever(records), context_builder=DUMMY_BUILDER)
    trace = "Previous failure:\nTraceback: fail\nPlease attempt a different solution."
    prompt = engine.build_prompt("goal", retry_trace=trace, context_builder=DUMMY_BUILDER)

    assert prompt.count("Previous failure:") == 1
    assert "Traceback: fail" in prompt
    assert prompt.strip().endswith("Please attempt a different solution.")


def test_prompt_engine_uses_template_on_low_confidence(monkeypatch):
    records = [_record(0.0, summary="bad", tests_passed=True)]
    engine = PromptEngine(patch_retriever=DummyRetriever(records), context_builder=DUMMY_BUILDER)
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    prompt = engine.build_prompt("goal", context_builder=DUMMY_BUILDER)
    assert prompt.user == DEFAULT_TEMPLATE


def test_prompt_engine_handles_fallback_result(monkeypatch):
    engine = PromptEngine(patch_retriever=DummyFallbackRetriever(), context_builder=DUMMY_BUILDER)
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    prompt = engine.build_prompt("goal", context_builder=DUMMY_BUILDER)
    assert prompt.user == DEFAULT_TEMPLATE


def test_weighted_scoring_alters_ordering():
    records = [
        _record(0.0, raroi=1.0, summary="roi", tests_passed=True, ts=1),
        _record(0.0, raroi=0.2, summary="recent", tests_passed=True, ts=2),
    ]
    engine = PromptEngine(
        patch_retriever=DummyRetriever(records),
        top_n=2,
        roi_weight=1.0,
        recency_weight=0.0,
        confidence_threshold=-10,
        context_builder=DUMMY_BUILDER,
    )

    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "roi"

    engine.roi_weight = 0.0
    engine.recency_weight = 1.0
    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "recent"


def test_roi_tag_weights_adjust_ranking():
    records = [
        _record(
            0.0,
            raroi=0.5,
            summary="good",
            tests_passed=True,
            ts=1,
            roi_tag="high-ROI",
        ),
        _record(
            0.0,
            raroi=0.5,
            summary="bad",
            tests_passed=True,
            ts=1,
            roi_tag="bug-introduced",
        ),
    ]
    engine = PromptEngine(
        patch_retriever=DummyRetriever(records),
        top_n=2,
        roi_weight=0.0,
        recency_weight=0.0,
        confidence_threshold=-10,
        context_builder=DUMMY_BUILDER,
    )

    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "good"

    engine.roi_tag_weights = {"high-ROI": -1.0, "bug-introduced": 1.0}
    ranked = engine._rank_records(records)
    assert ranked[0]["metadata"]["summary"] == "bad"


def test_prompt_memory_trainer_extracts_new_cues():
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    trainer = PromptMemoryTrainer(memory=object(), patch_db=object(), db_path=tmp.name)
    prompt = (
        "System: guidelines\nUser: do this\n- bullet\n```python\npass\n```"
    )
    feats = trainer._extract_style(prompt)
    assert feats["has_code"]
    assert feats["has_bullets"]
    assert feats["sections"] == ["system", "user"]


def test_prompt_memory_trainer_weights_success_by_roi_or_complexity():
    class Mem:
        def __init__(self):
            self.conn = sq3.connect(":memory:")
            self.conn.execute("CREATE TABLE interactions(prompt TEXT)")

    class PDB:
        def __init__(self):
            self.conn = sq3.connect(":memory:")
            self.conn.execute(
                "CREATE TABLE patch_history("
                "id INTEGER PRIMARY KEY, outcome TEXT, "
                "roi_before REAL, roi_after REAL, "
                "complexity_before REAL, complexity_after REAL)"
            )

    mem = Mem()
    pdb = PDB()
    pdb.conn.execute(
        "INSERT INTO patch_history("
        "id, outcome, roi_before, roi_after, complexity_before, complexity_after)"
        " VALUES (1, 'SUCCESS', 0.0, 2.0, 10.0, 5.0)"
    )
    pdb.conn.execute(
        "INSERT INTO patch_history("
        "id, outcome, roi_before, roi_after, complexity_before, complexity_after)"
        " VALUES (2, 'FAIL', 0.0, 0.0, 10.0, 5.0)"
    )
    mem.conn.execute("INSERT INTO interactions(prompt) VALUES ('PATCH:1\n# H')")
    mem.conn.execute("INSERT INTO interactions(prompt) VALUES ('PATCH:2\n# H')")
    mem.conn.commit()
    pdb.conn.commit()

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    trainer = PromptMemoryTrainer(memory=mem, patch_db=pdb, db_path=tmp.name)
    weights = trainer.train()
    hdr_key = json.dumps(["H"])
    assert weights["headers"][hdr_key] == pytest.approx(2 / 7)


def test_trim_tokens_with_tokenizer():
    pytest.importorskip("tiktoken")
    engine = PromptEngine(
        retriever=DummyRetriever([]),
        patch_retriever=DummyRetriever([]),
        context_builder=DUMMY_BUILDER,
    )
    text = "Hello, world!"
    assert engine._trim_tokens(text, 3) == "Hello, wo..."


def test_trim_tokens_without_tokenizer(monkeypatch):
    import prompt_engine as pe

    monkeypatch.setattr(pe, "_ENCODER", None)
    messages: list[str] = []
    monkeypatch.setattr(pe.logger, "warning", lambda msg: messages.append(msg))
    engine = PromptEngine(
        retriever=DummyRetriever([]),
        patch_retriever=DummyRetriever([]),
        context_builder=DUMMY_BUILDER,
    )
    text = "Hello, world!"
    assert engine._trim_tokens(text, 3) == text
    assert any("tiktoken" in m for m in messages)


def test_prompt_engine_refreshes_after_record(monkeypatch):
    class StubTrainer:

        def __init__(self):
            self.style_weights = {}

        def train(self):
            return self.style_weights

        def record(self, **_):
            self.style_weights = {
                "headers": {json.dumps(["H2", "F2"]): 1.0},
                "example_order": {json.dumps(["success", "failure"]): 1.0},
            }
            return True

        def save_weights(self, *_a, **_k):
            pass

    trainer = StubTrainer()
    engine = PromptEngine(trainer=trainer, context_builder=DUMMY_BUILDER)
    called: Dict[str, Dict[str, float]] | None = None

    def fake_load(summary=None, **kwargs):
        nonlocal called
        called = summary

    monkeypatch.setattr(engine, "_load_trained_config", fake_load)
    trainer.record(headers=["h"], example_order=["success"], tone="neutral", success=True)
    engine.after_patch_cycle()
    assert called == trainer.style_weights


def test_prompt_engine_uses_optimizer_preferences():
    class StubOptimizer:
        def __init__(self):
            self.calls = 0

        def select_format(self, module, action):
            return {
                "tone": "friendly",
                "structured_sections": ["overview"],
                "example_placement": "start",
            }

        def aggregate(self):
            self.calls += 1

    records = [_record(0.9, summary="ok", tests_passed=True, raroi=0.2)]
    opt = StubOptimizer()
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        optimizer=opt,
        optimizer_refresh_interval=1,
        confidence_threshold=-1.0,
        context_builder=DUMMY_BUILDER,
    )
    engine.build_prompt("task", context_builder=DUMMY_BUILDER)
    assert engine.tone == "friendly"
    assert engine.trained_structured_sections == ["overview"]
    assert engine.trained_example_placement == "start"
    assert opt.calls == 1


def test_build_prompt_trims_final_text():
    records = [_record(1.0, summary="good", tests_passed=True, ts=1)]
    engine = PromptEngine(
        patch_retriever=DummyRetriever(records),
        token_threshold=5,
        confidence_threshold=-1.0,
        context_builder=DUMMY_BUILDER,
    )
    long_context = "word " * 100
    prompt = engine.build_prompt("desc", context=long_context, context_builder=DUMMY_BUILDER)
    assert str(prompt).endswith("...")


def test_prompt_engine_includes_target_region_metadata(tmp_path):
    records = [_record(0.9, summary="ok", tests_passed=True)]
    engine = PromptEngine(
        retriever=DummyRetriever(records),
        patch_retriever=DummyRetriever(records),
        confidence_threshold=-1.0,
        context_builder=DUMMY_BUILDER,
    )
    lines = [
        "# Region func lines 3-5",
        "def func(a, b):",
        "# start",
        "    pass",
        "# end",
    ]
    context = "\n".join(lines)
    mod_path = tmp_path / "mod.py"  # path-ignore
    mod_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    region = TargetRegion(start_line=3, end_line=5, function="func", filename=str(mod_path))
    region.func_signature = "def func(a, b):"
    prompt = engine.build_prompt(
        "desc", context=context, target_region=region, context_builder=DUMMY_BUILDER
    )
    text = str(prompt)
    assert text.splitlines()[0].startswith(
        "Modify only lines 3-5 within function func"
    )
    assert "# start" in text and "# end" in text
    assert prompt.metadata["target_region"]["function"] == "func"
    assert prompt.metadata["target_region"]["signature"] == "def func(a, b):"
    assert prompt.metadata["target_region"]["original_lines"] == [
        "# start",
        "    pass",
        "# end",
    ]
    assert (
        prompt.metadata["target_region"]["original_snippet"]
        == "# start\n    pass\n# end"
    )


def test_diff_within_target_region_out_of_bounds(tmp_path):
    lines = ["one", "two", "three", "four"]
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    region = TargetRegion(start_line=2, end_line=3, function="f", filename=str(path))
    modified = lines[:]
    modified[0] = "ONE"  # change outside region
    assert not diff_within_target_region(lines, modified, region)


def test_diff_within_target_region_within_bounds(tmp_path):
    lines = ["one", "two", "three", "four"]
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    region = TargetRegion(start_line=2, end_line=3, function="f", filename=str(path))
    modified = lines[:]
    modified[1] = "TWO"  # change within region
    assert diff_within_target_region(lines, modified, region)
