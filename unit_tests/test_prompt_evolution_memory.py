import json
from pathlib import Path

import pytest

from llm_interface import Prompt
from prompt_evolution_memory import PromptEvolutionMemory


class DummyRetriever:
    def search(self, query: str, top_k: int = 5):
        return [
            {
                "score": 1.0,
                "metadata": {
                    "summary": "s",
                    "diff": "d",
                    "snippet": "code",
                    "tests_passed": True,
                    "raroi": 1.0,
                },
            }
        ]


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_log_prompt_records_success_and_failure(tmp_path: Path):
    success = tmp_path / "success.json"
    failure = tmp_path / "failure.json"
    logger = PromptEvolutionMemory(success_path=success, failure_path=failure)

    prompt = Prompt(system="sys", user="u", examples=["e"])
    logger.log(
        prompt,
        True,
        {"out": "ok"},
        {"roi": 1},
        format_meta={"fmt": "a"},
        module="m1",
        action="a1",
    )
    logger.log(
        prompt,
        False,
        {"out": "bad"},
        {"roi": -1},
        format_meta={"fmt": "b"},
        module="m1",
        action="a2",
    )

    s = read_lines(success)
    f = read_lines(failure)

    assert s and f
    assert s[0]["success"] is True
    assert f[0]["success"] is False
    assert s[0]["module"] == "m1"
    assert s[0]["action"] == "a1"
    assert f[0]["action"] == "a2"
    assert s[0]["prompt"]["user"] == "u"
    assert "prompt_text" in s[0]
    assert s[0]["roi"] == {"roi": 1}
    assert f[0]["result"]["out"] == "bad"


def test_optimizer_ranking_influences_prompt_engine(tmp_path: Path, monkeypatch):
    success = tmp_path / "success.json"
    failure = tmp_path / "failure.json"
    from prompt_optimizer import PromptOptimizer
    from prompt_engine import PromptEngine
    success.write_text(json.dumps({
        "module": "prompt_engine",
        "action": "build_prompt",
        "prompt": "A",
        "success": True,
        "roi": 2.0,
    }))
    failure.write_text(json.dumps({
        "module": "prompt_engine",
        "action": "build_prompt",
        "prompt": "B",
        "success": False,
        "roi": -1.0,
    }))

    def fake_extract(self, prompt: str):
        if prompt == "A":
            return {
                "tone": "cheerful",
                "header_set": ("h1",),
                "example_placement": "start",
                "has_code": False,
                "has_bullets": False,
            }
        return {
            "tone": "serious",
            "header_set": ("h2",),
            "example_placement": "end",
            "has_code": False,
            "has_bullets": False,
        }

    monkeypatch.setattr(PromptOptimizer, "_extract_features", fake_extract)
    opt = PromptOptimizer(success, failure, stats_path=tmp_path / "stats.json")

    import prompt_engine as pe

    monkeypatch.setattr(pe, "compress_snippets", lambda m: m)
    monkeypatch.setattr(pe, "audit_log_event", lambda *a, **k: None)

    engine = PromptEngine(
        retriever=DummyRetriever(),
        optimizer=opt,
        confidence_threshold=0.0,
        context_builder=object(),
    )
    engine.build_prompt("task", context_builder=engine.context_builder)

    assert engine.tone == "cheerful"
    assert "h1" in engine.last_metadata.get("structured_sections", [])


def test_optimizer_weighting_uses_roi(tmp_path: Path):
    success = tmp_path / "success.json"
    failure = tmp_path / "failure.json"
    from prompt_optimizer import PromptOptimizer
    entries = [
        {
            "module": "prompt_engine",
            "action": "build_prompt",
            "prompt": "# H\nExample",
            "success": True,
            "roi": 1.0,
            "coverage": 1.0,
        },
        {
            "module": "prompt_engine",
            "action": "build_prompt",
            "prompt": "# H\nExample",
            "success": True,
            "roi": 3.0,
            "coverage": 3.0,
        },
    ]
    success.write_text("\n".join(json.dumps(e) for e in entries))
    failure.write_text("")
    opt = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "stats.json",
        weight_by="coverage",
    )
    stat = next(iter(opt.stats.values()))
    expected = (1.0 * 1.0 + 3.0 * 3.0) / (1.0 + 3.0)
    assert stat.weighted_roi() == pytest.approx(expected)
