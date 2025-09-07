import json
from pathlib import Path

from prompt_engine import PromptEngine
from prompt_optimizer import PromptOptimizer
from prompt_types import Prompt


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
                    "roi_delta": 1.0,
                },
            }
        ]


def test_optimizer_ranking_and_influence(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "log.jsonl"
    entries = [
        {"module": "prompt_engine", "action": "build_prompt", "prompt": "A", "success": True, "roi": 2.0},
        {"module": "prompt_engine", "action": "build_prompt", "prompt": "B", "success": False, "roi": -1.0},
    ]
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

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
    opt = PromptOptimizer(log, log, stats_path=tmp_path / "stats.json")
    prefs = opt.select_format("prompt_engine", "build_prompt")
    assert prefs["tone"] == "cheerful"

    import prompt_engine as pe_mod

    monkeypatch.setattr(
        pe_mod,
        "compress_snippets",
        lambda m: {"diff": m.get("diff"), "snippet": m.get("snippet"), "test_log": m.get("test_log")},
    )
    monkeypatch.setattr(pe_mod, "audit_log_event", lambda *a, **k: None)

    engine = PromptEngine(
        retriever=DummyRetriever(),
        optimizer=opt,
        confidence_threshold=0.0,
        context_builder=object(),
    )
    prompt = engine.build_prompt("task", context_builder=engine.context_builder)
    assert engine.tone == "cheerful"
    assert "h1" in engine.last_metadata.get("structured_sections", [])
    assert isinstance(prompt, Prompt)
