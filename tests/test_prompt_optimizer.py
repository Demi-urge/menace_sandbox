import json
from pathlib import Path
from dynamic_path_router import resolve_path
from prompt_optimizer import PromptOptimizer


def test_prompt_optimizer_aggregates_missing_roi(tmp_path):
    log1 = Path(resolve_path(str(tmp_path / "log1.jsonl")))
    log2 = Path(resolve_path(str(tmp_path / "log2.jsonl")))
    log1.write_text(
        json.dumps(
            {
                "module": "m",
                "action": "a",
                "prompt": "# H\nExample\n```code```",
                "success": True,
                "roi": 2.0,
            }
        ) + "\n",
        encoding="utf-8",
    )
    log2.write_text(
        json.dumps(
            {
                "module": "m",
                "action": "a",
                "prompt": "plain text",
                "success": False,
            }
        ) + "\n",
        encoding="utf-8",
    )
    opt = PromptOptimizer(
        log1,
        log2,
        stats_path=Path(resolve_path(str(tmp_path / "stats.json"))),
    )
    assert len(opt.stats) == 2
    stats = list(opt.stats.values())
    stat_success = next(s for s in stats if s.success == 1)
    stat_fail = next(s for s in stats if s.success == 0)
    assert stat_success.total == 1
    assert stat_success.roi_sum == 2.0
    assert stat_fail.total == 1


def test_prompt_optimizer_applies_suggestion(tmp_path):
    log_path = Path(resolve_path(str(tmp_path / "log.jsonl")))
    log_path.write_text(
        json.dumps(
            {
                "module": "sample_module",
                "action": "build",
                "prompt": "do it",
                "success": True,
                "roi": 1.0,
            }
        ) + "\n",
        encoding="utf-8",
    )
    opt = PromptOptimizer(
        log_path,
        log_path,
        stats_path=Path(resolve_path(str(tmp_path / "stats.json"))),
    )

    class DummyPromptEngine:
        def __init__(self) -> None:
            self.tone = "negative"

    engine = DummyPromptEngine()
    suggestion = opt.suggest_format("sample_module", "build")[0]
    tone = suggestion.get("tone")
    if isinstance(tone, str):
        engine.tone = tone
    assert engine.tone == "neutral"
