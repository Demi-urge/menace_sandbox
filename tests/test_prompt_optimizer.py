import json

from prompt_optimizer import PromptOptimizer


def test_prompt_optimizer_aggregates_missing_roi(tmp_path):
    log1 = tmp_path / "log1.jsonl"
    log2 = tmp_path / "log2.jsonl"
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
    opt = PromptOptimizer(log1, log2, stats_path=tmp_path / "stats.json")
    key1 = ("m", "a", "neutral", 1, 1, True, False)
    key2 = ("m", "a", "neutral", 0, 0, False, False)
    assert key1 in opt.stats and key2 in opt.stats
    stat1 = opt.stats[key1]
    stat2 = opt.stats[key2]
    assert (stat1.success, stat1.total, stat1.roi_success, stat1.roi_total) == (
        1,
        1,
        2.0,
        2.0,
    )
    assert (stat2.success, stat2.total, stat2.roi_success, stat2.roi_total) == (
        0,
        1,
        0.0,
        1.0,
    )


def test_prompt_optimizer_applies_suggestion(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "module": "visual_agent",
                "action": "build",
                "prompt": "do it",
                "success": True,
                "roi": 1.0,
            }
        ) + "\n",
        encoding="utf-8",
    )
    opt = PromptOptimizer(log_path, log_path, stats_path=tmp_path / "stats.json")

    class DummyPromptEngine:
        def __init__(self) -> None:
            self.tone = "negative"

    engine = DummyPromptEngine()
    suggestion = opt.suggest_format("visual_agent", "build")
    tone = suggestion.get("tone")
    if isinstance(tone, str):
        engine.tone = tone
    assert engine.tone == "neutral"
