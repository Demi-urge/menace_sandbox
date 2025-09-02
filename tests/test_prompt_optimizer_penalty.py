import json
from prompt_optimizer import PromptOptimizer

def test_failure_fingerprints_penalize(tmp_path):
    success = tmp_path / "success.jsonl"
    failure = tmp_path / "failure.jsonl"
    fp = tmp_path / "failure_fingerprints.jsonl"
    success.write_text(
        json.dumps({
            "module": "m",
            "action": "a",
            "prompt": "# H\nExample",
            "success": True,
            "roi": 1.0,
        }) + "\n",
        encoding="utf-8",
    )
    failure.write_text("", encoding="utf-8")
    fp.write_text(
        json.dumps({
            "filename": "m",
            "function_name": "a",
            "prompt_text": "# H\nExample",
        }) + "\n",
        encoding="utf-8",
    )
    opt = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "stats.json",
        failure_fingerprint_path=fp,
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
    assert stat.success == 1
    assert stat.total == 2
    assert stat.roi_sum == 0.0
