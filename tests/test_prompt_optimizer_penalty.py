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
        })
        + "\n"
        + json.dumps({
            "filename": "m",
            "function_name": "a",
            "prompt_text": "# H\nExample",
        })
        + "\n",
        encoding="utf-8",
    )
    opt = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "stats.json",
        failure_fingerprints_path=fp,
        fingerprint_threshold=1,
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
    assert stat.success == 0
    assert stat.total == 1
    assert stat.roi_sum == 1.0


def test_failure_fingerprints_reduce_score(tmp_path):
    success = tmp_path / "success.jsonl"
    failure = tmp_path / "failure.jsonl"
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
    opt_base = PromptOptimizer(success, failure, stats_path=tmp_path / "s1.json")
    score_base = opt_base.stats[key].score()
    fp = tmp_path / "failure_fingerprints.jsonl"
    fp.write_text(
        json.dumps({
            "filename": "m",
            "function_name": "a",
            "prompt_text": "# H\nExample",
        })
        + "\n"
        + json.dumps({
            "filename": "m",
            "function_name": "a",
            "prompt_text": "# H\nExample",
        })
        + "\n",
        encoding="utf-8",
    )
    opt_pen = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "s2.json",
        failure_fingerprints_path=fp,
        fingerprint_threshold=1,
    )
    score_pen = opt_pen.stats[key].score()
    assert score_pen < score_base
