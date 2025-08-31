import json

import prompt_memory_trainer as pmt


def test_prompt_engine_applies_trained_headers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pmt.json = json
    trainer = pmt.PromptMemoryTrainer()
    trainer.record(
        tone="neutral",
        headers=["H1", "H2"],
        example_order=["success", "failure"],
        success=True,
    )
    trainer.record(
        tone="neutral",
        headers=["X", "Y"],
        example_order=["failure", "success"],
        success=False,
    )
    summary = trainer.train()
    assert summary["headers"][json.dumps(["H1", "H2"])] == 1.0
    assert summary["example_order"][json.dumps(["success", "failure"])] == 1.0

    import prompt_engine as pe

    pe.compress_snippets = lambda meta: {}
    engine = pe.PromptEngine(confidence_threshold=0.5)
    patches = [
        {
            "metadata": {"summary": "good", "tests_passed": True, "ts": 1.0},
            "weighted_score": 1.0,
        },
        {
            "metadata": {
                "summary": "bad",
                "outcome": "fail",
                "tests_passed": False,
                "ts": 2.0,
            },
            "weighted_score": 1.0,
        },
    ]
    engine.build_snippets(patches)
    assert engine.trained_headers == ["H1", "H2"]
    assert engine.last_metadata["headers"][0] == "H1"
    assert engine.last_metadata["example_order"] == ["success", "failure"]
