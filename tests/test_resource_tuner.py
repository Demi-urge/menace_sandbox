from types import SimpleNamespace

from menace_sandbox.sandbox_runner.resource_tuner import ResourceTuner


def _make_tracker(deltas, cpu=0.9, mem=0.9):
    vals = [0.0]
    for d in deltas:
        vals.append(vals[-1] + d)
    metrics = [(cpu, mem, 0.0, 0.0, 0.0)] * len(vals)
    return SimpleNamespace(roi_history=vals, resource_metrics=metrics)


def test_tuner_scales_up():
    tracker = _make_tracker([0.1, 0.1, 0.1], cpu=0.9, mem=0.9)
    presets = [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "512Mi"}]
    tuner = ResourceTuner()
    out = tuner.adjust(tracker, presets)
    assert float(out[0]["CPU_LIMIT"]) > 1
    assert out[0]["MEMORY_LIMIT"] != "512Mi"


def test_tuner_scales_down():
    tracker = _make_tracker([-0.1, -0.1, -0.1], cpu=0.1, mem=0.1)
    presets = [{"CPU_LIMIT": "2", "MEMORY_LIMIT": "1024Mi"}]
    tuner = ResourceTuner()
    out = tuner.adjust(tracker, presets)
    assert float(out[0]["CPU_LIMIT"]) < 2
    assert out[0]["MEMORY_LIMIT"] != "1024Mi"
