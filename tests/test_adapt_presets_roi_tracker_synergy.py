import menace_sandbox.environment_generator as eg
import menace_sandbox.roi_tracker as rt


def _tracker_with_synergy(values):
    t = rt.ROITracker()
    roi_pairs = [(0.0, 0.1), (0.1, 0.4), (0.4, 0.9)]
    for (before, after), v in zip(roi_pairs, values):
        t.update(
            before,
            after,
            metrics={
                "security_score": 70,
                "synergy_efficiency": v,
                "synergy_resilience": v,
                "synergy_antifragility": v,
            },
        )
    return t


def test_adapt_presets_synergy_positive(monkeypatch):
    tracker = _tracker_with_synergy([0.06, 0.07, 0.08])
    monkeypatch.setattr(
        tracker,
        "predict_synergy_metric",
        lambda name: 0.1 if name in {"efficiency", "resilience", "antifragility"} else 0.0,
    )
    presets = [{
        "CPU_LIMIT": "2",
        "MEMORY_LIMIT": "512Mi",
        "BANDWIDTH_LIMIT": "5Mbps",
        "MIN_BANDWIDTH": "1Mbps",
        "MAX_BANDWIDTH": "10Mbps",
        "THREAT_INTENSITY": 30,
    }]
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) <= 2
    assert eg._MEMORY_LIMITS.index(new[0]["MEMORY_LIMIT"]) <= eg._MEMORY_LIMITS.index("512Mi")
    assert eg._BANDWIDTHS.index(new[0]["BANDWIDTH_LIMIT"]) >= eg._BANDWIDTHS.index("5Mbps")
    assert new[0]["THREAT_INTENSITY"] > 30


def test_adapt_presets_synergy_negative(monkeypatch):
    tracker = _tracker_with_synergy([-0.06, -0.07, -0.08])
    monkeypatch.setattr(
        tracker,
        "predict_synergy_metric",
        lambda name: -0.1 if name in {"efficiency", "resilience", "antifragility"} else 0.0,
    )
    presets = [{
        "CPU_LIMIT": "2",
        "MEMORY_LIMIT": "512Mi",
        "BANDWIDTH_LIMIT": "10Mbps",
        "MIN_BANDWIDTH": "5Mbps",
        "MAX_BANDWIDTH": "50Mbps",
        "THREAT_INTENSITY": 50,
    }]
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) >= 2
    assert eg._MEMORY_LIMITS.index(new[0]["MEMORY_LIMIT"]) >= eg._MEMORY_LIMITS.index("512Mi")
    assert eg._BANDWIDTHS.index(new[0]["BANDWIDTH_LIMIT"]) <= eg._BANDWIDTHS.index("10Mbps")
    assert new[0]["THREAT_INTENSITY"] < 50


def test_adapt_presets_synergy_prediction(monkeypatch):
    tracker = _tracker_with_synergy([0.0, 0.0, 0.0])

    def fake_pred(name: str) -> float:
        mapping = {"efficiency": 0.3, "resilience": 0.3, "antifragility": 0.3}
        return mapping.get(name, 0.0)

    monkeypatch.setattr(tracker, "predict_synergy_metric", fake_pred)
    presets = [{
        "CPU_LIMIT": "2",
        "MEMORY_LIMIT": "512Mi",
        "BANDWIDTH_LIMIT": "5Mbps",
        "MIN_BANDWIDTH": "1Mbps",
        "MAX_BANDWIDTH": "10Mbps",
        "THREAT_INTENSITY": 30,
    }]
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) <= 2
    assert eg._MEMORY_LIMITS.index(new[0]["MEMORY_LIMIT"]) <= eg._MEMORY_LIMITS.index("512Mi")
    assert eg._BANDWIDTHS.index(new[0]["BANDWIDTH_LIMIT"]) >= eg._BANDWIDTHS.index("5Mbps")
    assert new[0]["THREAT_INTENSITY"] > 30


def test_synergy_roi_positive(monkeypatch, tracker_factory):
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", None, raising=False)
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", "")
    monkeypatch.setattr(eg, "AdaptivePresetAgent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()), raising=False)
    tracker = tracker_factory(
        roi=[0.2, 0.1, 0.0],
        metrics={
            "security_score": [70, 70, 70],
            "synergy_roi": [0.2, 0.3, 0.25],
        }
    )
    presets = [
        {"THREAT_INTENSITY": 30, "NETWORK_LATENCY_MS": 100, "MAX_BANDWIDTH": "10Mbps"}
    ]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 30
    assert new[0]["NETWORK_LATENCY_MS"] < 100
    assert eg._BANDWIDTHS.index(new[0]["MAX_BANDWIDTH"]) > eg._BANDWIDTHS.index("10Mbps")


def test_synergy_roi_negative(monkeypatch, tracker_factory):
    monkeypatch.setattr(eg.adapt_presets, "_rl_agent", None, raising=False)
    monkeypatch.setenv("SANDBOX_PRESET_RL_PATH", "")
    monkeypatch.setattr(eg, "AdaptivePresetAgent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()), raising=False)
    tracker = tracker_factory(
        roi=[0.2, 0.1, 0.0],
        metrics={
            "security_score": [70, 70, 70],
            "synergy_roi": [-0.3, -0.2, -0.25],
        }
    )
    presets = [
        {"THREAT_INTENSITY": 70, "NETWORK_LATENCY_MS": 50, "MAX_BANDWIDTH": "50Mbps"}
    ]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] < 70
    assert new[0]["NETWORK_LATENCY_MS"] > 50
    assert eg._BANDWIDTHS.index(new[0]["MAX_BANDWIDTH"]) < eg._BANDWIDTHS.index("50Mbps")
