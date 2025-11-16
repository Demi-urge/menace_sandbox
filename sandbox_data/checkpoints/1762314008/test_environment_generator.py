import sys
import random
import pytest

import environment_generator as eg
try:
    import menace.roi_tracker as rt
except Exception:  # pragma: no cover - fallback stub
    import types

    class _RT:
        def __init__(self, *a, **k):
            self.roi_history = []
            self.metrics_history = {"security_score": [70, 70, 70]}

        def update(self, before=0.0, after=0.0, *, metrics=None):
            self.roi_history.append(after)
            if metrics:
                for k2, v in metrics.items():
                    self.metrics_history.setdefault(k2, []).append(v)

        def diminishing(self):
            return 0.01

        def forecast(self):
            return 0.0, (0.0, 0.0)

        def forecast_synergy(self):
            val, _ = self.forecast()
            return val, (0.0, 0.0)

    rt = types.SimpleNamespace(ROITracker=_RT)
    sys.modules.setdefault("menace.roi_tracker", rt)


def test_generate_presets_count_and_keys():
    presets = eg.generate_presets(4)
    assert len(presets) == 4
    required = {"CPU_LIMIT", "NETWORK_LATENCY_MS", "SECURITY_LEVEL"}
    for preset in presets:
        assert required <= preset.keys()


def test_generate_presets_zero_count():
    assert eg.generate_presets(0) == []


def test_generate_presets_gpu_and_os(monkeypatch):
    monkeypatch.setattr(eg.random, "random", lambda: 0.0)
    presets = eg.generate_presets(1)
    assert "GPU_LIMIT" in presets[0] and "OS_TYPE" in presets[0]


def test_generate_presets_os_specific_fields(monkeypatch):
    monkeypatch.setattr(eg.random, "random", lambda: 0.0)

    orig_choice = eg.random.choice

    def fake_choice(seq):
        if seq is eg._ALT_OS_TYPES:
            return "windows"
        return orig_choice(seq)

    monkeypatch.setattr(eg.random, "choice", fake_choice)

    presets = eg.generate_presets(1)
    p = presets[0]
    assert p.get("OS_TYPE") == "windows"
    assert p.get("CONTAINER_IMAGE")
    assert p.get("VM_SETTINGS", {}).get("windows_image")


def test_generate_presets_missing_optional(monkeypatch):
    # Remove stubbed optional modules if present to simulate absent packages
    for name in ["pulp", "pandas", "sqlalchemy"]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    presets = eg.generate_presets(1)
    assert len(presets) == 1 and isinstance(presets[0], dict)


def test_generate_presets_multiple_failures(monkeypatch):
    monkeypatch.setattr(eg.random, "random", lambda: 0.0)
    monkeypatch.setattr(eg.random, "randint", lambda a, b: 2)
    monkeypatch.setattr(eg.random, "sample", lambda pop, k: pop[:k])
    presets = eg.generate_presets(1)
    fm = presets[0].get("FAILURE_MODES")
    assert isinstance(fm, list) and len(fm) == 2


def test_generate_presets_user_misuse(monkeypatch):
    monkeypatch.setattr(eg, "_select_failures", lambda: ["user_misuse"])
    presets = eg.generate_presets(1)
    p = presets[0]
    assert p.get("FAILURE_MODES") == "user_misuse"
    assert p.get("SCENARIO_NAME") == "user_misuse"
    assert p.get("SANDBOX_STUB_STRATEGY") == "misuse"
    assert p.get("INVALID_CONFIG") is True
    assert p.get("INVALID_PARAM_TYPES")
    assert p.get("UNEXPECTED_API_CALLS")


def test_generate_presets_hostile_input(monkeypatch):
    monkeypatch.setattr(eg, "_select_failures", lambda: ["hostile_input"])
    presets = eg.generate_presets(1)
    p = presets[0]
    assert p.get("SCENARIO_NAME") == "hostile_input"
    assert p.get("PAYLOAD_INDICATOR")
    assert p.get("MALICIOUS_DATA")


def test_generate_presets_high_latency_auto(monkeypatch):
    monkeypatch.setattr(eg, "_select_failures", lambda: [])

    def fake_choice(seq):
        if seq is eg._LATENCIES:
            return 500
        if seq is eg._JITTERS:
            return 50
        if seq is eg._PACKET_LOSS:
            return 0.1
        if seq is eg._API_LATENCIES:
            return 1000
        return seq[0]

    monkeypatch.setattr(eg.random, "choice", fake_choice)
    presets = eg.generate_presets(1)
    p = presets[0]
    assert p.get("SCENARIO_NAME") == "high_latency_api"
    assert p.get("FAILURE_MODES") == "api_latency"
    assert p.get("API_LATENCY_MS") == 1000


def test_generate_presets_profiles():
    presets = eg.generate_presets(
        profiles=[
            "high_latency_api",
            "hostile_input",
            "user_misuse",
            "concurrency_spike",
            "schema_drift",
            "flaky_upstream",
        ]
    )
    names = {p.get("SCENARIO_NAME") for p in presets}
    expected = {
        "high_latency_api",
        "hostile_input",
        "user_misuse",
        "concurrency_spike",
        "schema_drift",
        "flaky_upstream",
    }
    assert expected <= names
    hl = next(p for p in presets if p.get("SCENARIO_NAME") == "high_latency_api")
    assert hl.get("NETWORK_LATENCY_MS") == 500
    assert hl.get("NETWORK_JITTER_MS") == 50
    assert hl.get("PACKET_LOSS") == 0.1
    assert hl.get("API_LATENCY_MS") == 1000
    hi = next(p for p in presets if p.get("SCENARIO_NAME") == "hostile_input")
    assert hi.get("PAYLOAD_INDICATOR")
    assert hi.get("MALICIOUS_DATA")
    um = next(p for p in presets if p.get("SCENARIO_NAME") == "user_misuse")
    assert um.get("INVALID_CONFIG") is True
    assert um.get("INVALID_PARAM_TYPES")
    assert um.get("UNEXPECTED_API_CALLS")
    cs = next(p for p in presets if p.get("SCENARIO_NAME") == "concurrency_spike")
    assert cs.get("CPU_SPIKE")
    assert cs.get("MAX_THREADS") == 200
    assert cs.get("CONCURRENCY_LEVEL")
    sd = next(p for p in presets if p.get("SCENARIO_NAME") == "schema_drift")
    assert sd.get("SCHEMA_MISMATCHES") is not None
    assert sd.get("SCHEMA_CHECKS") is not None
    assert sd.get("SANDBOX_STUB_STRATEGY") == "legacy_schema"
    fu = next(p for p in presets if p.get("SCENARIO_NAME") == "flaky_upstream")
    assert fu.get("UPSTREAM_FAILURES") is not None
    assert fu.get("UPSTREAM_REQUESTS") is not None
    assert fu.get("SANDBOX_STUB_STRATEGY") == "flaky_upstream"


def test_generate_presets_new_scenario_keys():
    presets = eg.generate_presets(
        profiles=[
            "hostile_input",
            "user_misuse",
            "concurrency_spike",
            "schema_drift",
            "flaky_upstream",
        ]
    )
    by_name = {p.get("SCENARIO_NAME"): p for p in presets}
    hi = by_name["hostile_input"]
    um = by_name["user_misuse"]
    cs = by_name["concurrency_spike"]
    sd = by_name["schema_drift"]
    fu = by_name["flaky_upstream"]
    assert hi.get("THREAT_INTENSITY") is not None
    assert um.get("THREAT_INTENSITY") is not None
    assert cs.get("THREAT_INTENSITY") is not None
    assert sd.get("THREAT_INTENSITY") is not None
    assert fu.get("THREAT_INTENSITY") is not None
    assert cs.get("THREAD_BURST") is not None
    assert cs.get("ASYNC_TASK_BURST") is not None
    assert sd.get("SCHEMA_MISMATCHES") is not None
    assert fu.get("UPSTREAM_FAILURES") is not None


def test_generate_presets_profile_severity_levels():
    low = eg.generate_presets(profiles=["high_latency_api"], severity="low")[0]
    high = eg.generate_presets(profiles=["high_latency_api"], severity="high")[0]
    assert low["NETWORK_LATENCY_MS"] < high["NETWORK_LATENCY_MS"]
    assert low["THREAT_INTENSITY"] < high["THREAT_INTENSITY"]


def test_generate_presets_mixed_severity_profiles(monkeypatch):
    monkeypatch.setattr(eg.random, "random", lambda: 1.0)
    presets = eg.generate_presets(
        profiles=["high_latency_api", "concurrency_spike"],
        severity={"high_latency_api": "low", "concurrency_spike": "high"},
    )
    by_name = {p.get("SCENARIO_NAME"): p for p in presets}
    assert by_name["high_latency_api"]["NETWORK_LATENCY_MS"] == 200
    assert by_name["concurrency_spike"]["THREAD_BURST"] == 50


def test_generate_combined_presets_deterministic():
    presets = eg.generate_combined_presets([["hostile_input", "concurrency_spike"]])
    assert len(presets) == 1
    p = presets[0]
    assert p["SCENARIO_NAME"] == "hostile_input+concurrency_spike"
    assert p["THREAD_BURST"] == 50
    assert p["ASYNC_TASK_BURST"] == 100
    assert p["SANDBOX_STUB_STRATEGY"] == "hostile"
    assert p["FAILURE_MODES"] == ["hostile_input", "concurrency_spike", "cpu_spike"]


def test_generate_presets_combined_profiles(monkeypatch):
    monkeypatch.setattr(eg, "_select_failures", lambda: [])
    presets = eg.generate_presets(count=1, profiles=[["hostile_input", "concurrency_spike"]])
    assert len(presets) == 1
    p = presets[0]
    assert p["SCENARIO_NAME"] == "hostile_input+concurrency_spike"
    assert p["THREAD_BURST"] == 50
    assert p["SANDBOX_STUB_STRATEGY"] == "hostile"
    assert p["FAILURE_MODES"] == ["hostile_input", "concurrency_spike", "cpu_spike"]


class _DummyTracker:
    def __init__(
        self,
        scores,
        synergy_roi=None,
        synergy_sec=None,
        synergy_eff=None,
        synergy_af=None,
        synergy_res=None,
        synergy_ent=None,
        synergy_flex=None,
        synergy_energy=None,
        synergy_safe=None,
        synergy_adapt=None,
        synergy_risk=None,
        synergy_recovery=None,
        synergy_disc=None,
        synergy_gpu=None,
        synergy_cpu=None,
        synergy_mem=None,
        synergy_long_lucr=None,
        synergy_netlat=None,
        synergy_tp=None,
    ):
        self.metrics_history = {"security_score": scores}
        if synergy_roi is not None:
            self.metrics_history["synergy_roi"] = synergy_roi
        if synergy_sec is not None:
            self.metrics_history["synergy_security_score"] = synergy_sec
        if synergy_eff is not None:
            self.metrics_history["synergy_efficiency"] = synergy_eff
        if synergy_af is not None:
            self.metrics_history["synergy_antifragility"] = synergy_af
        if synergy_res is not None:
            self.metrics_history["synergy_resilience"] = synergy_res
        if synergy_ent is not None:
            self.metrics_history["synergy_shannon_entropy"] = synergy_ent
        if synergy_flex is not None:
            self.metrics_history["synergy_flexibility"] = synergy_flex
        if synergy_energy is not None:
            self.metrics_history["synergy_energy_consumption"] = synergy_energy
        if synergy_safe is not None:
            self.metrics_history["synergy_safety_rating"] = synergy_safe
        if synergy_adapt is not None:
            self.metrics_history["synergy_adaptability"] = synergy_adapt
        if synergy_risk is not None:
            self.metrics_history["synergy_risk_index"] = synergy_risk
        if synergy_recovery is not None:
            self.metrics_history["synergy_recovery_time"] = synergy_recovery
        if synergy_disc is not None:
            self.metrics_history["synergy_discrepancy_count"] = synergy_disc
        if synergy_gpu is not None:
            self.metrics_history["synergy_gpu_usage"] = synergy_gpu
        if synergy_cpu is not None:
            self.metrics_history["synergy_cpu_usage"] = synergy_cpu
        if synergy_mem is not None:
            self.metrics_history["synergy_memory_usage"] = synergy_mem
        if synergy_long_lucr is not None:
            self.metrics_history["synergy_long_term_lucrativity"] = synergy_long_lucr
        if synergy_netlat is not None:
            self.metrics_history["synergy_network_latency"] = synergy_netlat
        if synergy_tp is not None:
            self.metrics_history["synergy_throughput"] = synergy_tp


class _ResourceTracker:
    def __init__(self, roi, synergy=None, scores=None):
        self.roi_history = roi
        self.metrics_history = {"security_score": scores or [70, 70, 70]}
        if synergy is not None:
            self.metrics_history["synergy_roi"] = synergy

    def diminishing(self):
        return 0.01


class _PredictTracker:
    def __init__(self, *, roi_history=None, **preds):
        self.metrics_history = {"security_score": [70, 70, 70]}
        if roi_history is None:
            self.roi_history = [0.0]
        elif isinstance(roi_history, (list, tuple)):
            self.roi_history = list(roi_history)
        else:
            self.roi_history = [float(roi_history)]
        self._preds = preds

    def predict_synergy(self):
        return float(self._preds.get("roi", 0.0))

    def predict_synergy_metric(self, name):
        return float(self._preds.get(name, 0.0))

    def diminishing(self):
        return 0.01


class _EdgeTracker:
    def __init__(self, *, metrics=None, preds=None):
        self.metrics_history = {"security_score": [70, 70, 70]}
        if metrics:
            self.metrics_history.update(metrics)
        self._preds = preds or {}
        self.roi_history = [0.0]

    def predict_synergy_metric(self, name):
        return float(self._preds.get(name, 0.0))

    def diminishing(self):
        return 0.01


def test_adapt_presets_increase():
    presets = [{"THREAT_INTENSITY": 30}, {"THREAT_INTENSITY": 30}]
    tracker = _DummyTracker([85, 90, 88])
    new = eg.adapt_presets(tracker, presets)
    assert all(p["THREAT_INTENSITY"] > 30 for p in new)


def test_adapt_presets_decrease():
    presets = [{"THREAT_INTENSITY": 70}, {"THREAT_INTENSITY": 90}]
    tracker = _DummyTracker([40, 45, 42])
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] < 70
    assert new[1]["THREAT_INTENSITY"] < 90


def test_synergy_roi_adjustments():
    presets = [{"THREAT_INTENSITY": 30}, {"THREAT_INTENSITY": 30}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_roi=[0.2, 0.3, 0.25]
    )
    new = eg.adapt_presets(tracker, presets)
    assert all(p["THREAT_INTENSITY"] > 30 for p in new)


def test_synergy_security_level():
    presets = [{"SECURITY_LEVEL": 2}, {"SECURITY_LEVEL": 3}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_sec=[6.0, 5.5, 6.5]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["SECURITY_LEVEL"] > 2
    assert new[1]["SECURITY_LEVEL"] > 3


def test_synergy_risk_index_security_level():
    presets = [{"SECURITY_LEVEL": 2}, {"SECURITY_LEVEL": 3}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_risk=[6.0, 5.5, 6.5]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["SECURITY_LEVEL"] > 2
    assert new[1]["SECURITY_LEVEL"] > 3


def test_synergy_low_values():
    presets = [
        {"THREAT_INTENSITY": 50, "SECURITY_LEVEL": 4},
        {"THREAT_INTENSITY": 50, "SECURITY_LEVEL": 4},
    ]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_roi=[-0.3, -0.2, -0.25], synergy_sec=[-6, -5, -7]
    )
    new = eg.adapt_presets(tracker, presets)
    assert all(p["THREAT_INTENSITY"] < 50 for p in new)
    assert all(p["SECURITY_LEVEL"] < 4 for p in new)


def test_resource_scaling_stagnation():
    presets = [
        {
            "CPU_LIMIT": "1",
            "MEMORY_LIMIT": "256Mi",
            "BANDWIDTH_LIMIT": "5Mbps",
            "MIN_BANDWIDTH": "1Mbps",
            "MAX_BANDWIDTH": "10Mbps",
        }
    ]
    tracker = _ResourceTracker([0.0, 0.05, 0.055, 0.056])
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["CPU_LIMIT"] != "1"
    assert new[0]["MEMORY_LIMIT"] != "256Mi"
    assert new[0]["BANDWIDTH_LIMIT"] != "5Mbps"


def test_resource_scaling_improving():
    presets = [
        {
            "CPU_LIMIT": "2",
            "MEMORY_LIMIT": "512Mi",
            "BANDWIDTH_LIMIT": "10Mbps",
            "MIN_BANDWIDTH": "5Mbps",
            "MAX_BANDWIDTH": "50Mbps",
        }
    ]
    tracker = _ResourceTracker([0.0, 0.2, 0.35, 0.55])
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) < 2
    assert new[0]["MEMORY_LIMIT"] == "256Mi"
    assert new[0]["BANDWIDTH_LIMIT"] == "5Mbps"


def test_synergy_network_adjustment():
    presets = [{"NETWORK_LATENCY_MS": 100, "MAX_BANDWIDTH": "10Mbps"}]
    tracker = _ResourceTracker([0.0, 0.0], synergy=[0.2, 0.3, 0.25])
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["NETWORK_LATENCY_MS"] < 100
    assert new[0]["MAX_BANDWIDTH"] != "10Mbps"


def test_synergy_efficiency_adjusts_resources():
    presets = [{"CPU_LIMIT": "2", "MEMORY_LIMIT": "512Mi"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_eff=[0.1, 0.08, 0.09]
    )
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) < 2
    assert new[0]["MEMORY_LIMIT"] != "512Mi"


def test_generate_presets_synergy_efficiency_negative():
    tracker = _DummyTracker([70, 70, 70], synergy_eff=[-0.2, -0.3, -0.4])
    random.seed(1)
    base = eg.generate_presets(1)[0]
    random.seed(1)
    new = eg.generate_presets(1, tracker=tracker)[0]
    assert new["NETWORK_LATENCY_MS"] >= base["NETWORK_LATENCY_MS"]
    assert eg._BANDWIDTHS.index(new["BANDWIDTH_LIMIT"]) <= eg._BANDWIDTHS.index(base["BANDWIDTH_LIMIT"])


def test_generate_presets_synergy_efficiency_positive():
    tracker = _DummyTracker([70, 70, 70], synergy_eff=[0.2, 0.25, 0.3])
    random.seed(2)
    base = eg.generate_presets(1)[0]
    random.seed(2)
    new = eg.generate_presets(1, tracker=tracker)[0]
    assert new["NETWORK_LATENCY_MS"] <= base["NETWORK_LATENCY_MS"]
    assert eg._BANDWIDTHS.index(new["BANDWIDTH_LIMIT"]) >= eg._BANDWIDTHS.index(base["BANDWIDTH_LIMIT"])


def test_generate_presets_network_latency_history():
    tracker = _DummyTracker([70, 70, 70], synergy_netlat=[2.0, 2.5, 2.0])
    random.seed(3)
    base = eg.generate_presets(1)[0]
    random.seed(3)
    new = eg.generate_presets(1, tracker=tracker)[0]
    assert new["NETWORK_LATENCY_MS"] >= base["NETWORK_LATENCY_MS"]


def test_generate_presets_throughput_history():
    tracker = _DummyTracker([70, 70, 70], synergy_tp=[-6.0, -7.0, -5.0])
    random.seed(4)
    base = eg.generate_presets(1)[0]
    random.seed(4)
    new = eg.generate_presets(1, tracker=tracker)[0]
    assert eg._BANDWIDTHS.index(new["MAX_BANDWIDTH"]) <= eg._BANDWIDTHS.index(base["MAX_BANDWIDTH"])


def test_synergy_adaptability_adjusts_resources():
    presets = [{"CPU_LIMIT": "2", "MEMORY_LIMIT": "512Mi"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_adapt=[0.06, 0.07, 0.05]
    )
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) < 2
    assert new[0]["MEMORY_LIMIT"] != "512Mi"


def test_synergy_antifragility_adjusts_threat():
    presets = [{"THREAT_INTENSITY": 30}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_af=[0.06, 0.07, 0.05]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 30


def test_synergy_safety_rating_adjusts_threat():
    presets = [{"THREAT_INTENSITY": 30}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_safe=[6.0, 6.5, 6.2]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 30


def test_synergy_resilience_bandwidth():
    presets = [
        {
            "BANDWIDTH_LIMIT": "5Mbps",
            "MIN_BANDWIDTH": "1Mbps",
            "MAX_BANDWIDTH": "10Mbps",
        }
    ]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_res=[0.07, 0.08, 0.09]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["BANDWIDTH_LIMIT"] != "5Mbps"
    assert new[0]["MAX_BANDWIDTH"] != "10Mbps"


def test_synergy_entropy_cpu_increase():
    presets = [{"CPU_LIMIT": "1"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_ent=[0.1, 0.12, 0.09]
    )
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) > 1


def test_synergy_flexibility_memory_reduce():
    presets = [{"MEMORY_LIMIT": "512Mi"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_flex=[0.1, 0.1, 0.1]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["MEMORY_LIMIT"] != "512Mi"


def test_synergy_energy_consumption_bandwidth_reduce():
    presets = [
        {
            "BANDWIDTH_LIMIT": "10Mbps",
            "MIN_BANDWIDTH": "5Mbps",
            "MAX_BANDWIDTH": "50Mbps",
        }
    ]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_energy=[0.1, 0.15, 0.12]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["BANDWIDTH_LIMIT"] != "10Mbps"


def test_synergy_recovery_time_threat_decrease():
    presets = [{"THREAT_INTENSITY": 50}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_recovery=[0.2, 0.15, 0.18]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] < 50


def test_synergy_recovery_time_threat_increase():
    presets = [{"THREAT_INTENSITY": 20}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_recovery=[-0.2, -0.25, -0.22]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 20


def test_prediction_efficiency_adjusts_resources():
    presets = [{"CPU_LIMIT": "2", "MEMORY_LIMIT": "512Mi"}]
    tracker = _PredictTracker(efficiency=0.1)
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) < 2
    assert new[0]["MEMORY_LIMIT"] != "512Mi"


def test_prediction_resilience_bandwidth():
    presets = [
        {
            "BANDWIDTH_LIMIT": "5Mbps",
            "MIN_BANDWIDTH": "1Mbps",
            "MAX_BANDWIDTH": "10Mbps",
        }
    ]
    tracker = _PredictTracker(resilience=0.1)
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["BANDWIDTH_LIMIT"] != "5Mbps"
    assert new[0]["MAX_BANDWIDTH"] != "10Mbps"


def test_prediction_synergy_roi_network():
    presets = [{"NETWORK_LATENCY_MS": 100, "MAX_BANDWIDTH": "10Mbps"}]
    tracker = _PredictTracker(roi=0.2)
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["NETWORK_LATENCY_MS"] < 100
    assert new[0]["MAX_BANDWIDTH"] != "10Mbps"


def test_adapt_presets_synergy_prediction_increases_threat(monkeypatch):
    tracker = rt.ROITracker()
    for r in [0.1, 0.2, 0.1]:
        tracker.update(0.0, r, metrics={"security_score": 90, "synergy_roi": 0.5 * r})
    monkeypatch.setattr(tracker, "forecast", lambda: (0.5, (0.0, 0.0)))

    presets = [{"THREAT_INTENSITY": 30}]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 30


def test_adapt_presets_synergy_prediction_no_change(monkeypatch):
    tracker = rt.ROITracker()
    for r in [0.1, 0.2, 0.1]:
        tracker.update(0.0, r, metrics={"security_score": 70, "synergy_roi": 0.5 * r})
    monkeypatch.setattr(tracker, "forecast", lambda: (0.1, (0.0, 0.0)))

    presets = [{"THREAT_INTENSITY": 30}]
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] == 30


def test_synergy_discrepancy_increases_threat():
    presets = [{"THREAT_INTENSITY": 30}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_disc=[2, 3, 4]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 30


def test_synergy_gpu_usage_scales_gpu_limit():
    presets = [{"GPU_LIMIT": "1"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_gpu=[0.1, 0.1, 0.1]
    )
    new = eg.adapt_presets(tracker, presets)
    assert int(new[0]["GPU_LIMIT"]) > 1


def test_synergy_cpu_usage_scales_cpu_limit():
    presets = [{"CPU_LIMIT": "1"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_cpu=[0.1, 0.1, 0.1]
    )
    new = eg.adapt_presets(tracker, presets)
    assert float(new[0]["CPU_LIMIT"]) > 1


def test_synergy_memory_usage_scales_memory_limit():
    presets = [{"MEMORY_LIMIT": "512Mi"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_mem=[0.1, 0.1, 0.1]
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["MEMORY_LIMIT"] != "512Mi"


def test_synergy_long_term_lucrativity_scales_resources():
    presets = [{"GPU_LIMIT": "1", "DISK_LIMIT": "512Mi"}]
    tracker = _DummyTracker(
        [70, 70, 70], synergy_long_lucr=[0.1, 0.1, 0.1]
    )
    new = eg.adapt_presets(tracker, presets)
    assert int(new[0]["GPU_LIMIT"]) > 1
    assert new[0]["DISK_LIMIT"] != "512Mi"


def test_synergy_maintainability_prediction_conflict_cpu_down():
    presets = [{"CPU_LIMIT": "2"}]
    tracker = _EdgeTracker(
        metrics={"synergy_maintainability": [-0.1, -0.1, -0.2]},
        preds={"maintainability": 0.7},
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["CPU_LIMIT"] == "1"


def test_synergy_revenue_prediction_conflict_memory_up():
    presets = [{"MEMORY_LIMIT": "256Mi"}]
    tracker = _EdgeTracker(
        metrics={"synergy_revenue": [-0.1, -0.1, -0.1]},
        preds={"revenue": 0.9},
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["MEMORY_LIMIT"] == "512Mi"


def test_synergy_throughput_prediction_overrides_history():
    presets = [{"MAX_BANDWIDTH": "10Mbps", "MIN_BANDWIDTH": "5Mbps"}]
    tracker = _EdgeTracker(
        metrics={"synergy_throughput": [-6, -6, -6]},
        preds={"throughput": 40},
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["MAX_BANDWIDTH"] == "50Mbps"
    assert new[0]["MIN_BANDWIDTH"] == "10Mbps"


def test_synergy_code_quality_prediction_overrides_history():
    presets = [{"THREAT_INTENSITY": 50}]
    tracker = _EdgeTracker(
        metrics={"synergy_code_quality": [-0.3, -0.3, -0.3]},
        preds={"code_quality": 1.2},
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["THREAT_INTENSITY"] > 50


def test_synergy_network_latency_prediction_cancels_increase():
    presets = [{"NETWORK_LATENCY_MS": 100}]
    tracker = _EdgeTracker(
        metrics={"synergy_network_latency": [2, 2, 2]},
        preds={"network_latency": -5},
    )
    new = eg.adapt_presets(tracker, presets)
    assert new[0]["NETWORK_LATENCY_MS"] == 100


def test_infer_profiles_from_ast(tmp_path):
    mod = tmp_path / "sample_mod.py"  # path-ignore
    mod.write_text(
        "import network_utils\n"
        "from threading import Thread\n"
        "ENABLE_AUTH = True\n"
        "@cache_response\n"
        "def handler():\n"
        "    pass\n"
    )
    from environment_generator import infer_profiles_from_ast

    profiles = infer_profiles_from_ast(str(mod))
    assert set(profiles) == {
        "high_latency_api",
        "concurrency_spike",
        "user_misuse",
        "hostile_input",
        "schema_drift",
        "flaky_upstream",
    }


def test_suggest_profiles_merges_ast_and_name(tmp_path):
    mod = tmp_path / "auth_helper.py"  # path-ignore
    mod.write_text(
        "import threading\n"
        "@cache_response\n"
        "def run():\n    pass\n"
    )
    from environment_generator import suggest_profiles_for_module

    profiles = suggest_profiles_for_module(str(mod))
    assert set(profiles) == {
        "high_latency_api",
        "concurrency_spike",
        "user_misuse",
        "hostile_input",
    }

