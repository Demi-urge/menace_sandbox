import json

from menace_sandbox.foresight_tracker import ForesightTracker


class DummyROITracker:
    def __init__(self, deltas):
        self._deltas = iter(deltas)
        self.raroi_history = [0.0]
        self.confidence_history = [0.0]
        self.metrics_history = {"synergy_resilience": [0.0]}

    def next_delta(self):
        delta = next(self._deltas)
        self.raroi_history.append(self.raroi_history[-1] + delta / 2.0)
        return delta

    def scenario_degradation(self):
        return 0.0


class MiniSelfImprovementEngine:
    def __init__(self, tracker, foresight_tracker):
        self.tracker = tracker
        self.foresight_tracker = foresight_tracker
        self.workflow_ready = False

    def run_cycle(self, workflow_id="wf"):
        delta = self.tracker.next_delta()
        raroi_delta = self.tracker.raroi_history[-1] - self.tracker.raroi_history[-2]
        confidence = self.tracker.confidence_history[-1]
        resilience = self.tracker.metrics_history["synergy_resilience"][-1]
        scenario_deg = self.tracker.scenario_degradation()
        self.foresight_tracker.record_cycle_metrics(
            workflow_id,
            {
                "roi_delta": float(delta),
                "raroi_delta": float(raroi_delta),
                "confidence": float(confidence),
                "resilience": float(resilience),
                "scenario_degradation": float(scenario_deg),
            },
            compute_stability=True,
        )

    def attempt_promotion(self, workflow_id="wf"):
        risk = self.foresight_tracker.predict_roi_collapse(workflow_id)
        if risk.get("risk") == "Immediate collapse risk" or risk.get(
            "brittle"
        ):
            self.workflow_ready = False
        else:
            self.workflow_ready = True


def test_run_cycle_records_and_stability():
    ft = ForesightTracker(max_cycles=3, volatility_threshold=5.0)
    tracker = DummyROITracker([1.0, 2.0, 3.0, 0.0])
    eng = MiniSelfImprovementEngine(tracker, ft)

    for _ in range(3):
        eng.run_cycle()
    # initial positive trend
    assert ft.is_stable("wf")
    assert all("stability" in entry for entry in ft.history["wf"])

    eng.run_cycle()  # negative slope but low volatility
    history = ft.history["wf"]
    assert len(history) == 3
    assert [entry["roi_delta"] for entry in history] == [2.0, 3.0, 0.0]
    assert [entry["raroi_delta"] for entry in history] == [1.0, 1.5, 0.0]
    assert not ft.is_stable("wf")


def test_is_stable_reacts_to_high_volatility():
    ft = ForesightTracker(max_cycles=3, volatility_threshold=0.5)
    tracker = DummyROITracker([1.0, 5.0, 9.0])
    eng = MiniSelfImprovementEngine(tracker, ft)

    for _ in range(3):
        eng.run_cycle()
    assert not ft.is_stable("wf")


def test_metrics_persist_through_save_load(tmp_path):
    ft = ForesightTracker(max_cycles=3)
    tracker1 = DummyROITracker([1.0])
    eng1 = MiniSelfImprovementEngine(tracker1, ft)
    eng1.run_cycle()
    history_file = tmp_path / "foresight_history.json"
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(ft.to_dict(), fh, indent=2)

    with history_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    restored = ForesightTracker.from_dict(data)
    tracker2 = DummyROITracker([2.0])
    eng2 = MiniSelfImprovementEngine(tracker2, restored)
    eng2.run_cycle()
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(restored.to_dict(), fh, indent=2)

    with history_file.open("r", encoding="utf-8") as fh:
        final = json.load(fh)
    assert [e["roi_delta"] for e in final["history"]["wf"]] == [1.0, 2.0]


def test_promotion_blocked_by_risk_or_brittleness():
    ft = ForesightTracker()
    tracker = DummyROITracker([1.0])
    eng = MiniSelfImprovementEngine(tracker, ft)

    # Immediate collapse risk should block promotion
    ft.predict_roi_collapse = lambda wf: {
        "risk": "Immediate collapse risk",
        "brittle": False,
    }
    eng.attempt_promotion()
    assert not eng.workflow_ready

    # Brittleness alone should also block promotion
    ft.predict_roi_collapse = lambda wf: {"risk": "Stable", "brittle": True}
    eng.attempt_promotion()
    assert not eng.workflow_ready
