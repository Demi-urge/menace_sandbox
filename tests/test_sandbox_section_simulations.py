import types
from pathlib import Path


class DummyAudit:
    def __init__(self, path):
        self.path = Path(path)
        self.records = []

    def record(self, data):
        self.records.append(data)


class _SandboxMetaLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.audit = DummyAudit(path)
        self.records = []
        self.module_deltas = {}
        self.flagged_sections = set()
        self.last_patch_id = 0

    def log_cycle(self, cycle: int, roi: float, modules: list[str], reason: str) -> None:
        prev = self.records[-1][1] if self.records else 0.0
        delta = roi - prev
        self.records.append((cycle, roi, delta, modules, reason))
        for m in modules:
            self.module_deltas.setdefault(m, []).append(delta)
        try:
            self.audit.record({"cycle": cycle, "roi": roi, "delta": delta, "modules": modules, "reason": reason})
        except Exception:
            pass

    def diminishing(self, threshold: float | None = None, consecutive: int = 3) -> list[str]:
        flags = []
        thr = 0.0 if threshold is None else float(threshold)
        eps = 1e-3
        for m, vals in self.module_deltas.items():
            if m in self.flagged_sections:
                continue
            if len(vals) < consecutive:
                continue
            window = vals[-consecutive:]
            mean = sum(window) / consecutive
            if len(window) > 1:
                var = sum((v - mean) ** 2 for v in window) / len(window)
                std = var ** 0.5
            else:
                std = 0.0
            if abs(mean) <= thr and std < eps:
                flags.append(m)
                self.flagged_sections.add(m)
        return flags


class DummyEngine:
    def __init__(self):
        self.val = 0.0

    def run_cycle(self):
        self.val += 0.01
        return types.SimpleNamespace(roi=types.SimpleNamespace(roi=self.val))


class DummyTracker:
    def __init__(self):
        self.roi_history = []
        self.metrics_history = {}

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.roi_history.append(curr)
        for k, v in (metrics or {}).items():
            self.metrics_history.setdefault(k, []).append(v)
        return 0.0, [], False

    def diminishing(self):
        return 0.011


def test_section_metrics_and_diminishing(tmp_path):
    log = _SandboxMetaLogger(tmp_path / "log.txt")
    engine = DummyEngine()
    tracker = DummyTracker()
    for i in range(4):
        res = engine.run_cycle()
        tracker.update(0.01 * i, res.roi.roi, modules=["mod.py"], metrics={"m": i})
        log.log_cycle(i, res.roi.roi, ["mod.py"], "test")
    flags = log.diminishing(tracker.diminishing())
    assert "mod.py" in flags
    assert tracker.metrics_history["m"] == [0, 1, 2, 3]
