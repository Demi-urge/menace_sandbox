from __future__ import annotations

from pathlib import Path
from typing import Mapping


def _load_stub() -> type:
    """Extract the light-mode ForesightTracker stub from sandbox_runner.py."""

    path = Path(__file__).resolve().parents[1] / "sandbox_runner.py"
    lines = path.read_text().splitlines()

    start = None
    for idx, line in enumerate(lines):
        if line.strip() == 'if os.getenv("MENACE_LIGHT_IMPORTS"):':
            start = idx + 1
            break
    assert start is not None, "light-mode stub not found"

    stub_lines = []
    for line in lines[start:]:
        if line.startswith("else:"):
            break
        stub_lines.append(line[4:] if line.startswith("    ") else line)

    code = "\n".join(stub_lines)
    namespace = {"Mapping": Mapping}
    exec(code, namespace)
    return namespace["ForesightTracker"]


def test_light_stub_behaviour() -> None:
    ForesightTracker = _load_stub()
    tracker = ForesightTracker(N=3, volatility_threshold=2.0)

    # All operations are no-ops in the stub
    tracker.record_cycle_metrics("wf", {"m": 1.0})
    assert tracker.history == {}

    assert tracker.max_cycles == 3
    assert tracker.get_trend_curve("wf") == (0.0, 0.0, 0.0)
    assert not tracker.is_stable("wf")

    data = tracker.to_dict()
    assert data["max_cycles"] == 3
    assert data["volatility_threshold"] == 2.0

    restored = ForesightTracker.from_dict(data, N=2)
    assert restored.max_cycles == 2
    assert restored.history == {}

