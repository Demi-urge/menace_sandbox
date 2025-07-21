import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Add project root to import path
ROOT = Path(__file__).resolve().parents[1]
# Ensure the parent of the project directory is on ``sys.path`` so that the
# ``menace`` package can be imported without installation.  Using the project
# root itself would pick up the ``menace`` subpackage instead of the actual
# package.
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))
if "menace" in sys.modules:
    del sys.modules["menace"]
import types
menace_stub = types.ModuleType("menace")
metrics_stub = types.ModuleType("menace.metrics_dashboard")
metrics_stub.MetricsDashboard = lambda *a, **k: object()
menace_stub.metrics_dashboard = metrics_stub
sys.modules.setdefault("menace", menace_stub)
sys.modules.setdefault("menace.metrics_dashboard", metrics_stub)

# Provide lightweight stubs for optional heavy dependencies
if "pulp" not in sys.modules:
    import types

    stub = types.ModuleType("pulp")
    stub.__doc__ = "stub"
    stub.__version__ = "0"
    sys.modules["pulp"] = stub

if "sqlalchemy" not in sys.modules:
    sa = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    engine_mod.Engine = object
    sa.engine = engine_mod
    sa.create_engine = lambda *a, **k: None
    sa.Boolean = sa.Column = sa.Float = lambda *a, **k: None
    sa.ForeignKey = sa.Integer = sa.MetaData = lambda *a, **k: None
    sa.String = sa.Table = sa.Text = lambda *a, **k: None
    sys.modules["sqlalchemy"] = sa
    sys.modules["sqlalchemy.engine"] = engine_mod

# Pre-import sqlalchemy to prevent test stubs from overriding it
try:
    import sqlalchemy as _sa  # noqa: F401
except Exception:
    pass

# Provide a stub for preliminary_research_bot to avoid circular imports
if "menace.preliminary_research_bot" not in sys.modules:
    import types
    from dataclasses import dataclass

    pr_stub = types.ModuleType("menace.preliminary_research_bot")

    @dataclass
    class BusinessData:
        model_name: str = ""

    class PreliminaryResearchBot:
        def process_model(self, name: str, urls):
            return BusinessData(model_name=name)

    pr_stub.BusinessData = BusinessData
    pr_stub.PreliminaryResearchBot = PreliminaryResearchBot
    sys.modules["menace.preliminary_research_bot"] = pr_stub

import pytest


@pytest.fixture(autouse=True)
def _ensure_sqlalchemy():
    """Reload SQLAlchemy after each test to undo stubs."""
    yield
    try:
        import importlib, sys
        import sqlalchemy as _sa
        sys.modules.pop("sqlalchemy.engine", None)
        importlib.reload(_sa)
    except Exception:
        pass


class _TrackerMock:
    def __init__(self, roi, metrics, preds=None):
        self.roi_history = list(roi)
        self.metrics_history = {k: list(v) for k, v in metrics.items()}
        self._preds = preds or {}

    def diminishing(self):
        return 0.01

    def predict_synergy_metric(self, name: str) -> float:
        return self._preds.get(name, 0.0)

    def predict_synergy(self):
        return self._preds.get("synergy_roi", 0.0)

    def forecast_synergy(self):
        return [self._preds.get("synergy_roi", 0.0)]


@pytest.fixture
def tracker_factory():
    """Return a factory creating ROITracker mocks."""

    def _make(roi=None, metrics=None, preds=None):
        roi = roi or [0.0, 0.1, 0.2]
        metrics = metrics or {
            "security_score": [70, 70, 70],
            "synergy_roi": [0.0, 0.0, 0.0],
        }
        return _TrackerMock(roi, metrics, preds)

    return _make
