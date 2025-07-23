import os
import sys
import logging
import importlib.util
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
menace_stub.__path__ = [str(ROOT)]
spec = importlib.util.spec_from_file_location("menace", ROOT / "__init__.py")
real_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_mod)
for _name in dir(real_mod):
    if not _name.startswith("__"):
        setattr(menace_stub, _name, getattr(real_mod, _name))
sys.modules.setdefault("menace", menace_stub)
sys.modules.setdefault("menace.metrics_dashboard", metrics_stub)

# Stub cryptography for modules requiring it
if "cryptography" not in sys.modules:
    crypto_mod = types.ModuleType("cryptography")
    hazmat = types.ModuleType("cryptography.hazmat")
    primitives = types.ModuleType("primitives")
    asymmetric = types.ModuleType("asymmetric")
    ed25519 = types.ModuleType("ed25519")
    ed25519.Ed25519PrivateKey = type("Ed25519PrivateKey", (), {"generate": lambda: object()})
    ed25519.Ed25519PublicKey = object
    asymmetric.ed25519 = ed25519
    primitives.asymmetric = asymmetric
    serialization = types.ModuleType("serialization")
    primitives.serialization = serialization
    hazmat.primitives = primitives
    crypto_mod.hazmat = hazmat
    sys.modules["cryptography"] = crypto_mod
    sys.modules["cryptography.hazmat"] = hazmat
    sys.modules["cryptography.hazmat.primitives"] = primitives
    sys.modules["cryptography.hazmat.primitives.asymmetric"] = asymmetric
    sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"] = ed25519
    sys.modules["cryptography.hazmat.primitives.serialization"] = serialization

import importlib.util as _importlib_util
if "numpy" not in sys.modules and _importlib_util.find_spec("numpy") is None:
    class _Array(list):
        def reshape(self, *shape):
            return self

    np_stub = types.ModuleType("numpy")
    np_stub.isscalar = lambda x: isinstance(x, (int, float, complex))
    np_stub.bool_ = bool
    np_stub.arange = lambda n, *a, **k: _Array(range(n))
    np_stub.array = lambda x, dtype=None: _Array(list(x))
    np_stub.percentile = lambda data, p: 0.0
    np_stub.std = lambda x, ddof=0: 0.0
    np_stub.dot = lambda a, b: sum(float(x) * float(y) for x, y in zip(a, b))
    sys.modules["numpy"] = np_stub
if "sklearn" not in sys.modules and _importlib_util.find_spec("sklearn") is None:
    skl = types.ModuleType("sklearn")
    skl.pipeline = types.ModuleType("sklearn.pipeline")
    skl.preprocessing = types.ModuleType("sklearn.preprocessing")
    class _Poly:
        def __init__(self, *a, **k):
            pass

        def fit_transform(self, x):
            return x

    skl.preprocessing.PolynomialFeatures = _Poly
    skl.linear_model = types.ModuleType("sklearn.linear_model")
    class _LR:
        def fit(self, X, y):
            self.coef_ = [0.0, 0.0, 0.0]
            return self

        def predict(self, X):
            class Arr(list):
                def tolist(self_inner):
                    return list(self_inner)

            return Arr([0.0 for _ in range(len(X))])

    skl.linear_model.LinearRegression = _LR
    sys.modules["sklearn"] = skl
    sys.modules["sklearn.pipeline"] = skl.pipeline
    sys.modules["sklearn.preprocessing"] = skl.preprocessing
    sys.modules["sklearn.linear_model"] = skl.linear_model

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


@pytest.fixture(autouse=True)
def _set_synergy_method(monkeypatch):
    """Use ARIMA synergy forecasting during tests by default."""
    monkeypatch.setenv("SYNERGY_FORECAST_METHOD", "arima")
    yield
    monkeypatch.delenv("SYNERGY_FORECAST_METHOD", raising=False)


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
