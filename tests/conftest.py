# flake8: noqa
import os
import sys
import logging
import importlib.util
from pathlib import Path
import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Add project root to import path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SANDBOX_REPO_PATH", str(ROOT))
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

# Provide a lightweight stub to avoid pulling heavy ML dependencies
if "sentence_transformers" not in sys.modules:
    st_mod = types.ModuleType("sentence_transformers")

    class _Vec(list):
        def tolist(self) -> list[float]:
            return list(self)

    class _Sent:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts):
            return [_Vec([0.0])]

    st_mod.SentenceTransformer = _Sent
    sys.modules["sentence_transformers"] = st_mod

# Provide a lightweight sandbox_runner stub to avoid dependency checks
sandbox_env = types.ModuleType("sandbox_runner.environment")
sandbox_env.simulate_temporal_trajectory = lambda *a, **k: None
sandbox_env.SANDBOX_ENV_PRESETS = [{}]
sandbox_env.load_presets = lambda: sandbox_env.SANDBOX_ENV_PRESETS
sandbox_env.simulate_full_environment = lambda *a, **k: None
sandbox_pkg = types.ModuleType("sandbox_runner")
sandbox_pkg.environment = sandbox_env
sandbox_pkg.__path__ = [str(ROOT / "sandbox_runner")]
sys.modules["sandbox_runner"] = sandbox_pkg
sys.modules["sandbox_runner.environment"] = sandbox_env

# Minimal stubs for database helpers used during imports
embeddable_stub = types.ModuleType("embeddable_db_mixin")
embeddable_stub.log_embedding_metrics = lambda *a, **k: None
embeddable_stub.EmbeddableDBMixin = object
sys.modules.setdefault("embeddable_db_mixin", embeddable_stub)
code_db_stub = types.ModuleType("code_database")
code_db_stub.PatchHistoryDB = object
sys.modules.setdefault("code_database", code_db_stub)
sys.modules.setdefault("menace.code_database", code_db_stub)

# Stub retrieval_cache to avoid database initialisation during tests
retrieval_cache_stub = types.ModuleType("retrieval_cache")
retrieval_cache_stub.RetrievalCache = object
sys.modules.setdefault("retrieval_cache", retrieval_cache_stub)

# Provide lightweight ensure_fresh_weights without triggering heavy imports
context_builder_util_stub = types.ModuleType("context_builder_util")
def _ensure_fresh_weights(builder):
    try:
        builder.refresh_db_weights()
    except Exception:
        pass
context_builder_util_stub.ensure_fresh_weights = _ensure_fresh_weights
sys.modules.setdefault("context_builder_util", context_builder_util_stub)

# Stub synergy_history_db to avoid database router initialisation
synergy_history_db_stub = types.ModuleType("synergy_history_db")
synergy_history_db_stub.SynergyHistoryDB = object
sys.modules.setdefault("synergy_history_db", synergy_history_db_stub)

# Stub run_autonomous to bypass dependency checks if imported
run_auto_stub = types.ModuleType("run_autonomous")
run_auto_stub._verify_required_dependencies = lambda: None
sys.modules.setdefault("run_autonomous", run_auto_stub)
sys.modules.setdefault("menace.run_autonomous", run_auto_stub)

# Load the real menace package if possible but tolerate missing system deps
if os.getenv("MENACE_IMPORT_REAL", "0") == "1":
    spec = importlib.util.spec_from_file_location("menace", ROOT / "__init__.py")  # path-ignore
    real_mod = importlib.util.module_from_spec(spec)
    try:  # pragma: no cover - defensive against SystemExit
        spec.loader.exec_module(real_mod)  # type: ignore[attr-defined]
    except SystemExit:
        real_mod = types.ModuleType("menace")
else:
    real_mod = types.ModuleType("menace")

if not hasattr(real_mod, "ForesightTracker"):
    from foresight_tracker import ForesightTracker as _FT
    real_mod.ForesightTracker = _FT
if not hasattr(real_mod, "UpgradeForecaster"):
    from upgrade_forecaster import UpgradeForecaster as _UF
    real_mod.UpgradeForecaster = _UF

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


@pytest.fixture
def in_memory_dbs(monkeypatch):
    """Provide in-memory implementations of ROI and stability DBs."""

    from roi_results_db import ROIResult

    class InMemoryROIResultsDB:
        instances: list["InMemoryROIResultsDB"] = []

        def __init__(self, *a, **k):
            self.records: list[ROIResult] = []
            self.__class__.instances.append(self)

        def log_result(self, **kwargs):
            self.records.append(ROIResult(**kwargs))

        def fetch_results(self, workflow_id: str, run_id: str | None = None):
            return [
                r
                for r in self.records
                if r.workflow_id == workflow_id and (run_id is None or r.run_id == run_id)
            ]

    class InMemoryWorkflowStabilityDB:
        instances: list["InMemoryWorkflowStabilityDB"] = []

        def __init__(self, *a, **k):
            self.data: dict[str, dict[str, float]] = {}
            self.__class__.instances.append(self)

        def record_metrics(
            self,
            workflow_id: str,
            roi: float,
            failures: float,
            entropy: float,
            *,
            roi_delta: float | None = None,
            roi_var: float = 0.0,
            failures_var: float = 0.0,
            entropy_var: float = 0.0,
        ) -> None:
            prev = self.data.get(workflow_id, {}).get("roi", 0.0)
            delta = roi - prev if roi_delta is None else roi_delta
            self.data[workflow_id] = {
                "roi": roi,
                "roi_delta": delta,
                "roi_var": roi_var,
                "failures": failures,
                "failures_var": failures_var,
                "entropy": entropy,
                "entropy_var": entropy_var,
            }

        def is_stable(
            self, workflow_id: str, current_roi: float | None = None, threshold: float | None = None
        ) -> bool:
            entry = self.data.get(workflow_id)
            if not entry:
                return False
            if current_roi is not None and threshold is not None:
                if abs(current_roi - entry["roi"]) > threshold:
                    del self.data[workflow_id]
                    return False
            return True

    roi_mod = types.SimpleNamespace(ROIResultsDB=InMemoryROIResultsDB, ROIResult=ROIResult)
    stability_mod = types.SimpleNamespace(WorkflowStabilityDB=InMemoryWorkflowStabilityDB)
    monkeypatch.setitem(sys.modules, "menace.roi_results_db", roi_mod)
    monkeypatch.setitem(sys.modules, "menace.workflow_stability_db", stability_mod)
    return InMemoryROIResultsDB, InMemoryWorkflowStabilityDB

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
import yaml
from foresight_tracker import ForesightTracker


@pytest.fixture
def foresight_templates(tmp_path):
    data = {
        "profiles": {"wf": "wf"},
        "trajectories": {"wf": [0.5, 0.5, 0.5, 0.5, 0.5]},
        "entropy_profiles": {"wf": "wf"},
        "entropy_trajectories": {"wf": [0.2, 0.2, 0.2, 0.2, 0.2]},
    }
    path = tmp_path / "foresight_templates.yaml"
    with path.open("w", encoding="utf8") as fh:
        yaml.safe_dump(data, fh)
    return path


@pytest.fixture
def tracker_with_templates(foresight_templates):
    return ForesightTracker(templates_path=foresight_templates)


# ---------------------------------------------------------------------------
# Foresight gate helper fixtures
# ---------------------------------------------------------------------------


class _CollapseTracker:
    def __init__(self, result):
        self._result = result

    def predict_roi_collapse(self, workflow_id):
        return self._result


@pytest.fixture
def stable_tracker():
    """Tracker reporting no collapse risk."""

    return _CollapseTracker({"risk": "Stable", "brittle": False})


@pytest.fixture
def brittle_tracker():
    """Tracker flagging early collapse brittleness."""

    return _CollapseTracker({"risk": "Stable", "brittle": True})


@pytest.fixture
def stub_graph():
    """WorkflowGraph stub with no negative impact."""

    class _Graph:
        def simulate_impact_wave(self, workflow_id, roi_delta, synergy_delta):
            return {}

    return _Graph()


@pytest.fixture
def negative_impact_graph():
    """WorkflowGraph stub reporting a negative ROI impact."""

    class _Graph:
        def simulate_impact_wave(self, workflow_id, roi_delta, synergy_delta):
            return {"dep": {"roi": -0.1}}

    return _Graph()

