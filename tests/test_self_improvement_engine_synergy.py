import os
import importlib.util
import sys
import types
import math
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

try:
    import jinja2 as _j2  # noqa: F401
except Exception:  # pragma: no cover - optional
    jinja_stub = types.ModuleType("jinja2")
    jinja_stub.Template = lambda *a, **k: None
    sys.modules["jinja2"] = jinja_stub

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_stub)

if "networkx" not in sys.modules:
    nx_stub = types.ModuleType("networkx")
    nx_stub.DiGraph = object
    sys.modules["networkx"] = nx_stub

if "psutil" not in sys.modules:
    sys.modules["psutil"] = types.ModuleType("psutil")

if "loguru" not in sys.modules:
    loguru_mod = types.ModuleType("loguru")

    class DummyLogger:
        def __getattr__(self, name):
            def stub(*a, **k):
                return None

            return stub

        def add(self, *a, **k):  # pragma: no cover - optional
            pass

    loguru_mod.logger = DummyLogger()
    sys.modules["loguru"] = loguru_mod

if "git" not in sys.modules:
    git_mod = types.ModuleType("git")
    git_mod.Repo = object
    exc_mod = types.ModuleType("git.exc")

    class _Err(Exception):
        pass

    exc_mod.GitCommandError = _Err
    exc_mod.InvalidGitRepositoryError = _Err
    exc_mod.NoSuchPathError = _Err
    git_mod.exc = exc_mod
    sys.modules["git.exc"] = exc_mod
    sys.modules["git"] = git_mod

if "filelock" not in sys.modules:
    filelock_mod = types.ModuleType("filelock")
    filelock_mod.FileLock = lambda *a, **k: object()
    filelock_mod.Timeout = type("Timeout", (Exception,), {})
    sys.modules["filelock"] = filelock_mod

if "matplotlib" not in sys.modules:
    mpl_mod = types.ModuleType("matplotlib")
    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    mpl_mod.pyplot = pyplot_mod
    sys.modules["matplotlib"] = mpl_mod
    sys.modules["matplotlib.pyplot"] = pyplot_mod

if "dotenv" not in sys.modules:
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv_mod

if "prometheus_client" not in sys.modules:
    prom_mod = types.ModuleType("prometheus_client")
    prom_mod.CollectorRegistry = object
    prom_mod.Counter = prom_mod.Gauge = lambda *a, **k: object()
    sys.modules["prometheus_client"] = prom_mod

if "joblib" not in sys.modules:
    joblib_mod = types.ModuleType("joblib")
    joblib_mod.dump = joblib_mod.load = lambda *a, **k: None
    sys.modules["joblib"] = joblib_mod

if "sklearn" not in sys.modules:
    sk_mod = types.ModuleType("sklearn")
    fe_mod = types.ModuleType("sklearn.feature_extraction")
    text_mod = types.ModuleType("sklearn.feature_extraction.text")
    text_mod.TfidfVectorizer = object
    fe_mod.text = text_mod
    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.KMeans = object
    lm_mod = types.ModuleType("sklearn.linear_model")
    lm_mod.LinearRegression = object
    sk_mod.feature_extraction = fe_mod
    sys.modules["sklearn"] = sk_mod
    sys.modules["sklearn.feature_extraction"] = fe_mod
    sys.modules["sklearn.feature_extraction.text"] = text_mod
    sys.modules["sklearn.cluster"] = cluster_mod
    sys.modules["sklearn.linear_model"] = lm_mod

# additional crypto stubs for lightweight import
crypto_mod = types.ModuleType("cryptography")
crypto_haz = types.ModuleType("hazmat")
crypto_pri = types.ModuleType("primitives")
crypto_asym = types.ModuleType("asymmetric")
crypto_ed = types.ModuleType("ed25519")
crypto_ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
crypto_ed.Ed25519PublicKey = object
crypto_pri.asymmetric = crypto_asym
crypto_asym.ed25519 = crypto_ed
crypto_pri.serialization = types.ModuleType("serialization")

sys.modules.setdefault("cryptography", crypto_mod)
sys.modules.setdefault("cryptography.hazmat", crypto_haz)
sys.modules.setdefault("cryptography.hazmat.primitives", crypto_pri)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", crypto_asym
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", crypto_ed
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization", crypto_pri.serialization
)

sys.modules.setdefault("pulp", types.ModuleType("pulp"))

pytest.importorskip("pandas")

import menace.self_improvement_engine as sie

# use lightweight stubs to avoid heavy initialisation
sie.InfoDB = lambda *a, **k: object()
sie.ResearchAggregatorBot = lambda *a, **k: object()

class _StubPipeline:
    def run(self, model: str, energy: int = 1):
        return None

sie.ModelAutomationPipeline = lambda *a, **k: _StubPipeline()
sie.ErrorBot = lambda *a, **k: object()
sie.ErrorDB = lambda *a, **k: object()
sie.MetricsDB = lambda *a, **k: object()
sie.DiagnosticManager = lambda *a, **k: object()


def _metric_delta(vals, window=3):
    w = min(window, len(vals))
    current = sum(vals[-w:]) / w
    if len(vals) > w:
        prev_w = min(w, len(vals) - w)
        prev = sum(vals[-w - prev_w : -w]) / prev_w
    elif len(vals) >= 2:
        prev = vals[-2]
    else:
        return float(vals[-1])
    return float(current - prev)


def _expected(records, metrics, window=3):
    base = {
        "synergy_roi": 1.0,
        "synergy_efficiency": 1.0,
        "synergy_resilience": 1.0,
        "synergy_antifragility": 1.0,
    }
    roi_vals = [r.roi_delta for r in records]
    weights = dict(base)
    stats: dict[str, tuple[float, float]] = {}
    if len(roi_vals) >= 2:
        mx = sum(roi_vals) / len(roi_vals)
        den2 = sum((b - mx) ** 2 for b in roi_vals)
        for name in base:
            vals = [getattr(r, name) for r in records]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            stats[name] = (m, math.sqrt(var) if var > 0 else 1.0)
            num = sum((a - m) * (b - mx) for a, b in zip(vals, roi_vals))
            den1 = sum((a - m) ** 2 for a in vals)
            corr = num / math.sqrt(den1 * den2) if den1 > 0 and den2 > 0 else 0.0
            weights[name] = (1.0 / (var + 1e-6)) * (1.0 + max(0.0, corr))
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] = weights[k] / total * len(weights)

    def norm(name: str) -> float:
        val = _metric_delta(metrics[name], window)
        mean, std = stats.get(name, (0.0, 1.0))
        return (val - mean) / (std + 1e-6)

    adj = sum(norm(n) * weights[n] for n in base)
    return adj, weights


class _Rec:
    def __init__(self, r, sr, se, res, af, ts):
        self.roi_delta = r
        self.synergy_roi = sr
        self.synergy_efficiency = se
        self.synergy_resilience = res
        self.synergy_antifragility = af
        self.ts = ts


class _DummyDB:
    def __init__(self, recs):
        self._recs = list(recs)

    def filter(self):
        return list(self._recs)


class _DummyTracker:
    def __init__(self, metrics):
        self.metrics_history = metrics


def test_weighted_synergy_adjustment():
    records = [
        _Rec(1, 1, 4, 2, 1, "1"),
        _Rec(2, 2, 3, -2, -1, "2"),
        _Rec(3, 3, 2, 2, 1, "3"),
        _Rec(4, 4, 1, -2, -1, "4"),
    ]
    metrics = {
        "synergy_roi": [0.0, 0.1, 0.2, 0.3],
        "synergy_efficiency": [0.3, 0.2, 0.1, 0.0],
        "synergy_resilience": [0.0, 2.0, -2.0, 2.0],
        "synergy_antifragility": [0.0, 1.0, -1.0, 1.0],
    }
    engine = sie.SelfImprovementEngine(interval=0, patch_db=_DummyDB(records))
    engine.tracker = _DummyTracker(metrics)
    expected, weights = _expected(records, metrics)
    result = engine._weighted_synergy_adjustment()
    assert result == pytest.approx(expected)
    assert weights["synergy_roi"] > weights["synergy_efficiency"]
    assert weights["synergy_resilience"] < weights["synergy_efficiency"]
