import json
import sys
import types
import pytest

if "jinja2" not in sys.modules:
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

        def add(self, *a, **k):
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
    sk_mod.__path__ = []
    fe_mod = types.ModuleType("sklearn.feature_extraction")
    fe_mod.__path__ = []
    text_mod = types.ModuleType("sklearn.feature_extraction.text")
    text_mod.__path__ = []
    text_mod.TfidfVectorizer = object
    fe_mod.text = text_mod
    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.__path__ = []
    cluster_mod.KMeans = object
    lm_mod = types.ModuleType("sklearn.linear_model")
    lm_mod.__path__ = []
    lm_mod.LinearRegression = object
    sk_mod.feature_extraction = fe_mod
    sys.modules["sklearn"] = sk_mod
    sys.modules["sklearn.feature_extraction"] = fe_mod
    sys.modules["sklearn.feature_extraction.text"] = text_mod
    sys.modules["sklearn.cluster"] = cluster_mod
    sys.modules["sklearn.linear_model"] = lm_mod

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

sys.modules.setdefault("menace.ai_counter_bot", types.ModuleType("menace.ai_counter_bot"))
sys.modules["menace.ai_counter_bot"].AICounterBot = object
sys.modules.setdefault("menace.newsreader_bot", types.ModuleType("menace.newsreader_bot"))
sys.modules["menace.newsreader_bot"].NewsDB = object

np_mod = types.ModuleType("numpy")

class _Arr(list):
        @property
        def size(self):
            return len(self)

        def mean(self):
            return sum(self) / len(self) if self else 0.0

        def var(self):
            m = self.mean()
            return sum((x - m) ** 2 for x in self) / len(self) if self else 0.0

def _array(seq, dtype=float):
    return _Arr(float(x) for x in seq)

np_mod.array = _array
np_mod.isscalar = lambda x: isinstance(x, (int, float, complex))
np_mod.bool_ = bool
sys.modules["numpy"] = np_mod

flask_mod = types.ModuleType("flask")

class DummyFlask:
    def __init__(self, *a, **k):
        pass

    def add_url_rule(self, *a, **k):
        pass

flask_mod.Flask = DummyFlask
flask_mod.jsonify = lambda obj: obj
sys.modules["flask"] = flask_mod

from menace.self_improvement_engine import synergy_stats, SynergyDashboard


HISTORY = [
    {"synergy_roi": 0.1, "synergy_efficiency": 0.2},
    {"synergy_roi": 0.3, "synergy_efficiency": 0.1},
]


def test_synergy_stats():
    stats = synergy_stats(HISTORY)
    assert stats["synergy_roi"]["average"] == pytest.approx(0.2)
    assert stats["synergy_roi"]["variance"] == pytest.approx(0.01)




