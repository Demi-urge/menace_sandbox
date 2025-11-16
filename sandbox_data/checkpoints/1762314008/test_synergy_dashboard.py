import json
import sys
import types
import time
import socket
import logging
import os
import importlib
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
    pyplot_mod = types.ModuleType("matplotlib.pyplot")  # path-ignore
    mpl_mod.pyplot = pyplot_mod  # path-ignore
    sys.modules["matplotlib"] = mpl_mod
    sys.modules["matplotlib.pyplot"] = pyplot_mod  # path-ignore

if "dotenv" not in sys.modules:
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: None
    dotenv_mod.dotenv_values = lambda *a, **k: {}
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

if "pydantic" not in sys.modules:
    pyd_mod = types.ModuleType("pydantic")
    pyd_mod.Field = lambda default=None, **k: default
    pyd_mod.BaseSettings = object
    pyd_mod.BaseModel = object
    class _VE(Exception):
        pass
    pyd_mod.ValidationError = _VE
    pyd_dc = types.ModuleType("dataclasses")
    pyd_dc.dataclass = lambda *a, **k: (lambda cls: cls)
    pyd_mod.dataclasses = pyd_dc
    sys.modules["pydantic"] = pyd_mod
    sys.modules["pydantic.dataclasses"] = pyd_dc

if "pydantic_settings" not in sys.modules:
    ps_mod = types.ModuleType("pydantic_settings")
    ps_mod.BaseSettings = object
    ps_mod.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = ps_mod

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

if "gunicorn.app.base" not in sys.modules:
    gunicorn_mod = types.ModuleType("gunicorn")
    gunicorn_app_mod = types.ModuleType("gunicorn.app")
    gunicorn_base_mod = types.ModuleType("gunicorn.app.base")

    class _DummyGA:
        def __init__(self, *a, **k):
            self.cfg = types.SimpleNamespace(set=lambda *a, **k: None)

        def run(self):
            pass

    gunicorn_base_mod.BaseApplication = _DummyGA
    sys.modules["gunicorn"] = gunicorn_mod
    sys.modules["gunicorn.app"] = gunicorn_app_mod
    sys.modules["gunicorn.app.base"] = gunicorn_base_mod

if "uvicorn" not in sys.modules:
    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.run = lambda *a, **k: None
    sys.modules["uvicorn"] = uvicorn_mod

if "starlette.middleware.wsgi" not in sys.modules:
    star_mod = types.ModuleType("starlette")
    mid_mod = types.ModuleType("starlette.middleware")
    wsgi_mod = types.ModuleType("starlette.middleware.wsgi")

    class _WSGI:
        def __init__(self, app):
            self.app = app

    wsgi_mod.WSGIMiddleware = _WSGI
    sys.modules["starlette"] = star_mod
    sys.modules["starlette.middleware"] = mid_mod
    sys.modules["starlette.middleware.wsgi"] = wsgi_mod

if "requests" not in sys.modules:
    req_mod = types.ModuleType("requests")
    exc_mod = types.SimpleNamespace(Timeout=type("Timeout", (Exception,), {}))
    req_mod.exceptions = exc_mod
    req_mod.get = lambda *a, **k: None
    sys.modules["requests"] = req_mod

from menace.self_improvement import (
    synergy_stats,
    SynergyDashboard,
    synergy_ma,
)


HISTORY = [
    {"synergy_roi": 0.1, "synergy_efficiency": 0.2},
    {"synergy_roi": 0.3, "synergy_efficiency": 0.1},
]


def _load_cli():
    """Import sandbox_runner.cli with light imports enabled."""
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
    if "sandbox_runner.cli" in sys.modules:
        return importlib.reload(sys.modules["sandbox_runner.cli"])
    return importlib.import_module("sandbox_runner.cli")


def test_synergy_stats():
    stats = synergy_stats(HISTORY)
    assert stats["synergy_roi"]["average"] == pytest.approx(0.2)
    assert stats["synergy_roi"]["variance"] == pytest.approx(0.01)


def test_synergy_ma():
    ma = synergy_ma(HISTORY, window=2)
    assert ma[-1]["synergy_roi"] == pytest.approx(0.2)
    assert ma[-1]["synergy_efficiency"] == pytest.approx(0.15)


def test_dashboard_endpoints():
    dash = SynergyDashboard(ma_window=2)
    dash._load = lambda: HISTORY

    hist, code = dash.history()
    assert code == 200
    assert hist == HISTORY

    data, code = dash.stats()
    assert code == 200
    assert data["latest"] == HISTORY[-1]
    assert data["rolling_average"]["synergy_roi"] == pytest.approx(0.2)


def test_dashboard_update_loop(monkeypatch):
    updates = [{"synergy_roi": 0.4}, {"synergy_roi": 0.5}]
    iterator = iter(updates)

    def fake_fetch(self):
        try:
            return next(iterator)
        except StopIteration:
            return {}

    monkeypatch.setattr(SynergyDashboard, "_fetch_exporter_metrics", fake_fetch)

    dash = SynergyDashboard(exporter_host="localhost", refresh_interval=0.01)
    try:
        for _ in range(100):
            if len(dash._history) >= 2:
                break
            time.sleep(0.02)
        assert dash._history[:2] == updates
    finally:
        dash.stop()


def test_dashboard_max_history(monkeypatch):
    updates = [{"synergy_roi": i} for i in range(5)]
    iterator = iter(updates)

    def fake_fetch(self):
        try:
            return next(iterator)
        except StopIteration:
            return {}

    monkeypatch.setattr(SynergyDashboard, "_fetch_exporter_metrics", fake_fetch)

    dash = SynergyDashboard(
        exporter_host="localhost", refresh_interval=0.01, max_history=2
    )
    try:
        for _ in range(200):
            if len(dash._history) >= 2 and dash._history[-1].get("synergy_roi") == updates[-1]["synergy_roi"]:
                break
            time.sleep(0.02)
        assert len(dash._history) <= 2
        assert dash._history == updates[-2:]
    finally:
        dash.stop()


def test_dashboard_exporter_unreachable(monkeypatch):
    import requests

    responses = [types.SimpleNamespace(status_code=200, text="synergy_roi 0.4\n")]

    def fake_get(url, timeout=1.0):
        if responses:
            return responses.pop(0)
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(requests, "get", fake_get)

    dash = SynergyDashboard(exporter_host="localhost", refresh_interval=0.01)
    try:
        for _ in range(100):
            if len(dash._history) >= 2:
                break
            time.sleep(0.02)
        assert dash._history[0] == {"synergy_roi": 0.4}
        assert dash._history[1] == {"synergy_roi": 0.4}
        assert dash._thread and dash._thread.is_alive()
    finally:
        dash.stop()


def test_dashboard_run_gunicorn(monkeypatch):
    calls = {}

    class DummyGA:
        def __init__(self, *a, **k):
            self.cfg = types.SimpleNamespace(set=lambda *a, **k: None)

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(
        sys.modules["gunicorn.app.base"], "BaseApplication", DummyGA, raising=False
    )

    dash = SynergyDashboard()
    dash.run(port=0, wsgi="gunicorn")

    assert calls.get("ran") is True


def test_dashboard_run_uvicorn(monkeypatch):
    calls = {}

    class DummyWSGI:
        def __init__(self, app):
            self.app = app
            calls["wrapped"] = app

    def dummy_run(app, host="", port=0, workers=1):
        calls["run"] = {"app": app, "host": host, "port": port, "workers": workers}

    monkeypatch.setattr(
        sys.modules["starlette.middleware.wsgi"], "WSGIMiddleware", DummyWSGI
    )
    monkeypatch.setattr(sys.modules["uvicorn"], "run", dummy_run)

    dash = SynergyDashboard()
    dash.run(port=0, wsgi="uvicorn")

    assert calls["run"]["app"].app is dash.app


def test_dashboard_run_port_in_use(caplog):
    dash = SynergyDashboard()
    sock = socket.socket()
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    caplog.set_level(logging.ERROR)
    try:
        with pytest.raises(OSError):
            dash.run(port=port)
    finally:
        sock.close()
    assert f"port {port} in use" in caplog.text


def test_synergy_converged_noisy_metrics():
    cli = _load_cli()
    hist = [
        {"synergy_roi": 0.05},
        {"synergy_roi": -0.04},
        {"synergy_roi": 0.06},
        {"synergy_roi": -0.05},
        {"synergy_roi": 0.07},
        {"synergy_roi": -0.06},
    ]
    ok, _, conf = cli._synergy_converged(hist, 6, 0.01)
    assert ok is False
    assert conf < 0.95


def test_synergy_converged_variance_jump():
    cli = _load_cli()
    hist = [
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.001},
        {"synergy_roi": 0.3},
        {"synergy_roi": -0.3},
        {"synergy_roi": 0.3},
        {"synergy_roi": -0.3},
    ]
    ok, _, conf = cli._synergy_converged(hist, 8, 0.01)
    assert ok is False
    assert conf < 0.95




