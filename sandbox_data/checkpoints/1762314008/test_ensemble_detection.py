import types
import sys


def _prepare(monkeypatch):
    stub = types.ModuleType("stub")
    stub.Template = lambda *a, **k: types.SimpleNamespace(render=lambda **kw: "")
    stub.safe_load = lambda *a, **k: {}
    for mod in [
        "jinja2",
        "yaml",
        "numpy",
        "matplotlib",
        "matplotlib.pyplot",  # path-ignore
        "cryptography",
        "cryptography.hazmat",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.asymmetric",
        "sqlalchemy",
        "sqlalchemy.engine",
        "httpx",
    ]:
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    monkeypatch.setitem(sys.modules, "jinja2", stub)
    monkeypatch.setitem(sys.modules, "yaml", stub)
    mpl = sys.modules["matplotlib"]
    mpl.__path__ = []  # type: ignore[attr-defined]
    mpl.pyplot = sys.modules["matplotlib.pyplot"]  # path-ignore

    # minimal SQLAlchemy Engine and crypto stubs
    sys.modules["sqlalchemy.engine"].Engine = type("Engine", (), {})
    ed25519 = types.ModuleType("ed25519")
    class _Priv:
        @staticmethod
        def generate():
            return _Priv()
        @staticmethod
        def from_private_bytes(b):
            return _Priv()
        def public_key(self):
            return _Pub()
        def sign(self, data):
            return b"sig"
    class _Pub:
        @staticmethod
        def from_public_bytes(b):
            return _Pub()
        def public_bytes(self, encoding=None, format=None):
            return b"pub"
        def verify(self, sig, data):
            pass
    ed25519.Ed25519PrivateKey = _Priv
    ed25519.Ed25519PublicKey = _Pub
    monkeypatch.setitem(
        sys.modules,
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        ed25519,
    )
    serial = types.ModuleType("serialization")
    serial.Encoding = types.SimpleNamespace(Raw="raw")
    serial.PublicFormat = types.SimpleNamespace(Raw="raw")
    monkeypatch.setitem(
        sys.modules,
        "cryptography.hazmat.primitives.serialization",
        serial,
    )

    import menace.operational_monitor_bot as omb  # type: ignore
    import menace.data_bot as db  # type: ignore
    import menace.advanced_error_management as aem  # type: ignore
    return omb, db, aem

class StubES:
    def __init__(self) -> None:
        self.docs = []
    def add(self, doc_id: str, body: dict) -> None:
        self.docs.append({"id": doc_id, **body})
    def search_all(self):
        return list(self.docs)


def test_ensemble_detection(tmp_path, monkeypatch):
    omb, db, aem = _prepare(monkeypatch)
    monkeypatch.setattr(db, "Gauge", None)
    monkeypatch.setattr(aem, "_ae_scores", lambda vals: [0.0]*(len(vals)-1)+[5.0])
    monkeypatch.setattr(aem, "_cluster_scores", lambda vals: [0.0]*(len(vals)-1)+[5.0])
    mdb = db.MetricsDB(tmp_path / "m.db")
    es = StubES()
    a_db = omb.AnomalyDB(tmp_path / "a.db")
    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    calls = []

    class DummyPB:
        def __init__(self):
            pass
        def generate(self, anomalies):
            calls.append(anomalies)
            return "pb.json"

    monkeypatch.setattr(omb, "PlaybookGenerator", DummyPB)
    bot = omb.OperationalMonitoringBot(mdb, es, None, a_db, db_router=router)
    normal = db.MetricRecord("bot1", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(normal)
    anomaly = db.MetricRecord("bot1", 99.0, 99.0, 1.0, 1.0, 1.0, 5)
    mdb.add(anomaly)
    bot.detect_anomalies("bot1")
    assert calls
    assert "high_cpu" in calls[0] and "error_rate" in calls[0]


def test_ensemble_detection_weighting(tmp_path, monkeypatch):
    omb, db, aem = _prepare(monkeypatch)
    monkeypatch.setattr(db, "Gauge", None)
    monkeypatch.setattr(aem, "_ae_scores", lambda vals: [0.0] * len(vals))
    monkeypatch.setattr(aem, "_cluster_scores", lambda vals: [0.0]*(len(vals)-1)+[3.0])
    mdb = db.MetricsDB(tmp_path / "m.db")
    es = StubES()
    a_db = omb.AnomalyDB(tmp_path / "a.db")
    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    calls: list[list[str]] = []

    class DummyPB:
        def generate(self, anomalies):
            calls.append(anomalies)
            return "pb.json"

    monkeypatch.setattr(omb, "PlaybookGenerator", DummyPB)
    bot = omb.OperationalMonitoringBot(mdb, es, None, a_db, db_router=router)
    normal = db.MetricRecord("bot1", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(normal)
    anomaly = db.MetricRecord("bot1", 99.0, 99.0, 1.0, 1.0, 1.0, 5)
    mdb.add(anomaly)
    bot.detect_anomalies("bot1")
    assert not calls


def test_ensemble_detection_fallback(tmp_path, monkeypatch, caplog):
    # stub heavy optional dependencies to avoid import errors
    for mod in [
        "jinja2",
        "yaml",
        "numpy",
        "matplotlib",
        "matplotlib.pyplot",  # path-ignore
        "cryptography",
        "cryptography.hazmat",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.asymmetric",
        "sqlalchemy",
        "sqlalchemy.engine",
        "httpx",
    ]:
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    # ensure Template and safe_load exist on stubs
    sys.modules["jinja2"].Template = lambda *a, **k: types.SimpleNamespace(render=lambda **kw: "")
    sys.modules["yaml"].safe_load = lambda *a, **k: {}

    # minimal SQLAlchemy Engine stub
    sa_engine = sys.modules["sqlalchemy.engine"]
    sa_engine.Engine = type("Engine", (), {})

    # detailed crypto stubs used by AuditTrail
    ed25519 = types.ModuleType("ed25519")
    class _Priv:
        @staticmethod
        def generate():
            return _Priv()
        @staticmethod
        def from_private_bytes(b):
            return _Priv()
        def public_key(self):
            return _Pub()
        def sign(self, data):
            return b"sig"
    class _Pub:
        @staticmethod
        def from_public_bytes(b):
            return _Pub()
        def public_bytes(self, encoding=None, format=None):
            return b"pub"
        def verify(self, sig, data):
            pass
    ed25519.Ed25519PrivateKey = _Priv
    ed25519.Ed25519PublicKey = _Pub
    monkeypatch.setitem(
        sys.modules,
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        ed25519,
    )
    serial = types.ModuleType("serialization")
    serial.Encoding = types.SimpleNamespace(Raw="raw")
    serial.PublicFormat = types.SimpleNamespace(Raw="raw")
    monkeypatch.setitem(
        sys.modules,
        "cryptography.hazmat.primitives.serialization",
        serial,
    )

    import menace.data_bot as db  # type: ignore
    import menace.advanced_error_management as aem  # type: ignore

    monkeypatch.setattr(db, "Gauge", None)

    def boom(vals):
        raise RuntimeError("boom")

    monkeypatch.setattr(aem, "_ae_scores", boom)
    monkeypatch.setattr(aem, "_cluster_scores", boom)
    caplog.set_level("ERROR")

    mdb = db.MetricsDB(tmp_path / "m.db")
    detector = aem.AnomalyEnsembleDetector(mdb)
    normal = db.MetricRecord("b", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(normal)
    anomaly = db.MetricRecord("b", 99.0, 99.0, 1.0, 1.0, 1.0, 5)
    mdb.add(anomaly)
    anomalies = detector.detect()
    assert "high_cpu" in anomalies and "error_rate" in anomalies
    assert "_ae_scores failed" in caplog.text
    assert "_cluster_scores failed" in caplog.text
