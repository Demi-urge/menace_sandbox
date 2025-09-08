import types
import logging
from tests.test_self_debugger_sandbox import (
    sds,
    DummyTelem,
    DummyEngine,
    DummyTrail,
)
import menace.diagnostic_manager as dm
import patch_score_backend as psb


class DummyBuilder(dm.ContextBuilder):
    def refresh_db_weights(self):
        return {}


def test_http_backend_store_and_fetch(monkeypatch):
    sent = {}

    class Resp:
        def __init__(self, data=None):
            self._data = data or []

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=5):
        sent.update(json)
        return Resp()

    def fake_get(url, params=None, timeout=5):
        return Resp([["a", "ok"]])

    monkeypatch.setattr(psb, "requests", types.SimpleNamespace(post=fake_post, get=fake_get))
    be = psb.HTTPPatchScoreBackend("http://x")
    be.store({"description": "a"})
    rows = be.fetch_recent(1)
    assert sent["description"] == "a"
    assert rows == [("a", "ok")]


def test_s3_backend_store_and_fetch(monkeypatch):
    uploaded = {}

    class FakeClient:
        def put_object(self, Bucket=None, Key=None, Body=None):
            uploaded["key"] = Key
            uploaded["body"] = Body

        def list_objects_v2(self, Bucket=None, Prefix=None):
            return {"Contents": [{"Key": "k", "LastModified": 1}]}

        def get_object(self, Bucket=None, Key=None):
            return {"Body": types.SimpleNamespace(read=lambda: b'["x","ok"]')}

    fake_boto3 = types.SimpleNamespace(client=lambda *_a, **_k: FakeClient())
    monkeypatch.setattr(psb, "boto3", fake_boto3)
    be = psb.S3PatchScoreBackend("b", "p")
    be.store({"d": 1})
    rows = be.fetch_recent(1)
    assert uploaded["key"].startswith("p/")
    assert rows == [("x", "ok")]


def test_file_backend_store_and_fetch(tmp_path):
    be = psb.FilePatchScoreBackend(str(tmp_path))
    be.store(["y", "ok"])
    rows = be.fetch_recent(1)
    assert rows == [("y", "ok")]


def test_store_logs_outcome(monkeypatch, tmp_path):
    captured = {}

    class DummyMetrics:
        def log_patch_outcome(
            self, patch_id, success, vectors, *, session_id="", reverted=False, roi_tag=None
        ):
            captured.update(
                patch_id=patch_id,
                success=success,
                vectors=list(vectors),
                reverted=reverted,
                session_id=session_id,
                roi_tag=roi_tag,
            )

    monkeypatch.setattr(psb, "MetricsDB", lambda *_a, **_k: DummyMetrics())
    be = psb.FilePatchScoreBackend(str(tmp_path))
    rec = psb.attach_retrieval_info(
        {"description": "p1", "result": "reverted", "roi_tag": "bug"}, "s1", [("db", "v1", 0.0)]
    )
    be.store(rec)
    assert captured["patch_id"] == "p1"
    assert captured["success"] is False
    assert captured["vectors"] == [("db", "v1")]
    assert captured["reverted"] is True
    assert captured["session_id"] == "s1"
    assert captured["roi_tag"] == "bug"


def test_backend_from_url_file(tmp_path):
    url = f"file://{tmp_path}"  # path-like URL
    be = psb.backend_from_url(url)
    assert isinstance(be, psb.FilePatchScoreBackend)
    assert be.directory.endswith(str(tmp_path))


def test_backend_from_url_fallback(tmp_path):
    path = str(tmp_path)
    be = psb.backend_from_url(path)
    assert isinstance(be, psb.FilePatchScoreBackend)
    assert be.directory == path

    be2 = psb.backend_from_url("foo://bar")
    assert isinstance(be2, psb.FilePatchScoreBackend)


def test_sandbox_uses_backend(monkeypatch, tmp_path):
    calls = {}

    class FakeBackend(psb.PatchScoreBackend):
        def store(self, rec):
            calls["store"] = rec

        def fetch_recent(self, limit=20):
            return [("r", "ok")]

    monkeypatch.setenv("PATCH_SCORE_BACKEND_URL", "http://x")
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr(psb, "backend_from_url", lambda u: FakeBackend())

    class DummySandbox:
        def __init__(self):
            self.backend = psb.backend_from_url("http://x")

        def _log_patch(self, desc, result):
            self.backend.store({"description": desc, "result": result})

        def recent_scores(self, n):
            return self.backend.fetch_recent(n)

    dbg = DummySandbox()
    dbg._log_patch("d", "ok")
    assert calls["store"]["description"] == "d"
    assert dbg.recent_scores(1) == [("r", "ok")]


def test_engine_uses_backend(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jinja2":
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from tests import test_self_improvement_engine as sie_tests

    calls = {}

    class FakeBackend(psb.PatchScoreBackend):
        def store(self, rec):
            calls["store"] = rec

        def fetch_recent(self, limit=20):
            return [("e", "ok")]

    monkeypatch.setenv("PATCH_SCORE_BACKEND_URL", "http://x")
    monkeypatch.setattr(psb, "backend_from_url", lambda u: FakeBackend())

    mdb = sie_tests.db.MetricsDB(tmp_path / "m.db")
    edb = sie_tests.eb.ErrorDB(tmp_path / "e.db")
    info = sie_tests.rab.InfoDB(tmp_path / "i.db")
    builder = DummyBuilder()
    diag = sie_tests.dm.DiagnosticManager(
        mdb,
        sie_tests.eb.ErrorBot(edb, mdb, context_builder=builder),
        context_builder=builder,
    )

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return sie_tests.mp.AutomationResult(
                package=None,
                roi=sie_tests.prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
            )

    monkeypatch.setattr(sie_tests.sie, "bootstrap", lambda: 0)
    engine = sie_tests.sie.SelfImprovementEngine(
        interval=0, pipeline=StubPipeline(), diagnostics=diag, info_db=info
    )
    mdb.add(sie_tests.db.MetricRecord("bot", 5.0, 10.0, 3.0, 1.0, 1.0, 1))
    edb.log_discrepancy("fail")
    engine.run_cycle()
    assert calls["store"]["description"] == engine.bot_name
    assert engine.recent_scores(1) == [("e", "ok")]


def test_alignment_check_blocks_secret_diff():
    rec = {
        "description": "p1",
        "result": "ok",
        "diff": (
            "diff --git a/x.py b/x.py\n"  # path-ignore
            "--- a/x.py\n"  # path-ignore
            "+++ b/x.py\n"  # path-ignore
            "@@ -0,0 +1 @@\n"
            "+password = 'secret'\n"
        ),
    }
    psb._log_outcome(rec)
    assert rec.get("result") == "blocked"
    assert rec.get("alignment_severity") >= 3


def test_http_backend_retries(monkeypatch):
    calls = {"post": 0, "get": 0}

    class Resp:
        def __init__(self, data=None):
            self._data = data or []

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    def fail_then_ok_post(url, json=None, timeout=5):
        calls["post"] += 1
        if calls["post"] == 1:
            raise RuntimeError("boom")
        return Resp()

    def fail_then_ok_get(url, params=None, timeout=5):
        calls["get"] += 1
        if calls["get"] == 1:
            raise RuntimeError("boom")
        return Resp([["a", "ok"]])

    monkeypatch.setattr(psb.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        psb,
        "requests",
        types.SimpleNamespace(post=fail_then_ok_post, get=fail_then_ok_get),
    )

    be = psb.HTTPPatchScoreBackend("http://x")
    be.store({"description": "a"})
    rows = be.fetch_recent(1)
    assert calls["post"] == 2
    assert calls["get"] == 2
    assert rows == [("a", "ok")]


def test_s3_backend_retries(monkeypatch):
    counts = {"put": 0, "list": 0, "get": 0}

    class FakeClient:
        def put_object(self, Bucket=None, Key=None, Body=None):
            counts["put"] += 1
            if counts["put"] == 1:
                raise RuntimeError("fail")
            return None

        def list_objects_v2(self, Bucket=None, Prefix=None):
            counts["list"] += 1
            if counts["list"] == 1:
                raise RuntimeError("fail")
            return {"Contents": [{"Key": "k", "LastModified": 1}]}

        def get_object(self, Bucket=None, Key=None):
            counts["get"] += 1
            if counts["get"] == 1:
                raise RuntimeError("fail")
            return {"Body": types.SimpleNamespace(read=lambda: b'["x","ok"]')}

    fake_boto3 = types.SimpleNamespace(client=lambda *_a, **_k: FakeClient())
    monkeypatch.setattr(psb, "boto3", fake_boto3)
    monkeypatch.setattr(psb.time, "sleep", lambda *_: None)

    be = psb.S3PatchScoreBackend("b", "p")
    be.store({"d": 1})
    rows = be.fetch_recent(1)

    assert counts["put"] == 2
    assert counts["list"] == 2
    assert counts["get"] == 2
    assert rows == [("x", "ok")]


def test_sandbox_backend_retries(monkeypatch, tmp_path):
    calls = {"store": 0, "fetch": 0}

    class FlakyBackend(psb.PatchScoreBackend):
        def store(self, rec):
            calls["store"] += 1
            if calls["store"] == 1:
                raise RuntimeError("fail")

        def fetch_recent(self, limit=20):
            calls["fetch"] += 1
            if calls["fetch"] == 1:
                raise RuntimeError("fail")
            return [("x", "ok")]

    monkeypatch.setenv("PATCH_SCORE_BACKEND_URL", "http://x")
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr(psb, "backend_from_url", lambda u: FlakyBackend())
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_: None)

    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), audit_trail=DummyTrail()
    )
    dbg._log_patch("d", "ok")
    rows = dbg.recent_scores(1)

    assert calls["store"] == 2
    assert calls["fetch"] == 2
    assert rows == [("x", "ok")]


def test_sandbox_backend_unreachable_warns(monkeypatch, caplog, tmp_path):
    class FailBackend(psb.PatchScoreBackend):
        def store(self, rec):
            raise RuntimeError("boom")

        def fetch_recent(self, limit=20):
            raise RuntimeError("boom")

    monkeypatch.setenv("PATCH_SCORE_BACKEND_URL", "http://x")
    monkeypatch.setenv("SANDBOX_SCORE_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr(psb, "backend_from_url", lambda u: FailBackend())
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_: None)

    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder(), audit_trail=DummyTrail()
    )
    caplog.set_level(logging.WARNING)
    dbg._log_patch("d", "ok")
    rows = dbg.recent_scores(1)

    assert "patch score backend unreachable" in caplog.text
    assert rows[0][:2] == ("d", "ok")


def _fail(*_a, **_k):
    raise RuntimeError("boom")


def test_http_backend_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(psb.time, "sleep", lambda *_: None)
    monkeypatch.setattr(psb, "requests", types.SimpleNamespace(post=_fail, get=_fail))
    be = psb.HTTPPatchScoreBackend("http://x", fallback_dir=str(tmp_path))
    be.store(["z", "ok"])
    rows = be.fetch_recent(1)
    assert rows == [("z", "ok")]


def test_http_backend_fallback_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PATCH_SCORE_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setattr(psb.time, "sleep", lambda *_: None)
    monkeypatch.setattr(psb, "requests", types.SimpleNamespace(post=_fail, get=_fail))
    be = psb.HTTPPatchScoreBackend("http://x")
    be.store(["q", "ok"])
    rows = be.fetch_recent(1)
    assert rows == [("q", "ok")]
