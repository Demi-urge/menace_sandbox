# flake8: noqa
import asyncio
import importlib.util
import logging
import json
import os
import sys
import types
import time
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))
sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
od_stub = types.ModuleType("sandbox_runner.orphan_discovery")
od_stub.append_orphan_cache = lambda *a, **k: None
od_stub.append_orphan_classifications = lambda *a, **k: None
od_stub.prune_orphan_cache = lambda *a, **k: None
od_stub.load_orphan_cache = lambda *a, **k: set()
sys.modules.setdefault("sandbox_runner.orphan_discovery", od_stub)
sys.modules.setdefault("orphan_discovery", od_stub)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",  # path-ignore
)
mod = importlib.util.module_from_spec(spec)
import sys  # noqa: E402

pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
spec.loader.exec_module(mod)
import types  # noqa: E402

import menace.error_bot as eb  # noqa: E402
from menace.data_bot import DataBot, MetricsDB  # noqa: E402


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        if k.get("return_metadata"):
            return "", {}
        return ""


def test_context_builder_required():
    with pytest.raises(TypeError):
        mod.SelfTestService()
    with pytest.raises(ValueError):
        mod.SelfTestService(context_builder=None)


def test_scheduler_start(monkeypatch):
    calls: list[None] = []

    async def fake_run_once():
        calls.append(None)

    svc = mod.SelfTestService(context_builder=DummyBuilder())
    monkeypatch.setattr(svc, "_run_once", fake_run_once)

    async def runner():
        loop = asyncio.get_running_loop()
        svc.run_continuous(interval=0.01, loop=loop)
        await asyncio.sleep(0.03)
        await svc.stop()

    asyncio.run(runner())
    assert calls


def test_failure_logs_telemetry(tmp_path, monkeypatch, caplog):
    db = eb.ErrorDB(tmp_path / "e.db")
    svc = mod.SelfTestService(db, context_builder=DummyBuilder())

    async def fail_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 0, "failed": 1}}, fh)

        class P:
            returncode = 1

            async def communicate(self):
                return b"out-msg", b"err-msg"

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_exec)
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert "self test run failed" in caplog.text
    cur = db.conn.execute("SELECT COUNT(*) FROM telemetry")
    assert cur.fetchone()[0] == 1
    row = db.conn.execute("SELECT passed, failed FROM test_results").fetchone()
    assert row == (0, 1)
    assert "out-msg" in svc.results.get("stdout", "")
    assert "err-msg" in svc.results.get("stderr", "")


def test_success_logs_results(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e2.db")
    svc = mod.SelfTestService(db, context_builder=DummyBuilder())

    async def succeed_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 3, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", succeed_exec)
    svc.run_once()
    cur = db.conn.execute("SELECT COUNT(*) FROM telemetry")
    assert cur.fetchone()[0] == 0
    row = db.conn.execute("SELECT passed, failed FROM test_results").fetchone()
    assert row == (3, 0)


def test_custom_args(monkeypatch):
    recorded = {}

    async def fake_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 0, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    svc = mod.SelfTestService(pytest_args="-k pattern", context_builder=DummyBuilder())
    svc.run_once()
    assert any("-k" in str(x) for x in recorded["cmd"]) and any(
        "pattern" in str(x) for x in recorded["cmd"]
    )


def test_parallel_workers(monkeypatch):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 0, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    svc = mod.SelfTestService(pytest_args="path1 path2", workers=2, context_builder=DummyBuilder())
    svc.run_once()
    assert len(calls) == 2
    for cmd in calls:
        assert "-n" in cmd and "2" in cmd


def test_records_coverage_and_runtime(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e3.db")
    metrics = MetricsDB(tmp_path / "m.db")
    data_bot = DataBot(metrics)
    svc = mod.SelfTestService(db, data_bot=data_bot, context_builder=DummyBuilder())

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {"passed": 1, "failed": 0, "duration": 1.0},
                        "coverage": {"percent": 80.0},
                        "duration": 1.0,
                    },
                    fh,
                )

        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    svc.run_once()
    assert svc.results["coverage"] == 80.0
    assert svc.results["runtime"] == 1.0
    rows = metrics.fetch_eval("self_tests")
    metrics_names = [r[1] for r in rows]
    assert "coverage" in metrics_names and "runtime" in metrics_names


def test_json_pipe_when_callback(monkeypatch):
    recorded = {}

    async def fake_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(
                    json.dumps({"summary": {"passed": 0, "failed": 0}}).encode()
                )
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    svc = mod.SelfTestService(result_callback=lambda r: None, context_builder=DummyBuilder())
    svc.run_once()
    assert any("--json-report-file=-" in str(x) for x in recorded["cmd"])


def test_callback_emits_partial_results(monkeypatch):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(
                    json.dumps({"summary": {"passed": 1, "failed": 0}}).encode()
                )
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    svc = mod.SelfTestService(
        pytest_args="a b",
        result_callback=lambda r: calls.append(r.copy()),
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    # two partial results plus final summary
    assert len(calls) == 3
    passes = sorted(c["passed"] for c in calls[:-1])
    assert passes == [1, 2]
    assert calls[-1] == svc.results


def test_container_exec(monkeypatch):
    recorded = {}

    async def fake_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(
                    json.dumps({"summary": {"passed": 0, "failed": 0}}).encode()
                )
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True
    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)

    svc = mod.SelfTestService(use_container=True, container_image="img", context_builder=DummyBuilder())
    svc.run_once()
    assert recorded["cmd"][0] == "docker"
    assert "img" in recorded["cmd"]
    assert any(f"{os.getcwd()}:{os.getcwd()}:ro" in str(x) for x in recorded["cmd"])
    assert "pytest" in recorded["cmd"]
    assert any("--json-report-file=-" in str(x) for x in recorded["cmd"])


def test_container_metrics(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e4.db")
    metrics = MetricsDB(tmp_path / "m2.db")
    data_bot = DataBot(metrics)

    async def fake_exec(*cmd, **kwargs):
        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(
                    json.dumps(
                        {
                            "summary": {"passed": 1, "failed": 0, "duration": 1.0},
                            "coverage": {"percent": 75.0},
                            "duration": 1.0,
                        }
                    ).encode()
                )
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True
    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)

    svc = mod.SelfTestService(
        db, data_bot=data_bot, use_container=True, container_image="img", context_builder=DummyBuilder()
    )
    svc.run_once()
    assert svc.results["coverage"] == 75.0
    assert svc.results["runtime"] == 1.0
    names = [r[1] for r in metrics.fetch_eval("self_tests")]
    assert "coverage" in names and "runtime" in names


def test_container_fallback(monkeypatch):
    recorded = {}

    async def fake_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(
                    json.dumps({"summary": {"passed": 0, "failed": 0}}).encode()
                )
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return False

    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)

    svc = mod.SelfTestService(use_container=True, context_builder=DummyBuilder())
    svc.run_once()
    assert recorded["cmd"][0] != "docker"


def test_container_runtime_and_host(monkeypatch):
    recorded = []

    async def fake_exec(*cmd, **kwargs):
        recorded.append(cmd)

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(json.dumps({"summary": {"passed": 0, "failed": 0}}).encode())
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)

    svc = mod.SelfTestService(
        use_container=True,
        container_image="img",
        container_runtime="podman",
        docker_host="ssh://host",
        context_builder=DummyBuilder(),
    )
    svc.run_once()
    cmd = recorded[0]
    assert cmd[0] == "podman"
    assert "--url" in cmd and "ssh://host" in cmd


def test_container_workers_split(monkeypatch):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(json.dumps({"summary": {"passed": 0, "failed": 0}}).encode())
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)

    svc = mod.SelfTestService(
        pytest_args="a b",
        workers=4,
        use_container=True,
        container_image="img",
        context_builder=DummyBuilder(),
    )
    svc.run_once()
    assert len(calls) == 2
    for c in calls:
        assert "-n" in c and "2" in c


def _make_fake_exec(passed: int, failed: int):
    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": passed, "failed": failed}}, fh)

        class P:
            returncode = 0

            async def wait(self):
                return None

        return P()

    return fake_exec


def test_json_history_persistence(tmp_path, monkeypatch):
    path = tmp_path / "hist.json"
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _make_fake_exec(2, 0))
    svc = mod.SelfTestService(history_path=path, context_builder=DummyBuilder())
    svc.run_once()
    hist = svc.recent_history(1)
    assert hist and hist[0]["passed"] == 2


def test_sqlite_history_persistence(tmp_path, monkeypatch):
    path = tmp_path / "hist.db"
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _make_fake_exec(1, 1))
    svc = mod.SelfTestService(history_path=path, context_builder=DummyBuilder())
    svc.run_once()
    hist = svc.recent_history(1)
    assert hist and hist[0]["failed"] == 1


def test_run_files_concurrently(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                await asyncio.sleep(0.05)
                self.stdout.feed_data(json.dumps({"summary": {"passed": 0, "failed": 0}}).encode())
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    svc = mod.SelfTestService(pytest_args="a b", result_callback=lambda r: None, context_builder=DummyBuilder())
    start = time.perf_counter()
    svc.run_once()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1


def test_offline_container_load(monkeypatch, tmp_path):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                if "load" not in cmd:
                    self.stdout.feed_data(
                        json.dumps({"summary": {"passed": 0, "failed": 0}}).encode()
                    )
                    self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)

    tar_path = tmp_path / "img.tar"
    tar_path.write_text("dummy")
    monkeypatch.setenv("MENACE_OFFLINE_INSTALL", "1")
    monkeypatch.setenv("MENACE_SELF_TEST_IMAGE_TAR", str(tar_path))

    svc = mod.SelfTestService(use_container=True, container_image="img", context_builder=DummyBuilder())
    svc.run_once()

    assert any(c[0] == "docker" and "load" in c for c in calls)


def test_container_failure_logs_identifiers(monkeypatch, caplog):
    recorded = {}

    async def fail_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd
        if "logs" in cmd:
            class P:
                returncode = 0

                async def communicate(self):
                    return b"container logs", b""

                async def wait(self):
                    return None

            return P()

        class P:
            returncode = 1

            async def communicate(self):
                return b"out", b"err"

            async def wait(self):
                return None

        return P()

    async def avail(self):
        return True

    async def dummy(*a, **k):
        return None

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_exec)
    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)
    monkeypatch.setattr(mod.SelfTestService, "_remove_stale_containers", dummy)
    monkeypatch.setattr(mod.SelfTestService, "_force_remove_container", dummy)
    monkeypatch.setattr(mod.uuid, "uuid4", lambda: types.SimpleNamespace(hex="deadbeef"))

    svc = mod.SelfTestService(use_container=True, container_image="img", container_retries=0, context_builder=DummyBuilder())
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert "self test run failed" in caplog.text

    err = svc.results.get("stderr", "")
    assert "selftest_deadbeef" in err
    assert "docker" in err
    assert "container logs" in svc.results.get("logs", "")


def test_cli_cleanup(monkeypatch):
    called = []

    async def fake_remove(self):
        called.append(True)

    async def avail(self):
        return True

    monkeypatch.setattr(mod.SelfTestService, "_remove_stale_containers", fake_remove)
    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)

    mod.cli(["cleanup"])

    assert called


def test_gauge_updates(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {"passed": 3, "failed": 1, "duration": 2.0},
                        "coverage": {"percent": 90.0},
                        "duration": 2.0,
                    },
                    fh,
                )

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    mod.self_test_passed_total.set(0)
    mod.self_test_failed_total.set(0)
    mod.self_test_average_runtime_seconds.set(0)
    mod.self_test_average_coverage.set(0)
    svc = mod.SelfTestService(context_builder=DummyBuilder())
    svc.run_once()

    def _get(gauge):
        if hasattr(gauge, "_value"):
            return gauge._value.get()
        return gauge.labels().get()

    assert _get(mod.self_test_passed_total) == 3
    assert _get(mod.self_test_failed_total) == 1
    assert _get(mod.self_test_average_runtime_seconds) == 2.0
    assert _get(mod.self_test_average_coverage) == 90.0
    assert "suite_metrics" in svc.results
    first = next(iter(svc.results["suite_metrics"].values()))
    assert first["coverage"] == 90.0
    assert first["runtime"] == 2.0


def test_timeout_metric(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        class P:
            returncode = 0

            def kill(self):
                pass

            async def communicate(self):
                await asyncio.sleep(0.05)
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    async def dummy(self, *a, **k):
        return None

    monkeypatch.setattr(mod.SelfTestService, "_docker_available", avail)
    monkeypatch.setattr(mod.SelfTestService, "_force_remove_container", dummy)

    mod.self_test_container_timeouts_total.set(0)
    svc = mod.SelfTestService(use_container=True, container_timeout=0.01, container_retries=0, context_builder=DummyBuilder())
    svc.run_once()

    def _get(gauge):
        if hasattr(gauge, "_value"):
            return gauge._value.get()
        return gauge.labels().get()

    assert _get(mod.self_test_container_timeouts_total) == 1


def test_summary_artifact_on_failure(monkeypatch, tmp_path):
    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {"passed": 0, "failed": 1, "duration": 0.5},
                        "coverage": {"percent": 50.0},
                        "duration": 0.5,
                    },
                    fh,
                )

        class P:
            returncode = 1

            async def communicate(self):
                return b"", b"err"

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    svc = mod.SelfTestService(report_dir=tmp_path, context_builder=DummyBuilder())
    with pytest.raises(RuntimeError):
        svc.run_once()
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["failed"] == 1


def test_cli_report(tmp_path, capsys):
    report = tmp_path / "report_1.json"
    report.write_text(json.dumps({"hello": "world"}))
    rc = mod.cli(["report", "--report-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hello" in out
