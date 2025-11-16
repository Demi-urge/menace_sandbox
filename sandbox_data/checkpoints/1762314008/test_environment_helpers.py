import os
import sys
import types
import asyncio
import json
import yaml
import logging
import subprocess
import textwrap
from pathlib import Path
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "sqlalchemy" not in sys.modules:
    sa = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    engine_mod.Engine = object
    sa.engine = engine_mod
    sys.modules["sqlalchemy"] = sa
    sys.modules["sqlalchemy.engine"] = engine_mod
if "pyroute2" not in sys.modules:
    pr2 = types.ModuleType("pyroute2")
    pr2.IPRoute = pr2.NSPopen = pr2.netns = object
    sys.modules["pyroute2"] = pr2
sys.modules.pop("sandbox_runner", None)
import sandbox_runner.environment as env  # noqa: E402
from menace.sandbox_runner import test_harness as th  # noqa: E402


def test_parse_failure_modes():
    assert env._parse_failure_modes("disk,network") == {"disk", "network"}
    assert env._parse_failure_modes(["cpu_spike", "memory"]) == {"cpu_spike", "memory"}
    assert env._parse_failure_modes("hostile_input") == {"hostile_input"}


def test_inject_failure_modes_disk():
    snippet = "open('f','w').write('x')"
    out = env._inject_failure_modes(snippet, {"disk"})
    assert "_orig_open" in out
    assert "open('f','w').write('x')" in out


def test_inject_failure_modes_hostile(monkeypatch):
    monkeypatch.setenv("SANDBOX_INPUT_STUBS", "")
    monkeypatch.setenv("x", "1")
    out = env._inject_failure_modes("", {"hostile_input"})
    exec(out, {})
    data = os.environ.get("SANDBOX_INPUT_STUBS", "")
    assert "' OR '1'='1" in data or "<script>" in data or len(data) > 1000


def test_inject_failure_modes_user_misuse(capsys):
    out = env._inject_failure_modes("print('ok')", {"user_misuse"})
    exec(out, {})
    captured = capsys.readouterr()
    assert "len(" in captured.err
    assert captured.out.strip() == "ok"


def test_section_worker_user_misuse(monkeypatch):
    monkeypatch.setattr(env.time, "sleep", lambda s: None)
    res, _ = asyncio.run(
        env._section_worker("print('run')", {"FAILURE_MODES": ["user_misuse"]}, 0.0)
    )
    assert res["exit_code"] == 0
    assert "run" in res["stdout"]
    assert "len(" in res["stderr"]


def test_generate_presets_concurrency(monkeypatch):
    import menace.environment_generator as gen

    monkeypatch.setattr(gen, "_select_failures", lambda: ["concurrency_spike"])
    monkeypatch.setattr(gen.random, "choice", lambda seq: seq[0])
    presets = gen.generate_presets(1)
    preset = presets[0]
    assert preset["FAILURE_MODES"] == "concurrency_spike"
    assert "THREAD_BURST" in preset and "ASYNC_TASK_BURST" in preset


def test_section_worker_concurrency_spike():
    res, _ = asyncio.run(
        env._section_worker(
            "print('run')",
            {
                "FAILURE_MODES": ["concurrency_spike"],
                "THREAD_BURST": "5",
                "ASYNC_TASK_BURST": "5",
            },
            0.0,
        )
    )
    assert res["exit_code"] == 0
    assert res.get("concurrency_threads", 0) >= 5
    assert res.get("concurrency_tasks", 0) >= 5


def test_generate_input_stubs_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_INPUT_STUBS", '[{"a": 1}]')
    import importlib

    importlib.reload(env)
    stubs = env.generate_input_stubs()
    assert stubs == [{"a": 1}]


def test_generate_input_stubs_random(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "random")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib

    importlib.reload(env)
    monkeypatch.setattr(env.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(env.random, "randint", lambda a, b: a)
    monkeypatch.setattr(env.random, "random", lambda: 0.0)
    stubs = env.generate_input_stubs(2)
    assert len(stubs) == 2
    for stub in stubs:
        assert stub.get("mode") == "default" and stub.get("level") == 1
        assert "flag" in stub


def test_generate_input_stubs_templates(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    f = tmp_path / "temps.json"
    f.write_text('[{"mode": "x", "level": 9}]')
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "templates")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", str(f))
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib

    importlib.reload(env)
    stubs = env.generate_input_stubs(1)
    assert stubs == [{"mode": "x", "level": 9}]


def test_generate_input_stubs_smart(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "smart")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib

    importlib.reload(env)
    if env._FAKER is not None:
        monkeypatch.setattr(env._FAKER, "random_int", lambda *a, **k: 42)

    def target(a: int, email: str, flag: bool = False) -> None:
        pass

    stubs = env.generate_input_stubs(1, target=target)
    stub = stubs[0]
    assert isinstance(stub["a"], int) and stub["a"] == 42
    assert isinstance(stub["email"], str) and stub["email"]
    assert stub["flag"] is False


def test_generate_input_stubs_smart_no_faker(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "smart")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib

    importlib.reload(env)
    monkeypatch.setattr(env, "_FAKER", None)
    monkeypatch.setattr(env, "_hyp_strats", None)

    def target(a: int, name: str) -> None:
        pass

    stubs = env.generate_input_stubs(1, target=target)
    stub = stubs[0]
    assert stub["a"] == 0
    assert stub["name"] == ""


def test_generate_input_stubs_synthetic_plugin(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "synthetic")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    monkeypatch.setenv(
        "SANDBOX_STUB_PLUGINS", "sandbox_runner.generative_stub_provider"
    )
    import importlib
    import sandbox_runner.generative_stub_provider as gsp

    gsp.build_prompt = (
        lambda query, *, intent_metadata=None, context_builder, **kwargs: context_builder.build_prompt(  # type: ignore[assignment]
            query, intent_metadata=intent_metadata, **kwargs
        )
    )

    gsp = importlib.reload(gsp)

    class DummyGen:
        def __init__(self):
            self.prompts = []

        def __call__(
            self, prompt, max_length=64, num_return_sequences=1, *, context_builder=None
        ):
            self.prompts.append(prompt)
            return [{"generated_text": '{"foo": 7}'}]

    dummy = DummyGen()

    async def loader():
        return dummy

    monkeypatch.setattr(gsp, "_aload_generator", loader)
    importlib.reload(env)

    def target(foo: int) -> None:
        pass

    stubs = env.generate_input_stubs(1, target=target)
    assert stubs == [{"foo": 7}]
    assert any("foo=" in p and "target" in p for p in dummy.prompts)


def test_generative_load_default(monkeypatch):
    monkeypatch.delenv("SANDBOX_STUB_MODEL", raising=False)
    monkeypatch.setenv("SANDBOX_ENABLE_TRANSFORMERS", "1")
    import sandbox_runner.generative_stub_provider as gsp

    called = {}

    def dummy_pipeline(task, **kwargs):
        called["task"] = task
        called.update(kwargs)

        class Gen:
            pass

        return Gen()

    monkeypatch.setattr(gsp, "pipeline", dummy_pipeline)
    monkeypatch.setattr(gsp, "_GENERATOR", None)
    gen = gsp._load_generator()
    assert called["task"] == "text-generation"
    assert called["model"] == "distilgpt2"
    assert called["local_files_only"] is True
    assert gen is not None


def test_generative_model_validation(monkeypatch):
    monkeypatch.setenv("SANDBOX_STUB_MODEL", "unknown")

    class DummyEP:
        def __init__(self, name):
            self.name = name

    def dummy_entry_points(*args, **kwargs):
        if kwargs.get("group") == "sandbox.stub_models":
            return [DummyEP("known")]
        if not args and not kwargs:
            return {"sandbox.stub_models": [DummyEP("known")]}  # legacy API
        return []

    from importlib import metadata

    monkeypatch.setattr(metadata, "entry_points", dummy_entry_points)

    import importlib
    import sys

    sys.modules.pop("sandbox_runner.generative_stub_provider", None)
    with pytest.raises(ValueError):
        importlib.import_module("sandbox_runner.generative_stub_provider")
    sys.modules.pop("sandbox_runner.generative_stub_provider", None)


def test_generate_input_stubs_synthetic_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_STUB_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "synthetic")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    monkeypatch.setenv(
        "SANDBOX_STUB_PLUGINS", "sandbox_runner.generative_stub_provider"
    )
    import importlib
    import sandbox_runner.generative_stub_provider as gsp

    gsp = importlib.reload(gsp)

    async def loader():
        return None

    monkeypatch.setattr(gsp, "_aload_generator", loader)
    importlib.reload(env)

    monkeypatch.setattr(env, "_FAKER", None)
    monkeypatch.setattr(env, "_hyp_strats", None)

    def target(a: int) -> None:
        pass

    stubs = env.generate_input_stubs(1, target=target)
    assert gsp._type_matches(stubs[0]["a"], int)


def test_generate_input_stubs_history_db(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    db_path = tmp_path / "hist.db"
    from sandbox_runner.input_history_db import InputHistoryDB

    db = InputHistoryDB(db_path)
    db.add({"x": 5})

    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "history")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", str(db_path))
    import importlib

    importlib.reload(env)

    stubs = env.generate_input_stubs(1)
    assert stubs == [{"x": 5}]


def test_generate_input_stubs_history_db_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    db_path = tmp_path / "hist.db"
    from sandbox_runner.input_history_db import InputHistoryDB

    InputHistoryDB(db_path)  # create empty db

    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "history")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", str(db_path))
    import importlib

    importlib.reload(env)

    def target(z: int) -> None:
        pass

    stubs = env.generate_input_stubs(1, target=target)
    assert stubs == [{"z": 0}]


def test_generate_input_stubs_history_mean(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    db_path = tmp_path / "hist.db"
    from sandbox_runner.input_history_db import InputHistoryDB

    db = InputHistoryDB(db_path)
    db.add({"level": 1})
    db.add({"level": 3})

    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "history")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", str(db_path))
    import importlib

    importlib.reload(env)

    stubs = env.generate_input_stubs(1)
    assert stubs == [{"level": 2}]


def test_generate_input_stubs_history_common(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    db_path = tmp_path / "hist.db"
    from sandbox_runner.input_history_db import InputHistoryDB

    db = InputHistoryDB(db_path)
    db.add({"mode": "x"})
    db.add({"mode": "y"})
    db.add({"mode": "x"})

    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "history")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", "")
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", str(db_path))
    import importlib

    importlib.reload(env)

    stubs = env.generate_input_stubs(1)
    assert stubs == [{"mode": "x"}]


def test_generate_input_stubs_history_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    db_path = tmp_path / "hist.db"
    from sandbox_runner.input_history_db import InputHistoryDB

    db = InputHistoryDB(db_path)
    db.add({"a": 10})

    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "templates")
    monkeypatch.setenv("SANDBOX_INPUT_TEMPLATES_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", str(db_path))
    import importlib

    importlib.reload(env)

    stubs = env.generate_input_stubs(1)
    assert stubs == [{"a": 10}]


def test_generate_input_stubs_hostile(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "hostile")
    monkeypatch.delenv("SANDBOX_INPUT_TEMPLATES_FILE", raising=False)
    monkeypatch.delenv("SANDBOX_HOSTILE_PAYLOADS", raising=False)
    monkeypatch.delenv("SANDBOX_HOSTILE_PAYLOADS_FILE", raising=False)
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib

    importlib.reload(env)
    stubs = env.generate_input_stubs(1, strategy="hostile")
    val = next(iter(stubs[0].values()))
    assert isinstance(val, str) and ("' OR '1'='1" in val or len(val) > 1000)


def test_hostile_strategy_env_var(monkeypatch, caplog):
    monkeypatch.delenv("SANDBOX_HOSTILE_PAYLOADS_FILE", raising=False)
    monkeypatch.delenv("SANDBOX_INPUT_TEMPLATES_FILE", raising=False)
    monkeypatch.delenv("SANDBOX_INPUT_HISTORY", raising=False)
    monkeypatch.setenv("SANDBOX_HOSTILE_PAYLOADS", json.dumps(["env", {"bad": 1}]))
    with caplog.at_level(logging.WARNING):
        stubs = env._hostile_strategy(1)
    assert stubs == [{"payload": "env"}]
    assert any("invalid hostile payload" in r.message for r in caplog.records)


def test_hostile_strategy_file_yaml(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_HOSTILE_PAYLOADS", raising=False)
    monkeypatch.delenv("SANDBOX_INPUT_TEMPLATES_FILE", raising=False)
    monkeypatch.delenv("SANDBOX_INPUT_HISTORY", raising=False)
    data = ["file", 1]
    path = tmp_path / "payloads.yaml"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("SANDBOX_HOSTILE_PAYLOADS_FILE", str(path))
    stubs = env._hostile_strategy(2)
    assert stubs[0]["payload"] == "file"


def test_generate_input_stubs_misuse(monkeypatch):
    monkeypatch.delenv("SANDBOX_INPUT_STUBS", raising=False)
    monkeypatch.setenv("SANDBOX_STUB_STRATEGY", "misuse")
    monkeypatch.delenv("SANDBOX_INPUT_TEMPLATES_FILE", raising=False)
    monkeypatch.setenv("SANDBOX_INPUT_HISTORY", "")
    import importlib

    importlib.reload(env)

    def target(a: int, name: str) -> None:
        pass

    stubs = env.generate_input_stubs(1, target=target)
    stub = stubs[0]
    assert (
        "a" not in stub
        or not isinstance(stub.get("a"), int)
        or "name" not in stub
        or not isinstance(stub.get("name"), str)
    )


def _stub_docker_logs(holder):
    class DummyExec:
        def __init__(self):
            self.exit_code = 0
            self.output = (b"out", b"err")

    class DummyContainer:
        id = "dummy"
        status = "running"

        def exec_run(self, *a, **k):
            return DummyExec()

        def wait(self):
            return {"StatusCode": 0}

        def stats(self, stream=False):
            return {
                "blkio_stats": {"io_service_bytes_recursive": []},
                "cpu_stats": {"cpu_usage": {"total_usage": 1}},
                "memory_stats": {"max_usage": 1},
                "networks": {},
            }

        def remove(self):
            holder.append("removed")

        def stop(self, timeout=0):
            pass

        def reload(self):
            pass

    class DummyContainers:
        def run(self, image, cmd, **kwargs):
            holder.append(image)
            return DummyContainer()

    class DummyClient:
        containers = DummyContainers()

    dummy = types.ModuleType("docker")
    dummy.from_env = lambda: DummyClient()
    dummy.types = types
    err_mod = types.ModuleType("docker.errors")

    class DummyErr(Exception):
        pass

    err_mod.DockerException = DummyErr
    err_mod.APIError = DummyErr
    dummy.errors = err_mod
    sys.modules["docker.errors"] = err_mod
    sys.modules["docker"] = dummy


def test_execute_in_container_logs_created(monkeypatch):
    calls = []
    _stub_docker_logs(calls)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None

    res = asyncio.run(env._execute_in_container("print('hi')", {}))
    assert res["exit_code"] == 0.0
    assert os.path.exists(res["stdout_log"])
    assert os.path.exists(res["stderr_log"])

    env._cleanup_pools()
    assert not os.path.exists(res["stdout_log"])
    assert not os.path.exists(res["stderr_log"])


def test_harness_failure_frames(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("")
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    tests_dir.joinpath("test_mod.py").write_text(  # path-ignore
        textwrap.dedent(
            """
            def helper():
                assert False

            def test_fail():
                helper()
            """
        )
    )
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)

    monkeypatch.setattr(th, "_python_bin", lambda v: Path(sys.executable))
    result = th.run_tests(repo)
    if isinstance(result, list):
        result = result[0]
    assert not result.success
    assert result.failure is not None
    assert result.failure["function"] == "helper"
    assert result.failure["line"] == "3"
    assert result.failure["file"].endswith("test_mod.py")  # path-ignore
    frames = result.failure["frames"]
    assert frames[-1]["function"] == "helper"
    assert frames[-2]["function"] == "test_fail"
