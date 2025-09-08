import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "ctx", "s", {}


spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",  # path-ignore
)
mod = importlib.util.module_from_spec(spec)
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
settings_mod = types.ModuleType("sandbox_settings")
settings_mod.SandboxSettings = lambda: types.SimpleNamespace(
    sandbox_central_logging=False, menace_light_imports=True
)
sys.modules["sandbox_settings"] = settings_mod
sys.modules["menace.sandbox_settings"] = settings_mod
log_mod = types.ModuleType("logging_utils")
log_mod.setup_logging = lambda: None
log_mod.log_record = lambda **k: {}
log_mod.get_logger = lambda *a, **k: None
sys.modules["logging_utils"] = log_mod
sys.modules["menace.logging_utils"] = log_mod
vec_mod = types.ModuleType("vector_service.context_builder")
vec_mod.ContextBuilder = DummyBuilder
sys.modules["vector_service.context_builder"] = vec_mod
router_mod = types.ModuleType("db_router")
class _Router:
    def __init__(self, *a, **k):
        pass

    def get_connection(self, name):
        return None

router_mod.DBRouter = _Router
router_mod.GLOBAL_ROUTER = None
router_mod.LOCAL_TABLES = {}
router_mod.init_db_router = lambda *a, **k: _Router()
sys.modules["db_router"] = router_mod
from prometheus_client import REGISTRY
REGISTRY._names_to_collectors.clear()
db_mod = types.ModuleType("data_bot")
db_mod.DataBot = object
sys.modules["data_bot"] = db_mod
sys.modules["menace.data_bot"] = db_mod
err_db_mod = types.ModuleType("error_bot")
class _ErrDB:
    def __init__(self, *a, **k):
        pass
err_db_mod.ErrorDB = _ErrDB
sys.modules["error_bot"] = err_db_mod
sys.modules["menace.error_bot"] = err_db_mod
err_log_mod = types.ModuleType("error_logger")
class _ErrLogger:
    def __init__(self, *a, **k):
        pass
err_log_mod.ErrorLogger = _ErrLogger
sys.modules["error_logger"] = err_log_mod
sys.modules["menace.error_logger"] = err_log_mod
ae_mod = types.ModuleType("auto_env_setup")
ae_mod.get_recursive_isolated = lambda: True
ae_mod.set_recursive_isolated = lambda val: None
sys.modules["auto_env_setup"] = ae_mod
sys.modules["menace.auto_env_setup"] = ae_mod
spec.loader.exec_module(mod)
# simplify redundancy checks for tests
mod.analyze_redundancy = lambda p: False


def test_include_orphans(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "orphan_modules.json").write_text(json.dumps(["foo.py", "bar.py"]))  # path-ignore

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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = lambda repo, module_map=None: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    svc.run_once()

    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)  # path-ignore
    assert any("bar.py" in c for c in joined)  # path-ignore
    assert svc.results.get("orphan_total") == 2
    assert svc.results.get("orphan_failed") == 0


def test_auto_discover_orphans(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types
    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = (
        lambda repo, module_map=None: {"foo": [], "bar": []}
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    svc = mod.SelfTestService(context_builder=DummyBuilder())
    svc.run_once()

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert set(data) == {"foo.py", "bar.py"}  # path-ignore
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)  # path-ignore
    assert any("bar.py" in c for c in joined)  # path-ignore


def test_discover_orphans_option(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = (
        lambda repo, module_map=None: {"foo": [], "bar": []}
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    svc = mod.SelfTestService(discover_orphans=True, context_builder=DummyBuilder())
    svc.run_once()

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert set(data) == {"foo.py", "bar.py"}  # path-ignore
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)  # path-ignore
    assert any("bar.py" in c for c in joined)  # path-ignore


def test_discover_isolated_option(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    mod_iso = types.ModuleType("scripts.discover_isolated_modules")
    mod_iso.discover_isolated_modules = lambda root, *, recursive=True: ["foo.py", "bar.py"]  # path-ignore
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod_iso
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod_iso)
    monkeypatch.setitem(sys.modules, "scripts", pkg)

    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = lambda repo, module_map=None: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    svc = mod.SelfTestService(discover_isolated=True, context_builder=DummyBuilder())
    svc.run_once()

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py", "bar.py"]  # path-ignore
    joined = [" ".join(map(str, c)) for c in calls]
    assert any("foo.py" in c for c in joined)  # path-ignore
    assert any("bar.py" in c for c in joined)  # path-ignore


def test_recursive_option_used(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    async def fake_exec(*cmd, **kwargs):
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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    calls = {}

    def discover(repo, module_map=None):
        calls["used"] = True
        calls["repo"] = repo
        calls["map"] = module_map
        return {"foo": []}

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", types.ModuleType("env"))

    svc = mod.SelfTestService(context_builder=DummyBuilder())
    svc.run_once(refresh_orphans=True)

    assert calls.get("used") is True
    assert Path(calls["repo"]) == tmp_path
    assert Path(calls["map"]).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


def test_discover_orphans_append(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    async def fake_exec(*cmd, **kwargs):
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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    seq = [{"foo": []}, {"bar": []}]
    calls = []

    def discover(repo, module_map=None):
        calls.append((repo, module_map))
        return seq.pop(0)

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", types.ModuleType("env"))

    svc = mod.SelfTestService(
        include_orphans=False,
        discover_orphans=True,
        context_builder=DummyBuilder(),
    )
    svc.run_once()
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py"]  # path-ignore

    svc.run_once()
    data2 = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data2) == ["bar.py", "foo.py"]  # path-ignore
    assert len(calls) == 2
    for repo, m in calls:
        assert Path(repo) == tmp_path
        assert Path(m).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


def test_recursive_chain_modules(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import c\n")  # path-ignore
    (tmp_path / "c.py").write_text("x = 1\n")  # path-ignore

    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(" ".join(map(str, cmd)))
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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    params = {}

    def discover(repo, module_map=None):
        params.setdefault("repo", repo)
        params.setdefault("map", module_map)
        return {"a": [], "b": ["a"], "c": ["b"]}

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    svc = mod.SelfTestService(discover_orphans=False, context_builder=DummyBuilder())
    svc.run_once()

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data) == ["a.py", "b.py", "c.py"]  # path-ignore
    joined = "\n".join(calls)
    assert "a.py" in joined  # path-ignore
    assert "b.py" in joined  # path-ignore
    assert "c.py" in joined  # path-ignore
    assert Path(params["repo"]) == tmp_path
    assert Path(params["map"]).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


def test_recursive_orphan_multi_scan(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    calls = []
    discover_calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(" ".join(map(str, cmd)))
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

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.chdir(tmp_path)

    import types

    def discover(repo, module_map=None):
        discover_calls.append((repo, module_map))
        return {"a": [], "b": []}

    helper = types.ModuleType("sandbox_runner")
    helper.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", helper)

    svc = mod.SelfTestService(discover_orphans=False, context_builder=DummyBuilder())
    svc.run_once(refresh_orphans=True)

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert sorted(data) == ["a.py", "b.py"]  # path-ignore
    joined = "\n".join(calls)
    assert "a.py" in joined  # path-ignore
    assert "b.py" in joined  # path-ignore
    assert len(discover_calls) == 1
    repo, m = discover_calls[0]
    assert Path(repo) == tmp_path
    assert Path(m).resolve() == (tmp_path / "sandbox_data" / "module_map.json").resolve()


def test_env_disables_recursive_orphans(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SELF_TEST_RECURSIVE_ORPHANS", "0")

    import types

    called = {}

    def discover(repo, module_map=None):
        called["used"] = True
        return {"foo": []}

    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    find_mod = types.ModuleType("scripts.find_orphan_modules")
    find_mod.find_orphan_modules = lambda root: [tmp_path / "foo.py"]  # path-ignore
    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.find_orphan_modules = find_mod.find_orphan_modules
    monkeypatch.setitem(sys.modules, "scripts.find_orphan_modules", find_mod)
    monkeypatch.setitem(sys.modules, "scripts", scripts_pkg)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    svc = mod.SelfTestService(context_builder=DummyBuilder())
    modules = svc._discover_orphans()
    assert modules == ["foo.py"]  # path-ignore
    assert called.get("used") is None


def test_discover_orphans_skips_redundant(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("# deprecated\n")  # path-ignore

    import types

    def discover(repo, module_map=None):
        assert Path(repo) == tmp_path
        return {"a": [], "b": ["a"]}

    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = discover
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    def fake_analyze(path: Path) -> bool:
        return path.name == "b.py"  # path-ignore

    monkeypatch.setattr(mod, "analyze_redundancy", fake_analyze)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    svc = mod.SelfTestService(context_builder=DummyBuilder())
    svc.logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)
    modules = svc._discover_orphans()
    assert modules == ["a.py"]  # path-ignore
    assert svc.orphan_traces["b.py"]["parents"] == ["a.py"]  # path-ignore
    assert svc.orphan_traces["b.py"]["redundant"] is True  # path-ignore


@pytest.mark.parametrize("var", ["SELF_TEST_DISABLE_ORPHANS", "SANDBOX_DISABLE_ORPHANS"])
def test_env_orphans_disabled(tmp_path, monkeypatch, var):
    (tmp_path / "sandbox_data").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(var, "1")

    svc = mod.SelfTestService(context_builder=DummyBuilder())

    assert svc.include_orphans is False


async def _fake_proc(*cmd, **kwargs):
    class P:
        returncode = 0

        async def communicate(self):
            return b"", b""

        async def wait(self):
            return None

    return P()


def _setup_isolated(monkeypatch):
    import types

    called = {}

    def discover(root, *, recursive=True):
        called["recursive"] = recursive
        return ["foo.py"]  # path-ignore

    mod_iso = types.ModuleType("scripts.discover_isolated_modules")
    mod_iso.discover_isolated_modules = discover
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod_iso
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod_iso)
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = lambda repo, module_map=None: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_proc)
    return called


def test_recursive_isolated_setting(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    monkeypatch.chdir(tmp_path)

    class DummySettings:
        auto_include_isolated = False
        recursive_isolated = False

    monkeypatch.setattr(mod, "SandboxSettings", lambda: DummySettings())

    called = _setup_isolated(monkeypatch)
    svc = mod.SelfTestService(discover_isolated=True, context_builder=DummyBuilder())
    svc.run_once()

    assert called.get("recursive") is False
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py"]  # path-ignore


def test_recursive_isolated_arg(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    monkeypatch.chdir(tmp_path)

    called = _setup_isolated(monkeypatch)
    svc = mod.SelfTestService(
        discover_isolated=True,
        recursive_isolated=True,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    assert called.get("recursive") is True
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == ["foo.py"]  # path-ignore


def test_isolated_cleanup_passed(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    (tmp_path / "foo.py").write_text("x = 1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 1, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    import types

    mod_iso = types.ModuleType("scripts.discover_isolated_modules")
    mod_iso.discover_isolated_modules = lambda root, *, recursive=True: ["foo.py"]  # path-ignore
    pkg = types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod_iso
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod_iso)
    monkeypatch.setitem(sys.modules, "scripts", pkg)
    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = lambda repo, module_map=None: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    calls: list[list[str]] = []

    def integration(mods: list[str]) -> None:
        calls.append(list(mods))

    svc = mod.SelfTestService(
        integration_callback=integration,
        clean_orphans=True,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == []
    assert calls == [["foo.py"]]  # path-ignore
    assert svc.results.get("orphan_passed") == ["foo.py"]  # path-ignore


def test_orphan_cleanup_passed(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()
    (tmp_path / "foo.py").write_text("x = 1\n")  # path-ignore
    (tmp_path / "sandbox_data" / "orphan_modules.json").write_text(
        json.dumps(["foo.py"])  # path-ignore
    )
    monkeypatch.chdir(tmp_path)

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 1, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    import types

    runner = types.ModuleType("sandbox_runner")
    runner.discover_recursive_orphans = lambda repo, module_map=None: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner", runner)

    svc = mod.SelfTestService(
        integration_callback=lambda mods: None,
        clean_orphans=True,
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert data == []
    assert svc.results.get("orphan_passed") == ["foo.py"]  # path-ignore


def test_default_integration_reports(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (tmp_path / "foo.py").write_text("VALUE = 1\n")  # path-ignore

    mod.analyze_redundancy = lambda p: False

    calls: dict[str, object] = {}

    class DummyIndex:
        def __init__(self, path):
            self.path = path

        def refresh(self, modules, force=False):
            calls["refresh"] = list(modules)

        def save(self):
            calls["save"] = True

    idx_mod = types.ModuleType("module_index_db")
    idx_mod.ModuleIndexDB = DummyIndex
    monkeypatch.setitem(sys.modules, "module_index_db", idx_mod)

    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.auto_include_modules = lambda mods, **k: calls.setdefault("auto", list(mods))
    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.environment = env_mod
    sr_pkg.discover_recursive_orphans = lambda repo, module_map=None: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_pkg)

    iso_mod = types.ModuleType("scripts.discover_isolated_modules")
    iso_mod.discover_isolated_modules = lambda root, *, recursive=True: ["foo.py"]  # path-ignore
    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.discover_isolated_modules = iso_mod
    monkeypatch.setitem(sys.modules, "scripts", scripts_pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", iso_mod)

    async def fake_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            Path(path).write_text(json.dumps({"summary": {"passed": 1, "failed": 0}}))

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    svc = mod.SelfTestService(context_builder=DummyBuilder())
    svc.logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)
    svc.run_once()

    assert svc.results.get("integration") == {"integrated": ["foo.py"], "redundant": []}  # path-ignore

