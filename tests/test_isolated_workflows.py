import importlib.util
import json
import sys
import types
import asyncio
import subprocess
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def _load_thb():
    spec = importlib.util.spec_from_file_location(
        "menace.task_handoff_bot",
        ROOT / "task_handoff_bot.py",  # path-ignore
        submodule_search_locations=[str(ROOT)],
    )
    thb = importlib.util.module_from_spec(spec)
    sys.modules["menace.task_handoff_bot"] = thb
    assert spec and spec.loader
    spec.loader.exec_module(thb)  # type: ignore[attr-defined]
    return thb


from tests.test_recursive_orphans import _load_methods

_integrate_orphans, _update_orphan_modules, _refresh_module_map, _test_orphan_modules = _load_methods()


class DummyIndex:
    def __init__(self, path: Path | None = None) -> None:
        self._map: dict[str, int] = {}
        self._groups: dict[str, int] = {}
        self.path = path

    def refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[Path(m).name] = 1
            self._groups[str(1)] = 1
        self.save()

    def get(self, name):
        return 1

    def save(self):
        if self.path:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"modules": self._map, "groups": self._groups}, fh)


class DummyLogger:
    def exception(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


class SimpleEngine:
    def __init__(self, index: DummyIndex, wf_path: Path) -> None:
        self.module_index = index
        self.module_clusters: dict[str, int] = {}
        self.logger = DummyLogger()
        self.wf_path = wf_path

    def refresh_module_map(self, modules: list[str]) -> None:
        for m in modules:
            self.module_index.refresh([m], force=True)
            self.module_clusters[m] = 1
        self.module_index.save()
        thb = _load_thb()
        wf_db = thb.WorkflowDB(self.wf_path)
        for m in modules:
            dotted = Path(m).with_suffix("").as_posix().replace("/", ".")
            wf_db.add(thb.WorkflowRecord(workflow=[dotted], title=dotted))


def test_isolated_modules_written_to_workflows(tmp_path, monkeypatch):
    thb = _load_thb()

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "iso.py").write_text("import dep\n")  # path-ignore
    (repo / "dep.py").write_text("x = 1\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    import types as _types

    def discover(path, *, recursive=True):
        assert Path(path) == repo
        return ["iso.py", "dep.py"]  # path-ignore

    mod = _types.ModuleType("scripts.discover_isolated_modules")
    mod.discover_isolated_modules = discover
    pkg = _types.ModuleType("scripts")
    pkg.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)
    monkeypatch.setitem(sys.modules, "scripts", pkg)

    async def fake_exec(*cmd, **kwargs):
        path_file = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path_file = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path_file:
            with open(path_file, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 1, "failed": 0}}, fh)

        class P:
            returncode = 0

            async def communicate(self):
                return b"", b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    spec = importlib.util.spec_from_file_location(
        "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
    )
    if "menace.self_test_service" in sys.modules:
        sts = sys.modules["menace.self_test_service"]
    else:
        sts = importlib.util.module_from_spec(spec)
        sys.modules["menace.self_test_service"] = sts
        spec.loader.exec_module(sts)  # type: ignore[attr-defined]

    wf_db_path = tmp_path / "workflows.db"
    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    index = DummyIndex(map_path)
    eng = SimpleEngine(index, wf_db_path)

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *a, **k):
            return "", "", {}

    svc = sts.SelfTestService(
        discover_isolated=True,
        recursive_isolated=True,
        integration_callback=eng.refresh_module_map,
        context_builder=DummyBuilder(),
    )
    svc.run_once()
    passed = svc.results.get("orphan_passed") or []
    assert sorted(passed) == ["dep.py", "iso.py"]  # path-ignore

    wf_db = thb.WorkflowDB(wf_db_path)
    recs = wf_db.fetch(limit=10)
    names = {step for wf in recs for step in wf.workflow}
    assert "iso" in names
    assert "dep" in names
    data = json.loads(map_path.read_text())
    assert "iso.py" in data.get("modules", {})  # path-ignore
    assert "dep.py" in data.get("modules", {})  # path-ignore
