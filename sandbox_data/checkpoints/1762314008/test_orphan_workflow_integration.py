import json
import sys
import types
from pathlib import Path

from tests.test_recursive_orphans import _load_methods


class DummyLogger:
    def exception(self, *a, **k):
        pass
    def info(self, *a, **k):
        pass


def test_orphan_module_appends_to_existing_workflow(tmp_path, monkeypatch):
    _integrate_orphans, _update_orphan_modules, _refresh_module_map, _test_orphan_modules = _load_methods()

    # setup engine with dummy index
    class DummyIndex:
        def __init__(self, path: Path) -> None:
            self.path = path
            self._map = {"existing.py": 1}  # path-ignore
            self._groups = {"1": 1}
        def refresh(self, modules=None, force=False):
            for m in modules or []:
                self._map[Path(m).name] = 1
                self._groups.setdefault("1", 1)
        def get(self, name):
            return self._map.get(name, 1)
        def save(self):
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"modules": self._map, "groups": self._groups}, fh)

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {"existing.py": 1}, "groups": {"1": 1}}))  # path-ignore

    index = DummyIndex(map_path)
    engine = types.SimpleNamespace(
        module_index=index,
        module_clusters={"existing.py": 1},  # path-ignore
        logger=DummyLogger(),
    )
    engine._collect_recursive_modules = lambda mods: set(mods)
    engine._integrate_orphans = types.MethodType(_integrate_orphans, engine)
    engine._test_orphan_modules = types.MethodType(_test_orphan_modules, engine)
    engine._refresh_module_map = types.MethodType(_refresh_module_map, engine)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    sr = types.ModuleType("sandbox_runner")
    sr.run_repo_section_simulations = (
        lambda repo_path, modules=None, return_details=False, **k: (
            (
                types.SimpleNamespace(
                    module_deltas={m: [1.0] for m in (modules or [])},
                    metrics_history={"synergy_roi": [0.0]},
                ),
                {m: {"sec": [{"result": {"exit_code": 0}}]} for m in modules or []},
            )
            if return_details
            else types.SimpleNamespace(
                module_deltas={m: [1.0] for m in (modules or [])},
                metrics_history={"synergy_roi": [0.0]},
            )
        )
    )
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr)

    # create modules
    (tmp_path / "existing.py").write_text("x = 1\n")  # path-ignore
    (tmp_path / "orphan.py").write_text("y = 2\n")  # path-ignore

    # simple workflow store
    workflows = {1: ["existing"]}
    next_id = 2

    def fake_generate(mods, workflows_db="workflows.db"):
        nonlocal next_id
        ids = []
        for m in mods:
            workflows[next_id] = [Path(m).with_suffix("").as_posix().replace("/", ".")]
            ids.append(next_id)
            next_id += 1
        return ids

    def fake_try(mods):
        updated = []
        for wid, steps in workflows.items():
            step_groups = {index.get(f"{s}.py") for s in steps}  # path-ignore
            for m in mods:
                if index.get(Path(m).name) in step_groups:
                    name = Path(m).with_suffix("").as_posix().replace("/", ".")
                    if name not in steps:
                        steps.append(name)
                        updated.append(wid)
        return updated

    def fake_run():
        pass

    g = _integrate_orphans.__globals__
    def fake_auto(mods):
        fake_generate(mods)
        fake_try(mods)
        fake_run()
    g["auto_include_modules"] = fake_auto
    g["analyze_redundancy"] = lambda p: False

    engine._refresh_module_map(["orphan.py"])  # path-ignore

    # orphan module appended to existing workflow
    assert workflows[1] == ["existing", "orphan"]
    data = json.loads(map_path.read_text())
    assert "orphan.py" in data["modules"]  # path-ignore
