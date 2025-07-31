import asyncio
import importlib.util
import json
import sys
import types
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

# load SelfTestService from source
spec = importlib.util.spec_from_file_location(
    "menace.self_test_service", ROOT / "self_test_service.py"
)
sts = importlib.util.module_from_spec(spec)
sys.modules["menace.self_test_service"] = sts
spec.loader.exec_module(sts)

# load ModuleIndexDB
spec = importlib.util.spec_from_file_location(
    "module_index_db", ROOT / "module_index_db.py"
)
mod_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_db)

class DummyLogger:
    def __init__(self) -> None:
        self.exc: list[str] = []

    def exception(self, msg: str, exc: Exception) -> None:
        self.exc.append(msg)

class DummyEngine:
    def __init__(self, index: mod_db.ModuleIndexDB) -> None:
        self.module_index = index
        self.module_clusters: dict[str, int] = {}
        self.logger = DummyLogger()
        self._last_map_refresh = 0.0
        self.auto_refresh_map = True
        self.meta_logger = None

    def _integrate_orphans(self, paths: Iterable[str]) -> None:
        if not self.module_index:
            return
        mods = {Path(p).name for p in paths}
        unknown = [m for m in mods if m not in self.module_clusters]
        if not unknown:
            return
        try:
            self.module_index.refresh(mods, force=True)
            grp_map = {m: self.module_index.get(m) for m in mods}
            self.module_clusters.update(grp_map)
            self.module_index.save()
            self._last_map_refresh = time.time()
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.exception("orphan integration failed: %s", exc)

    def _refresh_module_map(self, modules: list[str] | None = None) -> None:
        if modules:
            self._integrate_orphans(modules)
            return


def test_orphan_module_mapping(tmp_path, monkeypatch):
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "orphan_modules.json").write_text(json.dumps(["foo.py"]))
    (tmp_path / "foo.py").write_text("def foo():\n    return 1\n")

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
    monkeypatch.chdir(tmp_path)

    svc = sts.SelfTestService(include_orphans=True)
    asyncio.run(svc._run_once())

    orphans = json.loads((data_dir / "orphan_modules.json").read_text())
    passed = [m for m in orphans if m not in svc.results.get("orphan_failed", [])]

    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))
    index = mod_db.ModuleIndexDB(map_path)
    engine = DummyEngine(index)

    def fake_refresh(self, modules=None, force=False):
        for m in modules or []:
            self._map[m] = 1
            self._groups[str(1)] = 1
        self.save()

    monkeypatch.setattr(mod_db.ModuleIndexDB, "refresh", fake_refresh)

    engine._refresh_module_map(passed)

    assert engine.module_clusters.get("foo.py") == 1
    data = json.loads(map_path.read_text())
    assert "foo.py" in data.get("modules", {})
