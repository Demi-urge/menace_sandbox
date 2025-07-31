import json
import os
from pathlib import Path
from typing import Dict

try:  # optional dependency only needed when auto mapping
    from scripts.generate_module_map import generate_module_map
except Exception:  # pragma: no cover - during tests optional dep may be missing
    generate_module_map = None  # type: ignore


class ModuleIndexDB:
    """Persist mapping of module names to numeric indices."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        auto_map: bool | None = None,
    ) -> None:
        default_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
        default_path = default_dir / "module_map.json"
        self.path = Path(path) if path is not None else default_path
        if not self.path.exists() and self.path != default_path and default_path.exists():
            self.path = default_path
        self._map: Dict[str, int] = {}
        self._groups: Dict[str, int] = {}

        auto_env = os.getenv("SANDBOX_AUTODISCOVER_MODULES") == "1"
        legacy_env = os.getenv("SANDBOX_AUTO_MAP") == "1"
        if (auto_map or (auto_map is None and (auto_env or legacy_env))) and generate_module_map:
            try:
                algo = os.getenv("SANDBOX_MODULE_ALGO", "greedy")
                try:
                    threshold = float(os.getenv("SANDBOX_MODULE_THRESHOLD", "0.1"))
                except Exception:
                    threshold = 0.1
                use_semantic = os.getenv("SANDBOX_MODULE_SEMANTIC") == "1"
                repo_path = Path(os.getenv("SANDBOX_REPO_PATH", "."))
                mapping = generate_module_map(
                    self.path,
                    root=repo_path,
                    algorithm=algo,
                    threshold=threshold,
                    semantic=use_semantic,
                )
                if self.path != repo_path / "sandbox_data" / "module_map.json":
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "w", encoding="utf-8") as fh:
                        json.dump(mapping, fh, indent=2)
            except Exception:
                pass

        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    if "modules" in data or "groups" in data:
                        mods = data.get("modules", {})
                        grps = data.get("groups", {})
                        if isinstance(mods, dict):
                            self._map = {str(k): int(v) for k, v in mods.items()}
                        if isinstance(grps, dict):
                            self._groups = {str(k): int(v) for k, v in grps.items()}
                    elif all(isinstance(v, int) for v in data.values()):
                        # ``build_module_map`` writes module -> int mappings
                        self._map = {k: int(v) for k, v in data.items()}
                    elif all(isinstance(v, list) for v in data.values()):
                        # ``module_mapper.save_module_map`` stores group -> [modules]
                        for grp, modules in data.items():
                            try:
                                idx = int(grp)
                            except Exception:
                                idx = abs(hash(grp)) % 1000
                            self._groups[str(grp)] = idx
                            for mod in modules:
                                self._map[str(mod)] = idx
                    else:
                        # Fallback for module -> group mappings with string groups
                        grp_idx: Dict[str, int] = {}
                        for mod, grp in data.items():
                            key = str(grp)
                            idx = grp_idx.setdefault(key, abs(hash(key)) % 1000)
                            self._groups.setdefault(key, idx)
                            self._map[mod] = idx
            except Exception:
                self._map = {}
                self._groups = {}

    # --------------------------------------------------------------
    def get(self, name: str) -> int:
        """Return persistent index for ``name`` creating it if needed."""
        parts = [name]
        base = Path(name).with_suffix("").as_posix()
        stem = Path(name).name
        stem_no_ext = Path(stem).with_suffix("").as_posix()
        parts.extend({base, stem, stem_no_ext})
        for key in parts:
            if key in self._map:
                return int(self._map[key])

        idx = abs(hash(name)) % 1000
        self._map[name] = idx
        self.save()
        return idx

    # --------------------------------------------------------------
    def group_id(self, group: str) -> int:
        """Return persistent index for group ``group`` creating it if needed."""
        if group in self._groups:
            return int(self._groups[group])
        next_idx = max([*self._groups.values(), *self._map.values()], default=-1) + 1
        self._groups[group] = next_idx
        self.save()
        return next_idx

    # --------------------------------------------------------------
    def merge_groups(self, groups: Dict[str, int]) -> None:
        """Merge ``groups`` mapping into this DB keeping indices stable."""
        group_to_idx: Dict[int, int] = {}
        for mod, idx in self._map.items():
            grp = groups.get(mod)
            if grp is not None:
                group_to_idx.setdefault(grp, idx)
                self._groups.setdefault(str(grp), idx)

        next_idx = max(self._map.values(), default=0) + 1
        for mod, grp in groups.items():
            if mod in self._map:
                self._groups.setdefault(str(grp), self._map[mod])
                continue
            if grp in group_to_idx:
                self._map[mod] = group_to_idx[grp]
            else:
                self._map[mod] = next_idx
                group_to_idx[grp] = next_idx
                self._groups.setdefault(str(grp), next_idx)
                next_idx += 1
        self.save()

    # --------------------------------------------------------------
    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"modules": self._map, "groups": self._groups}, fh, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass


__all__ = ["ModuleIndexDB"]
