import json
import os
from pathlib import Path
from typing import Dict

try:  # optional dependency only needed when auto mapping
    from scripts.generate_module_map import generate_module_map
except Exception:  # pragma: no cover - during tests script may not exist
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

        if (
            auto_map or (auto_map is None and os.getenv("SANDBOX_AUTO_MAP") == "1")
        ) and generate_module_map:
            try:
                generate_module_map(self.path)
            except Exception:
                pass

        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    if all(isinstance(v, int) for v in data.values()):
                        # ``build_module_map`` writes module -> int mappings
                        self._map = {k: int(v) for k, v in data.items()}
                    elif all(isinstance(v, list) for v in data.values()):
                        # ``module_mapper.save_module_map`` stores group -> [modules]
                        for grp, modules in data.items():
                            try:
                                idx = int(grp)
                            except Exception:
                                idx = abs(hash(grp)) % 1000
                            for mod in modules:
                                self._map[str(mod)] = idx
                    else:
                        # Fallback for module -> group mappings with string groups
                        grp_idx: Dict[str, int] = {}
                        for mod, grp in data.items():
                            key = str(grp)
                            idx = grp_idx.setdefault(key, abs(hash(key)) % 1000)
                            self._map[mod] = idx
            except Exception:
                self._map = {}

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
    def merge_groups(self, groups: Dict[str, int]) -> None:
        """Merge ``groups`` mapping into this DB keeping indices stable."""
        group_to_idx: Dict[int, int] = {}
        for mod, idx in self._map.items():
            grp = groups.get(mod)
            if grp is not None:
                group_to_idx.setdefault(grp, idx)

        next_idx = max(self._map.values(), default=0) + 1
        for mod, grp in groups.items():
            if mod in self._map:
                continue
            if grp in group_to_idx:
                self._map[mod] = group_to_idx[grp]
            else:
                self._map[mod] = next_idx
                group_to_idx[grp] = next_idx
                next_idx += 1
        self.save()

    # --------------------------------------------------------------
    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._map, fh, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass


__all__ = ["ModuleIndexDB"]
