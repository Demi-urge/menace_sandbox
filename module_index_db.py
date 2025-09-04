import json
import os
from pathlib import Path
from typing import Dict, Iterable

try:  # pragma: no cover - prefer package import
    from .dynamic_path_router import resolve_path, get_project_root  # type: ignore
except Exception:  # pragma: no cover - allow running as script
    from dynamic_path_router import resolve_path, get_project_root  # type: ignore

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
        default_dir = Path(resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")))
        default_path = default_dir / "module_map.json"
        self.path = Path(path) if path is not None else default_path
        if not self.path.exists() and self.path != default_path and default_path.exists():
            self.path = default_path
        self._map: Dict[str, int] = {}
        self._groups: Dict[str, int] = {}
        self._tags: Dict[str, list[str]] = {}

        auto_env = os.getenv("SANDBOX_AUTO_MAP") == "1"
        legacy_env = os.getenv("SANDBOX_AUTODISCOVER_MODULES") == "1"
        if legacy_env and not auto_env:
            import warnings

            warnings.warn(
                "SANDBOX_AUTODISCOVER_MODULES is deprecated; use SANDBOX_AUTO_MAP",
                stacklevel=2,
            )

        if (auto_map or (auto_map is None and (auto_env or legacy_env))) and generate_module_map:
            try:
                algo = os.getenv("SANDBOX_MODULE_ALGO", "greedy")
                try:
                    threshold = float(os.getenv("SANDBOX_MODULE_THRESHOLD", "0.1"))
                except Exception:
                    threshold = 0.1
                sem_env = os.getenv("SANDBOX_SEMANTIC_MODULES")
                if sem_env is None:
                    sem_env = os.getenv("SANDBOX_MODULE_SEMANTIC")  # legacy
                use_semantic = sem_env == "1"
                exclude_env = os.getenv("SANDBOX_EXCLUDE_DIRS")
                exclude = [e for e in exclude_env.split(",") if e] if exclude_env else None
                mapping = generate_module_map(
                    self.path,
                    root=get_project_root(),
                    algorithm=algo,
                    threshold=threshold,
                    semantic=use_semantic,
                    exclude=exclude,
                )
                if self.path != resolve_path("sandbox_data/module_map.json"):
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
                            self._map = {
                                self._norm(str(k)): int(v) for k, v in mods.items()
                            }
                        if isinstance(grps, dict):
                            self._groups = {str(k): int(v) for k, v in grps.items()}
                        tag_data = data.get("tags", {})
                        if isinstance(tag_data, dict):
                            self._tags = {
                                self._norm(str(k)): [str(t) for t in v]
                                for k, v in tag_data.items()
                                if isinstance(v, list)
                            }
                    elif all(isinstance(v, int) for v in data.values()):
                        # ``build_module_map`` writes module -> int mappings
                        self._map = {
                            self._norm(k): int(v) for k, v in data.items()
                        }
                    elif all(isinstance(v, list) for v in data.values()):
                        # ``module_mapper.save_module_map`` stores group -> [modules]
                        for grp, modules in data.items():
                            try:
                                idx = int(grp)
                            except Exception:
                                idx = abs(hash(grp)) % 1000
                            self._groups[str(grp)] = idx
                            for mod in modules:
                                self._map[self._norm(str(mod))] = idx
                    else:
                        # Fallback for module -> group mappings with string groups
                        grp_idx: Dict[str, int] = {}
                        for mod, grp in data.items():
                            key = str(grp)
                            idx = grp_idx.setdefault(key, abs(hash(key)) % 1000)
                            self._groups.setdefault(key, idx)
                            norm = self._norm(mod)
                            self._map[norm] = idx
                            if "tags" in data:
                                tags = data.get("tags", {}).get(mod)
                                if isinstance(tags, list):
                                    self._tags[norm] = [str(t) for t in tags]
            except Exception:
                self._map = {}
                self._groups = {}
                self._tags = {}

    # --------------------------------------------------------------
    def _norm(self, name: str) -> str:
        """Return repository-relative POSIX path for ``name``."""
        repo_path = get_project_root()
        p = Path(name)
        try:
            p = p.resolve()
            return p.relative_to(repo_path).as_posix()
        except Exception:
            try:
                return p.as_posix()
            except Exception:
                return str(p)

    # --------------------------------------------------------------
    def get(self, name: str) -> int:
        """Return persistent index for ``name`` creating it if needed."""
        norm = self._norm(name)
        parts = [norm]
        base = Path(norm).with_suffix("").as_posix()
        parts.append(base)
        if "/" not in norm:
            stem = Path(norm).name
            stem_no_ext = Path(stem).with_suffix("").as_posix()
            parts.extend({stem, stem_no_ext})
        for key in parts:
            if key in self._map:
                return int(self._map[key])

        idx = abs(hash(norm)) % 1000
        self._map[norm] = idx
        self.save()
        return idx

    # --------------------------------------------------------------
    def get_tags(self, name: str) -> list[str]:
        """Return stored tags for ``name`` or an empty list."""
        return list(self._tags.get(self._norm(name), []))

    # --------------------------------------------------------------
    def set_tags(self, name: str, tags: Iterable[str]) -> None:
        """Persist ``tags`` for ``name``."""
        norm = self._norm(name)
        self._tags[norm] = sorted({str(t) for t in tags})
        self.save()

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
            grp = groups.get(mod) if mod in groups else groups.get(self._norm(mod))
            if grp is not None:
                group_to_idx.setdefault(grp, idx)
                self._groups.setdefault(str(grp), idx)

        next_idx = max(self._map.values(), default=0) + 1
        for mod, grp in groups.items():
            key = self._norm(mod)
            if key in self._map:
                self._groups.setdefault(str(grp), self._map[key])
                continue
            if grp in group_to_idx:
                self._map[key] = group_to_idx[grp]
            else:
                self._map[key] = next_idx
                group_to_idx[grp] = next_idx
                self._groups.setdefault(str(grp), next_idx)
                next_idx += 1
        self.save()

    # --------------------------------------------------------------
    def refresh(self, modules: Iterable[str] | None = None, *, force: bool = False) -> None:
        """Regenerate the module map if ``modules`` contain unknown entries."""
        if generate_module_map is None:
            return
        if not force:
            for m in modules or []:
                if self._norm(m) not in self._map:
                    force = True
                    break
        if not force:
            return
        try:
            algo = os.getenv("SANDBOX_MODULE_ALGO", "greedy")
            try:
                threshold = float(os.getenv("SANDBOX_MODULE_THRESHOLD", "0.1"))
            except Exception:
                threshold = 0.1
            sem_env = os.getenv("SANDBOX_SEMANTIC_MODULES")
            if sem_env is None:
                sem_env = os.getenv("SANDBOX_MODULE_SEMANTIC")
            use_semantic = sem_env == "1"
            repo_path = get_project_root()
            exclude_env = os.getenv("SANDBOX_EXCLUDE_DIRS")
            exclude = [e for e in exclude_env.split(",") if e] if exclude_env else None
            mapping = generate_module_map(
                self.path,
                root=repo_path,
                algorithm=algo,
                threshold=threshold,
                semantic=use_semantic,
                exclude=exclude,
            )
            self._map = {self._norm(str(k)): int(v) for k, v in mapping.items()}
            self._groups = {str(v): v for v in mapping.values()}
            self.save()
        except Exception:
            pass

    # --------------------------------------------------------------
    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "modules": self._map,
                        "groups": self._groups,
                        "tags": self._tags,
                    },
                    fh,
                    indent=2,
                )
            os.replace(tmp, self.path)
        except Exception:
            pass


__all__ = ["ModuleIndexDB"]
