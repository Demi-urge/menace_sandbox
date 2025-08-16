from __future__ import annotations

"""Service for archiving or compressing modules flagged for retirement."""

from pathlib import Path
from typing import Dict, Iterable
import logging
import shutil

from module_graph_analyzer import build_import_graph
from quick_fix_engine import generate_patch
from metrics_exporter import (
    update_module_retirement_metrics,
    retired_modules_total,
    compressed_modules_total,
    replaced_modules_total,
)


class ModuleRetirementService:
    """Handle archival or compression of modules based on relevancy flags."""

    def __init__(self, repo_root: Path | str = ".") -> None:
        self.root = Path(repo_root)
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self._graph = build_import_graph(self.root)
        except Exception:  # pragma: no cover - dependency graph failures
            self._graph = None
            self.logger.exception("failed to build import graph")

    # ------------------------------------------------------------------
    def _normalise(self, module: str) -> str:
        mod = module
        if mod.endswith(".py"):
            mod = mod[:-3]
        return mod.replace("\\", "/")

    def _dependents(self, module: str) -> Iterable[str]:
        if self._graph is None:
            return []
        mod = self._normalise(module)
        if mod in self._graph:
            return list(self._graph.predecessors(mod))
        return []

    # ------------------------------------------------------------------
    def retire_module(self, module: str) -> bool:
        """Archive ``module`` if no other modules depend on it."""

        path = self.root / (module if module.endswith(".py") else f"{module}.py")
        dependents = list(self._dependents(module))
        if dependents:
            self.logger.warning("cannot retire %s; dependents exist: %s", module, dependents)
            return False
        archive_dir = self.root / "sandbox_data" / "retired_modules"
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            if path.exists():
                shutil.move(str(path), archive_dir / path.name)
                retired_modules_total.inc()
                return True
            self.logger.error("module not found: %s", module)
        except Exception:  # pragma: no cover - filesystem issues
            self.logger.exception("failed to retire module %s", module)
        return False

    def compress_module(self, module: str) -> bool:
        """Invoke quick fix tooling to minimise ``module``."""

        path = self.root / (module if module.endswith(".py") else f"{module}.py")
        if not path.exists():
            self.logger.error("module not found: %s", module)
            return False
        try:
            patch_id = generate_patch(str(path))
            if patch_id is not None:
                compressed_modules_total.inc()
                return True
        except Exception:  # pragma: no cover - patching issues
            self.logger.exception("compression failed for %s", module)
        return False

    def replace_module(self, module: str) -> bool:
        """Invoke quick fix tooling to propose replacement for ``module``."""

        path = self.root / (module if module.endswith(".py") else f"{module}.py")
        if not path.exists():
            self.logger.error("module not found: %s", module)
            return False
        try:
            patch_id = generate_patch(str(path))
            if patch_id is not None:
                replaced_modules_total.inc()
                return True
        except Exception:  # pragma: no cover - patching issues
            self.logger.exception("replacement failed for %s", module)
        return False

    # ------------------------------------------------------------------
    def process_flags(self, flags: Dict[str, str]) -> Dict[str, str]:
        """Process relevancy ``flags`` mapping modules to actions."""

        results: Dict[str, str] = {}
        for mod, flag in flags.items():
            success = False
            if flag == "retire":
                success = self.retire_module(mod)
                if success:
                    results[mod] = "retired"
            elif flag == "compress":
                success = self.compress_module(mod)
                if success:
                    results[mod] = "compressed"
            elif flag == "replace":
                success = self.replace_module(mod)
                if success:
                    results[mod] = "replaced"
            if not success:
                results.setdefault(mod, "skipped")
        update_module_retirement_metrics(results)
        return results


__all__ = ["ModuleRetirementService"]

