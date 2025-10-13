"""Compatibility helpers for importing menace_sandbox modules in any layout."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

PACKAGE_NAME = "menace_sandbox"

_MODULE_CACHE: dict[str, ModuleType] = {}


def _register_self_aliases() -> None:
    module = sys.modules[__name__]
    sys.modules.setdefault("import_compat", module)
    sys.modules.setdefault(f"{PACKAGE_NAME}.import_compat", module)


def _discover_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return start


def _ensure_package(repo_root: Path) -> ModuleType:
    module = sys.modules.get(PACKAGE_NAME)
    if module is None:
        init_path = repo_root / "__init__.py"
        spec = None
        if init_path.exists():
            spec = importlib.util.spec_from_file_location(
                PACKAGE_NAME,
                init_path,
                submodule_search_locations=[str(repo_root)],
            )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[PACKAGE_NAME] = module
            spec.loader.exec_module(module)
        else:
            module = ModuleType(PACKAGE_NAME)
            module.__file__ = str(init_path)
            module.__path__ = [str(repo_root)]
            sys.modules[PACKAGE_NAME] = module
    repo_root_str = str(repo_root)
    pkg_path = getattr(module, "__path__", None)
    if pkg_path is None:
        module.__path__ = [repo_root_str]
    else:
        try:
            existing = list(pkg_path)
        except TypeError:
            module.__path__ = [repo_root_str]
        else:
            if repo_root_str not in existing:
                try:
                    pkg_path.insert(0, repo_root_str)
                except Exception:
                    module.__path__ = [repo_root_str, *existing]
    if getattr(module, "__file__", None) is None:
        init_path = repo_root / "__init__.py"
        if init_path.exists():
            module.__file__ = str(init_path)
    return module


def bootstrap(module_name: str, module_file: str | Path | None = None) -> ModuleType:
    module = sys.modules.get(module_name)
    if module is None:
        module = ModuleType(module_name)
        if module_file is not None:
            module.__file__ = str(Path(module_file))
        sys.modules[module_name] = module

    module_path: Optional[Path]
    if module_file is not None:
        module_path = Path(module_file).resolve()
    else:
        mod_file = getattr(module, "__file__", None)
        module_path = Path(mod_file).resolve() if mod_file else None
    if module_path is None:
        module_path = Path.cwd()

    repo_root = _discover_repo_root(module_path.parent)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    _ensure_package(repo_root)

    if module_name == "__main__":
        flat_name = module_path.stem
        qualified_name = f"{PACKAGE_NAME}.{flat_name}"
    elif module_name.startswith(f"{PACKAGE_NAME}."):
        qualified_name = module_name
        flat_name = module_name[len(PACKAGE_NAME) + 1 :]
    elif module_name == PACKAGE_NAME:
        qualified_name = module_name
        flat_name = module_name
    else:
        flat_name = module_name
        qualified_name = f"{PACKAGE_NAME}.{flat_name}"

    module = sys.modules.get(module_name, module)

    # Normalise metadata so reloaders can treat the module as the packaged name
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    origin = getattr(spec, "origin", None)
    submodule_locations = getattr(spec, "submodule_search_locations", None)
    if spec is None or spec.name != qualified_name:
        try:
            new_spec = importlib.util.spec_from_loader(
                qualified_name,
                loader,
                origin=origin,
            )
        except Exception:
            new_spec = None
        if new_spec is not None and submodule_locations is not None:
            new_spec.submodule_search_locations = list(submodule_locations)
        if new_spec is not None:
            spec = new_spec
    if spec is not None:
        module.__spec__ = spec
        if getattr(spec, "loader", None) is not None:
            module.__loader__ = spec.loader  # type: ignore[attr-defined]

    if module.__name__ != qualified_name:
        module.__name__ = qualified_name
        # Retain the original entry so flat imports continue to succeed
        sys.modules[module_name] = module

    sys.modules[qualified_name] = module
    sys.modules[flat_name] = module

    if qualified_name.count("."):
        parent_package = qualified_name.rsplit(".", 1)[0]
    else:
        parent_package = PACKAGE_NAME
    if not getattr(module, "__package__", None):
        module.__package__ = parent_package

    return module


def load_internal(name: str) -> ModuleType:
    cached = _MODULE_CACHE.get(name)
    if cached is not None:
        return cached

    qualified = f"{PACKAGE_NAME}.{name}"
    try:
        module = importlib.import_module(qualified)
    except ModuleNotFoundError as primary_exc:
        if getattr(primary_exc, "name", None) not in {qualified, PACKAGE_NAME}:
            raise
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as secondary_exc:
            raise primary_exc from secondary_exc
    sys.modules[name] = module
    sys.modules[qualified] = module
    _MODULE_CACHE[name] = module
    return module


_register_self_aliases()

__all__ = ["PACKAGE_NAME", "bootstrap", "load_internal"]
