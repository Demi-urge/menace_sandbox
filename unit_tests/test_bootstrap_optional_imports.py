"""Regression coverage for optional module bootstrap helpers."""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
import types
from collections.abc import Iterable
from pathlib import Path

import pytest


def _setup_bootstrap_stubs() -> None:
    """Provide lightweight stubs so ``sandbox_runner.bootstrap`` can import."""

    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

    menace_pkg = sys.modules.get("menace")
    if menace_pkg is None:
        menace_pkg = types.ModuleType("menace")
        menace_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["menace"] = menace_pkg

    sys.modules.setdefault(
        "menace.auto_env_setup",
        types.SimpleNamespace(ensure_env=lambda *a, **k: None),
    )
    sys.modules.setdefault(
        "menace.default_config_manager",
        types.SimpleNamespace(
            DefaultConfigManager=lambda *a, **k: types.SimpleNamespace(
                apply_defaults=lambda: None
            )
        ),
    )
    sys.modules.setdefault(
        "sandbox_runner.cli", types.SimpleNamespace(main=lambda *a, **k: None)
    )
    sys.modules.setdefault(
        "sandbox_runner.cycle",
        types.SimpleNamespace(ensure_vector_service=lambda: None),
    )

    dpr = sys.modules.setdefault(
        "dynamic_path_router",
        types.SimpleNamespace(
            resolve_path=lambda p: Path(p),
            resolve_dir=lambda p: Path(p),
            path_for_prompt=lambda p: Path(p).as_posix(),
        ),
    )
    if not hasattr(dpr, "resolve_path"):
        dpr.resolve_path = lambda p: Path(p)
    if not hasattr(dpr, "resolve_dir"):
        dpr.resolve_dir = lambda p: Path(p)
    if not hasattr(dpr, "path_for_prompt"):
        dpr.path_for_prompt = lambda p: Path(p).as_posix()
    if not hasattr(dpr, "repo_root"):
        dpr.repo_root = lambda: Path(".")


_setup_bootstrap_stubs()
bootstrap = importlib.import_module("sandbox_runner.bootstrap")


def _restore_modules(names: Iterable[str]) -> None:
    """Remove transient modules from ``sys.modules``."""

    for name in names:
        sys.modules.pop(name, None)


def test_candidate_names_prioritise_repo_package() -> None:
    """Package-qualified candidates should precede the bare name."""

    name = "example_optional_module"
    candidates = bootstrap._candidate_optional_module_names(name)

    assert candidates[0] == f"{bootstrap._REPO_PACKAGE}.{name}"
    assert candidates[-1] == name
    assert name in candidates


def test_optional_import_cleans_partial_modules(tmp_path: Path) -> None:
    """Relative import failures are cleaned before retrying with package context."""

    module_name = "_bootstrap_optional_retry"
    helper_name = f"{module_name}_helper"

    (tmp_path / f"{helper_name}.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (tmp_path / f"{module_name}.py").write_text(
        textwrap.dedent(
            f"""
            if __package__ is None:
                PARTIAL_STAMP = 'bare'
            from . import {helper_name} as helper
            VALUE = helper.VALUE
            """
        ),
        encoding="utf-8",
    )

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))

    original_menace_sandbox = sys.modules.get("menace_sandbox")
    menace_stub = types.ModuleType("menace_sandbox")
    menace_stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules["menace_sandbox"] = menace_stub

    cache_snapshot = dict(bootstrap._OPTIONAL_MODULE_CACHE)
    for key in (
        module_name,
        f"{bootstrap._REPO_PACKAGE}.{module_name}",
        f"sandbox_runner.{module_name}",
    ):
        bootstrap._OPTIONAL_MODULE_CACHE.pop(key, None)

    _restore_modules(
        (
            module_name,
            helper_name,
            f"{bootstrap._REPO_PACKAGE}.{module_name}",
            f"{bootstrap._REPO_PACKAGE}.{helper_name}",
            f"sandbox_runner.{module_name}",
        )
    )

    try:
        importlib.invalidate_caches()

        with pytest.raises(ImportError) as excinfo:
            bootstrap._import_optional_module(module_name)

        assert "relative import with no known parent package" in str(excinfo.value).lower()
        assert module_name not in sys.modules

        menace_stub.__path__.append(str(tmp_path))
        importlib.invalidate_caches()

        module = bootstrap._import_optional_module(module_name)

        assert module.VALUE == "ok"
        assert module.__name__ == f"{bootstrap._REPO_PACKAGE}.{module_name}"
        assert sys.modules[module_name] is module
        assert not hasattr(module, "PARTIAL_STAMP")
        assert bootstrap._OPTIONAL_MODULE_CACHE[module_name] is module
        assert (
            bootstrap._OPTIONAL_MODULE_CACHE[f"{bootstrap._REPO_PACKAGE}.{module_name}"]
            is module
        )
    finally:
        sys.path[:] = original_sys_path
        if original_menace_sandbox is None:
            sys.modules.pop("menace_sandbox", None)
        else:
            sys.modules["menace_sandbox"] = original_menace_sandbox
        importlib.invalidate_caches()
        _restore_modules(
            (
                module_name,
                helper_name,
                f"{bootstrap._REPO_PACKAGE}.{module_name}",
                f"{bootstrap._REPO_PACKAGE}.{helper_name}",
                f"sandbox_runner.{module_name}",
            )
        )
        bootstrap._OPTIONAL_MODULE_CACHE.clear()
        bootstrap._OPTIONAL_MODULE_CACHE.update(cache_snapshot)


def test_optional_import_dependency_failure_resets_repo_modules(tmp_path: Path) -> None:
    """Dependency failures should not poison future package-qualified imports."""

    module_name = "embeddable_db_mixin"
    repo_package = bootstrap._REPO_PACKAGE

    bad_root = tmp_path / "bad"
    good_root = tmp_path / "good"
    bad_pkg = bad_root / repo_package
    good_pkg = good_root / repo_package

    bad_pkg.mkdir(parents=True)
    good_pkg.mkdir(parents=True)

    (bad_pkg / "__init__.py").write_text("", encoding="utf-8")
    (good_pkg / "__init__.py").write_text("", encoding="utf-8")

    import_compat_stub = textwrap.dedent(
        """
        import sys

        def bootstrap(name: str, filename: str) -> None:
            return None

        sys.modules.setdefault("import_compat", sys.modules[__name__])
        """
    )

    (bad_pkg / "import_compat.py").write_text(import_compat_stub, encoding="utf-8")
    (good_pkg / "import_compat.py").write_text(import_compat_stub, encoding="utf-8")

    (bad_pkg / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            from . import import_compat  # ensure alias entries are created
            raise ImportError("simulated dependency failure")
            """
        ),
        encoding="utf-8",
    )

    (bad_root / f"{module_name}.py").write_text(
        "raise ImportError('attempted relative import with no known parent package')\n",
        encoding="utf-8",
    )

    (good_pkg / "helper_module.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (good_pkg / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            from . import import_compat
            from . import helper_module as helper

            VALUE = helper.VALUE
            """
        ),
        encoding="utf-8",
    )

    (good_root / f"{module_name}.py").write_text(
        "from . import missing_helper  # pragma: no cover - fallback guard\n",
        encoding="utf-8",
    )

    tracked_modules = [
        module_name,
        f"{repo_package}.{module_name}",
        f"sandbox_runner.{module_name}",
        f"{repo_package}.import_compat",
        f"{repo_package}.helper_module",
        "import_compat",
        repo_package,
    ]

    original_sys_path = list(sys.path)
    original_modules = {name: sys.modules.get(name) for name in tracked_modules}
    cache_snapshot = dict(bootstrap._OPTIONAL_MODULE_CACHE)
    missing_snapshot = set(bootstrap._MISSING_OPTIONAL)
    warned_snapshot = set(bootstrap._OPTIONAL_DEPENDENCY_WARNED)

    cache_keys = (
        module_name,
        f"{repo_package}.{module_name}",
        f"sandbox_runner.{module_name}",
    )

    try:
        for name in tracked_modules:
            sys.modules.pop(name, None)
        for key in cache_keys:
            bootstrap._OPTIONAL_MODULE_CACHE.pop(key, None)
        bootstrap._MISSING_OPTIONAL.discard(module_name)
        bootstrap._OPTIONAL_DEPENDENCY_WARNED.discard(module_name)

        sys.path.insert(0, str(bad_root))
        importlib.invalidate_caches()

        with pytest.raises(ModuleNotFoundError):
            bootstrap._import_optional_module(module_name)

        assert repo_package not in sys.modules
        assert f"{repo_package}.import_compat" not in sys.modules
        assert "import_compat" not in sys.modules

        sys.path.pop(0)
        (bad_pkg / f"{module_name}.py").unlink()
        sys.path.insert(0, str(good_root))
        importlib.invalidate_caches()

        for name in tracked_modules:
            sys.modules.pop(name, None)
        for key in cache_keys:
            bootstrap._OPTIONAL_MODULE_CACHE.pop(key, None)
        bootstrap._MISSING_OPTIONAL.discard(module_name)
        bootstrap._OPTIONAL_DEPENDENCY_WARNED.discard(module_name)

        try:
            module = bootstrap._import_optional_module(module_name)
        except ImportError as exc:  # pragma: no cover - regression guard
            pytest.fail(f"optional import should have recovered: {exc}")

        assert module.VALUE == "ok"
        assert module.__name__ == f"{repo_package}.{module_name}"
        assert str(good_pkg) in (module.__file__ or "")
        assert sys.modules[module_name] is module
    finally:
        sys.path[:] = original_sys_path
        importlib.invalidate_caches()

        for name in tracked_modules:
            sys.modules.pop(name, None)

        for name, module in original_modules.items():
            if module is not None:
                sys.modules[name] = module

        bootstrap._OPTIONAL_MODULE_CACHE.clear()
        bootstrap._OPTIONAL_MODULE_CACHE.update(cache_snapshot)
        bootstrap._MISSING_OPTIONAL.clear()
        bootstrap._MISSING_OPTIONAL.update(missing_snapshot)
        bootstrap._OPTIONAL_DEPENDENCY_WARNED.clear()
        bootstrap._OPTIONAL_DEPENDENCY_WARNED.update(warned_snapshot)
