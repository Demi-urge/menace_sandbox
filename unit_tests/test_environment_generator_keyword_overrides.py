import importlib
import importlib.util
import pathlib
import sys

import pytest


def test_load_keyword_overrides_repo_moved(tmp_path, monkeypatch):
    config = "keyword_profiles:\n  sample: [alpha, beta]\n"
    moved_root = tmp_path / "moved_repo"
    moved_root.mkdir()
    (moved_root / "sandbox_settings.yaml").write_text(config, encoding="utf-8")

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(moved_root))
    monkeypatch.setenv("SANDBOX_SETTINGS_YAML", "sandbox_settings.yaml")

    sys.modules.pop("dynamic_path_router", None)
    spec = importlib.util.spec_from_file_location(
        "dynamic_path_router",
        pathlib.Path(__file__).resolve().parents[1] / "dynamic_path_router.py",
    )
    dynamic_path_router = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dynamic_path_router)
    dynamic_path_router._PROJECT_ROOT = None  # type: ignore[attr-defined]
    dynamic_path_router._PATH_CACHE.clear()  # type: ignore[attr-defined]

    sys.modules.pop("environment_generator", None)
    env_gen = importlib.import_module("environment_generator")

    overrides = env_gen._load_keyword_overrides()
    assert overrides == {"sample": ["alpha", "beta"]}
