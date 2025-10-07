from __future__ import annotations

import types

import self_coding_dependency_probe as probe


def test_probe_reports_missing(monkeypatch):
    checked = {}

    def fake_find_spec(name: str):
        checked[name] = checked.get(name, 0) + 1
        if name == "pydantic":
            return None
        return types.SimpleNamespace()

    monkeypatch.setattr(probe, "find_spec", fake_find_spec)
    probe._runtime_dependency_issues.cache_clear()

    ready, missing = probe.ensure_self_coding_ready()

    assert not ready
    assert "pydantic" in missing
    assert checked["pydantic"] == 1
    probe._runtime_dependency_issues.cache_clear()


def test_probe_all_present(monkeypatch):
    monkeypatch.setattr(
        probe,
        "find_spec",
        lambda name: types.SimpleNamespace(),
    )

    ready, missing = probe.ensure_self_coding_ready(["pydantic", "sklearn"])

    assert ready
    assert missing == ()
    probe._runtime_dependency_issues.cache_clear()


def test_runtime_probe_reports_nested_missing(monkeypatch):
    monkeypatch.setattr(
        probe,
        "find_spec",
        lambda name: types.SimpleNamespace(),
    )

    def _boom(name: str):  # pragma: no cover - exercised via probe
        raise ImportError(
            "cannot import name 'Engine' from 'menace.quick_fix_engine'"
            " (DLL load failed while importing helper_lib:"
            " The specified module could not be found.)"
        )

    monkeypatch.setattr(probe.importlib, "import_module", _boom)
    probe._runtime_dependency_issues.cache_clear()

    ready, missing = probe.ensure_self_coding_ready()

    assert not ready
    assert "helper_lib" in missing
    assert "quick_fix_engine" in missing
    probe._runtime_dependency_issues.cache_clear()


def test_runtime_probe_handles_generic_import_failure(monkeypatch):
    monkeypatch.setattr(
        probe,
        "find_spec",
        lambda name: types.SimpleNamespace(),
    )

    def _boom(name: str):  # pragma: no cover - exercised via probe
        raise ImportError("QuickFixEngine is required but could not be imported")

    monkeypatch.setattr(probe.importlib, "import_module", _boom)
    probe._runtime_dependency_issues.cache_clear()

    ready, missing = probe.ensure_self_coding_ready()

    assert not ready
    assert "quick_fix_engine" in missing
    probe._runtime_dependency_issues.cache_clear()


def test_runtime_probe_handles_windows_error_loading(monkeypatch):
    monkeypatch.setattr(
        probe,
        "find_spec",
        lambda name: types.SimpleNamespace(),
    )

    def _boom(name: str):  # pragma: no cover - exercised via probe
        raise ImportError('Error loading "C:/menace/quick_fix_engine.pyd" or one of its dependencies.')

    monkeypatch.setattr(probe.importlib, "import_module", _boom)
    probe._runtime_dependency_issues.cache_clear()

    ready, missing = probe.ensure_self_coding_ready()

    assert not ready
    assert any("quick_fix_engine" in item for item in missing)
    probe._runtime_dependency_issues.cache_clear()
