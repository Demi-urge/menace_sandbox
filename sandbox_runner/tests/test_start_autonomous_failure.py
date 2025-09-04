import sys
import types
import logging
from pathlib import Path
from dynamic_path_router import resolve_path

import pytest

sys.path.append(str(resolve_path("")))


def _setup_base_packages():
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["menace"] = menace_pkg
    sys.modules["menace.auto_env_setup"] = types.SimpleNamespace(ensure_env=lambda *a, **k: None)
    sys.modules["menace.default_config_manager"] = types.SimpleNamespace(
        DefaultConfigManager=lambda *a, **k: types.SimpleNamespace(apply_defaults=lambda: None)
    )
    sys.modules["menace.environment_generator"] = types.SimpleNamespace(
        _CPU_LIMITS={}, _MEMORY_LIMITS={}
    )
    sys.modules["sandbox_runner.cli"] = types.SimpleNamespace(main=lambda *a, **k: None)
    sys.modules["sandbox_runner.cycle"] = types.SimpleNamespace(
        ensure_vector_service=lambda: None
    )
    si_pkg = types.ModuleType("self_improvement")
    sys.modules["self_improvement"] = si_pkg


_setup_base_packages()


def test_failed_launch_surfaces_error(monkeypatch, caplog):
    import start_autonomous_sandbox as sas

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(sas, "launch_sandbox", boom)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            sas.main([])

    assert exc.value.code == 1
    assert any("Failed to launch sandbox" in record.message for record in caplog.records)
