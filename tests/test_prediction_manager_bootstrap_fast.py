"""Bootstrap-fast regression tests for :mod:`prediction_manager_bot`."""

import importlib
import importlib.util
from pathlib import Path
import sys
import types

package = types.ModuleType("menace_sandbox")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
package.__spec__ = importlib.util.spec_from_loader(
    "menace_sandbox", loader=None, is_package=True
)
sys.modules.setdefault("menace_sandbox", package)

import menace_sandbox.prediction_manager_bot as pmb


class _FailingDataBot:
    def __getattr__(self, name: str):  # pragma: no cover - defensive
        raise AssertionError(f"DataBot should not be accessed during bootstrap_fast (got {name})")


class _StubDataBot:
    def __init__(self) -> None:
        self.db = types.SimpleNamespace(fetch=lambda _limit=20: [])


def test_bootstrap_fast_skips_default_registration(tmp_path, monkeypatch):
    """Fast bootstrap avoids spinning up default metric bots."""

    monkeypatch.setattr(pmb, "_import_future_prediction_bots", lambda: (_ for _ in ()).throw(AssertionError("futures")))
    manager = pmb.PredictionManager(
        tmp_path / "reg.json", data_bot=_FailingDataBot(), bootstrap_fast=True
    )
    assert manager.registry == {}
    assert manager.bootstrap_fast is True


def test_standard_bootstrap_populates_registry(tmp_path):
    """Normal initialisation still hydrates the default metric bots."""

    manager = pmb.PredictionManager(
        tmp_path / "reg.json", data_bot=_StubDataBot(), bootstrap_fast=False
    )
    registered_metrics = {metric for entry in manager.registry.values() for metric in entry.profile.get("metric", [])}
    assert registered_metrics, "default bots should register metrics when not bootstrapping fast"

def test_prediction_manager_reuses_placeholder_when_broker_inactive(monkeypatch):
    module = importlib.reload(importlib.import_module("prediction_manager_bot"))

    placeholder_pipeline = object()
    placeholder_manager = object()
    broker = types.SimpleNamespace(
        active_owner=False,
        active_pipeline=placeholder_pipeline,
        active_sentinel=placeholder_manager,
    )

    monkeypatch.setattr(
        module,
        "resolve_bootstrap_placeholders",
        lambda **_: (placeholder_pipeline, placeholder_manager, broker),
    )

    pipeline, manager, resolved_broker = module._bootstrap_placeholders()

    assert pipeline is placeholder_pipeline
    assert manager is placeholder_manager
    assert resolved_broker is broker


def test_prediction_manager_registry_uses_active_placeholder(monkeypatch):
    module = importlib.reload(importlib.import_module("prediction_manager_bot"))

    placeholder_pipeline = object()
    placeholder_manager = object()
    broker = types.SimpleNamespace(
        active_owner=False,
        active_pipeline=placeholder_pipeline,
        active_sentinel=placeholder_manager,
    )

    module._BOOTSTRAP_PLACEHOLDER_PIPELINE = None
    module._BOOTSTRAP_PLACEHOLDER_MANAGER = None
    module._BOOTSTRAP_BROKER = None
    module._get_registry.cache_clear()
    module._get_data_bot.cache_clear()

    module.get_active_bootstrap_pipeline = lambda: (
        placeholder_pipeline,
        placeholder_manager,
    )
    module._bootstrap_dependency_broker = lambda: broker
    module.resolve_bootstrap_placeholders = lambda **_: (_ for _ in ()).throw(
        AssertionError("resolve_bootstrap_placeholders should not be called")
    )
    advertise_calls: list[tuple[object | None, object | None]] = []

    def _advertise(**kwargs):
        advertise_calls.append((kwargs.get("pipeline"), kwargs.get("manager")))
        broker.active_pipeline = kwargs.get("pipeline")
        broker.active_sentinel = kwargs.get("manager")
        return kwargs.get("pipeline"), kwargs.get("manager")

    module.advertise_bootstrap_placeholder = _advertise
    module.ensure_bootstrapped = lambda: (_ for _ in ()).throw(
        AssertionError("ensure_bootstrapped should be skipped when bootstrap active")
    )
    module.bootstrap_state_snapshot = lambda: {"ready": False, "in_progress": True}

    monkeypatch.setattr(module._REGISTRY_PROXY, "_hydrate", lambda: "registry")
    monkeypatch.setattr(module._DATA_BOT_PROXY, "_hydrate", lambda: "data_bot")

    registry = module._get_registry()
    data_bot = module._get_data_bot()

    assert registry == "registry"
    assert data_bot == "data_bot"
    assert broker.active_pipeline is placeholder_pipeline
    assert broker.active_sentinel is placeholder_manager
    assert advertise_calls == [(placeholder_pipeline, placeholder_manager)]
