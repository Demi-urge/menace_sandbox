import importlib.util
import sys
import types
from dynamic_path_router import resolve_path
from pathlib import Path


def path_for_prompt(p: str) -> str:
    return Path(p).name

PKG_DIR = resolve_path("self_improvement")


def load_module(
    module_name: str, file_name: str, deps: dict[str, types.ModuleType] | None = None
):
    if module_name.startswith("menace_sandbox."):
        root = sys.modules.setdefault(
            "menace_sandbox", types.ModuleType("menace_sandbox")
        )
        root.__path__ = [str(PKG_DIR.parent)]
        pkg = sys.modules.setdefault(
            "menace_sandbox.self_improvement",
            types.ModuleType("menace_sandbox.self_improvement"),
        )
        pkg.__path__ = [str(PKG_DIR)]
    else:
        pkg = sys.modules.setdefault(
            "self_improvement", types.ModuleType("self_improvement")
        )
        pkg.__path__ = [str(PKG_DIR)]
    if deps:
        for name, mod in deps.items():
            sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location(module_name, PKG_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_update_alignment_baseline_delegates():
    called = {}
    metrics_stub = types.ModuleType("self_improvement.metrics")

    def fake_update(settings):
        called['settings'] = settings
        return 'ok'

    metrics_stub._update_alignment_baseline = fake_update
    module = load_module(
        "self_improvement.roi_tracking",
        path_for_prompt("self_improvement/roi_tracking.py"),
        {"self_improvement.metrics": metrics_stub},
    )
    assert module.update_alignment_baseline('cfg') == 'ok'
    assert called['settings'] == 'cfg'


def test_generate_patch_delegates():
    record = {}
    patch_stub = types.ModuleType(
        "menace_sandbox.self_improvement.patch_generation"
    )

    def fake_generate(*args, **kwargs):
        record['args'] = args
        record['kwargs'] = kwargs
        return 'patch'

    patch_stub.generate_patch = fake_generate
    utils_stub = types.ModuleType("menace_sandbox.self_improvement.utils")
    utils_stub._load_callable = lambda *_: fake_generate
    utils_stub._call_with_retries = lambda func, *a, **k: func(*a, **k)

    ss_stub = types.ModuleType("menace_sandbox.sandbox_settings")
    ss_stub.SandboxSettings = lambda: types.SimpleNamespace(
        patch_retries=1, patch_retry_delay=0.1
    )

    module = load_module(
        "menace_sandbox.self_improvement.patch_application",
        path_for_prompt("self_improvement/patch_application.py"),
        {
            "menace_sandbox.self_improvement.patch_generation": patch_stub,
            "menace_sandbox.self_improvement.utils": utils_stub,
            "menace_sandbox.sandbox_settings": ss_stub,
        },
    )
    manager = object()
    assert module.generate_patch('a', manager, context_builder='b', key='v') == 'patch'
    assert record['args'] == ('a', manager)
    assert record['kwargs'] == {'key': 'v', 'context_builder': 'b'}
    sys.modules.pop("menace_sandbox.sandbox_settings", None)


def test_orphan_handlers_delegate():
    calls = {}
    utils_stub = types.ModuleType("self_improvement.utils")

    def call_with_retries(func, *args, **kwargs):
        return func(*args, **kwargs)

    utils_stub._call_with_retries = call_with_retries

    class DummySettings:
        orphan_retry_attempts = 5
        orphan_retry_delay = 0.7

    ss_stub = types.ModuleType("sandbox_settings")
    ss_stub.SandboxSettings = lambda: DummySettings()

    module = load_module(
        "self_improvement.orphan_handling",
        path_for_prompt("self_improvement/orphan_handling.py"),
        {
            "self_improvement.utils": utils_stub,
            "sandbox_settings": ss_stub,
        },
    )

    def loader(name):
        def _func(*args, **kwargs):
            calls[name] = (args, kwargs)
            return name
        return _func

    module._load_orphan_module = loader

    assert module.integrate_orphans(1) == 'integrate_orphans'
    assert calls['integrate_orphans'] == ((1,), {'retries': 5, 'delay': 0.7})
    assert module.post_round_orphan_scan() == 'post_round_orphan_scan'
    assert calls['post_round_orphan_scan'] == ((), {'retries': 5, 'delay': 0.7})
    sys.modules.pop("sandbox_settings", None)
