import contextvars
import types
import sys

import pytest

from objective_surface_policy import OBJECTIVE_ADJACENT_UNSAFE_PATHS

REQUIRED_OBJECTIVE_ADJACENT_PATHS = {
    "config/objective_hash_lock.json",
    "objective_guard.py",
    "objective_hash_lock.py",
    "tools/objective_guard_manifest_cli.py",
    "reward_dispatcher.py",
    "kpi_reward_core.py",
    "reward_sanity_checker.py",
    "kpi_editing_detector.py",
    "mvp_evaluator.py",
    "menace/core/evaluator.py",
    "neurosales/neurosales/hierarchical_reward.py",
    "neurosales/neurosales/reward_ledger.py",
    "billing/billing_ledger.py",
    "billing/billing_logger.py",
    "billing/stripe_ledger.py",
    "stripe_billing_router.py",
    "finance_router_bot.py",
    "stripe_watchdog.py",
    "startup_health_check.py",
    "finance_logs/",
}


class DummyManager:
    def __init__(self, *_, **kwargs):
        self.bot_registry = kwargs.get("bot_registry")
        self.data_bot = kwargs.get("data_bot")
        self.evolution_orchestrator = kwargs.get("evolution_orchestrator")
        self.quick_fix = kwargs.get("quick_fix") or object()


class DummyRegistry:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register_bot(self, name: str, **_kwargs) -> None:
        self.registered.append(name)

    def update_bot(self, name: str, module_path: str, **_kwargs) -> None:  # pragma: no cover - minimal stub
        self.updated = (name, module_path)


class DummyDB:
    def __init__(self) -> None:
        self.logged: list[tuple[str, str, float]] = []

    def log_eval(self, name: str, metric: str, value: float) -> None:
        self.logged.append((name, metric, value))


class DummyDataBot:
    def __init__(self) -> None:
        self.db = DummyDB()

    def roi(self, name: str) -> float:  # pragma: no cover - simple stub
        return 0.0


class DummyOrchestrator:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register_bot(self, name: str) -> None:
        self.registered.append(name)


@pytest.fixture(autouse=True)
def stub_self_coding_runtime(monkeypatch):
    manager_mod = types.ModuleType("menace.self_coding_manager")
    manager_mod.SelfCodingManager = DummyManager
    engine_mod = types.ModuleType("menace.self_coding_engine")
    engine_mod.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
    engine_mod.SelfCodingEngine = object
    monkeypatch.setitem(sys.modules, "menace.self_coding_manager", manager_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", manager_mod)
    monkeypatch.setitem(sys.modules, "menace.self_coding_engine", engine_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", engine_mod)
    yield


@pytest.fixture(autouse=True)
def reset_policy_cache():
    from menace_sandbox.self_coding_policy import get_self_coding_policy

    get_self_coding_policy.cache_clear()
    yield
    get_self_coding_policy.cache_clear()


def _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator):
    manager = DummyManager(
        bot_registry=registry,
        data_bot=data_bot,
        evolution_orchestrator=orchestrator,
    )

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class DummyBot:
        name = "DummyPolicyBot"

        def __init__(self, manager=None, evolution_orchestrator=None):
            self.manager = manager
            self.evolution_orchestrator = evolution_orchestrator

    DummyBot(manager=manager, evolution_orchestrator=orchestrator)
    return DummyBot


def test_all_bots_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MENACE_SELF_CODING_ALLOWLIST", raising=False)
    monkeypatch.delenv("MENACE_SELF_CODING_DENYLIST", raising=False)
    from menace_sandbox import coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator)

    assert "DummyPolicyBot" in registry.registered
    assert "DummyPolicyBot" in orchestrator.registered


def test_denylist_disables_matching_bot(monkeypatch):
    monkeypatch.setenv("MENACE_SELF_CODING_DENYLIST", "DummyPolicyBot")
    monkeypatch.delenv("MENACE_SELF_CODING_ALLOWLIST", raising=False)
    from menace_sandbox import coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator)

    assert registry.registered == []
    assert orchestrator.registered == []


def test_allowlist_limits_participation(monkeypatch):
    monkeypatch.setenv("MENACE_SELF_CODING_ALLOWLIST", "OtherBot")
    monkeypatch.delenv("MENACE_SELF_CODING_DENYLIST", raising=False)
    from menace_sandbox import coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator)

    assert registry.registered == []
    assert orchestrator.registered == []


def test_patch_promotion_defaults_block_objective_adjacent_paths(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import get_patch_promotion_policy

    monkeypatch.delenv("MENACE_SELF_CODING_SAFE_PATHS", raising=False)
    monkeypatch.delenv("MENACE_SELF_CODING_UNSAFE_PATHS", raising=False)
    repo_root = tmp_path
    blocked = [
        repo_root / "reward_dispatcher.py",
        repo_root / "mvp_evaluator.py",
        repo_root / "kpi_reward_core.py",
        repo_root / "billing" / "stripe_ledger.py",
        repo_root / "finance_router_bot.py",
        repo_root / "stripe_billing_router.py",
    ]
    allowed = repo_root / "tools" / "safe_module.py"
    for path in blocked + [allowed]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n", encoding="utf-8")

    policy = get_patch_promotion_policy(repo_root=repo_root)

    for path in blocked:
        assert not policy.is_safe_target(path)
    assert policy.is_safe_target(allowed)


def test_patch_promotion_env_cannot_override_canonical_blocked_path(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import get_patch_promotion_policy

    monkeypatch.delenv("MENACE_SELF_CODING_SAFE_PATHS", raising=False)
    monkeypatch.setenv("MENACE_SELF_CODING_UNSAFE_PATHS", "")
    blocked_default = tmp_path / "reward_dispatcher.py"
    blocked_default.write_text("pass\n", encoding="utf-8")

    policy = get_patch_promotion_policy(repo_root=tmp_path)

    assert not policy.is_safe_target(blocked_default)


def test_patch_promotion_env_extend_merges_with_canonical_paths(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import get_patch_promotion_policy

    monkeypatch.delenv("MENACE_SELF_CODING_SAFE_PATHS", raising=False)
    monkeypatch.setenv("MENACE_SELF_CODING_UNSAFE_PATHS", "custom_sensitive")
    custom_target = tmp_path / "custom_sensitive" / "worker.py"
    canonical_target = tmp_path / "kpi_reward_core.py"
    safe_target = tmp_path / "src" / "worker.py"
    for path in (custom_target, canonical_target, safe_target):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n", encoding="utf-8")

    policy = get_patch_promotion_policy(repo_root=tmp_path)

    assert not policy.is_safe_target(custom_target)
    assert not policy.is_safe_target(canonical_target)
    assert policy.is_safe_target(safe_target)


def test_patch_policy_merges_canonical_paths_into_environment(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import get_patch_promotion_policy

    monkeypatch.delenv("MENACE_SELF_CODING_SAFE_PATHS", raising=False)
    monkeypatch.setenv("MENACE_SELF_CODING_UNSAFE_PATHS", "custom_sensitive")

    get_patch_promotion_policy(repo_root=tmp_path)

    import os

    merged = os.environ.get("MENACE_SELF_CODING_UNSAFE_PATHS")
    assert merged is not None
    assert "custom_sensitive" in merged
    assert "reward_dispatcher.py" in merged
    assert "menace/core/evaluator.py" in merged




def test_patch_promotion_rejects_protected_paths_even_without_env(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import evaluate_patch_promotion, get_patch_promotion_policy

    monkeypatch.delenv("MENACE_SELF_CODING_SAFE_PATHS", raising=False)
    monkeypatch.delenv("MENACE_SELF_CODING_UNSAFE_PATHS", raising=False)

    protected = tmp_path / "billing" / "billing_ledger.py"
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text("x\n", encoding="utf-8")

    policy = get_patch_promotion_policy(repo_root=tmp_path)
    decision = evaluate_patch_promotion(
        policy=policy,
        roi_delta={"total": "1.0"},
        patch_validation={"valid": True},
        source_path=protected,
    )

    assert decision.allowed is False
    assert "unsafe_target" in decision.reasons


def test_is_self_coding_unsafe_path_uses_default_and_env(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import is_self_coding_unsafe_path

    monkeypatch.setenv("MENACE_SELF_CODING_UNSAFE_PATHS", "custom_area")

    assert is_self_coding_unsafe_path("reward_dispatcher.py", repo_root=tmp_path)
    assert is_self_coding_unsafe_path("custom_area/worker.py", repo_root=tmp_path)
    assert not is_self_coding_unsafe_path("safe/worker.py", repo_root=tmp_path)


def test_is_self_coding_unsafe_path_blocks_all_canonical_objective_paths(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import is_self_coding_unsafe_path

    monkeypatch.delenv("MENACE_SELF_CODING_UNSAFE_PATHS", raising=False)

    for rule in OBJECTIVE_ADJACENT_UNSAFE_PATHS:
        if rule.endswith("/"):
            candidate = tmp_path / rule.rstrip("/") / "nested.py"
        else:
            candidate = tmp_path / rule
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("pass\n", encoding="utf-8")
        assert is_self_coding_unsafe_path(candidate, repo_root=tmp_path), rule


def test_objective_adjacent_inventory_contains_required_control_and_payout_paths():
    assert REQUIRED_OBJECTIVE_ADJACENT_PATHS.issubset(set(OBJECTIVE_ADJACENT_UNSAFE_PATHS))
