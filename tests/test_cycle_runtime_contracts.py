import types

import sandbox_runner.cycle as cycle


def test_sandbox_env_presets_accepts_dict(monkeypatch):
    env_mod = types.SimpleNamespace(SANDBOX_ENV_PRESETS={"A": "1"}, ERROR_CATEGORY_COUNTS={})
    monkeypatch.setattr(cycle, "_ENVIRONMENT_MODULE", env_mod)

    assert cycle._get_sandbox_env_presets() == [{"A": "1"}]

    cycle.SANDBOX_ENV_PRESETS = {"B": "2"}
    assert env_mod.SANDBOX_ENV_PRESETS == [{"B": "2"}]


def test_include_orphan_modules_allows_reintroduction_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(cycle, "resolve_path", lambda p: str(p))
    monkeypatch.setattr(cycle, "load_orphan_cache", lambda *_a, **_k: {})
    monkeypatch.setattr(cycle, "load_orphan_traces", lambda *_a, **_k: {})
    monkeypatch.setattr(cycle, "append_orphan_traces", lambda *_a, **_k: None)
    monkeypatch.setattr(cycle, "append_orphan_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(cycle, "append_orphan_classifications", lambda *_a, **_k: None)
    monkeypatch.setattr(cycle, "prune_orphan_cache", lambda *_a, **_k: None)

    monkeypatch.setattr(
        cycle,
        "discover_recursive_orphans",
        lambda *_a, **_k: {"helper": {"parents": [], "redundant": False}},
    )
    (tmp_path / "helper.py").write_text("x = 1\n")

    calls = {"mods": None}

    def fake_auto_include(mods, **_kwargs):
        calls["mods"] = list(mods)
        return types.SimpleNamespace(module_deltas={}), {"added": list(mods), "failed": [], "redundant": []}

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)

    ctx = types.SimpleNamespace(
        repo=tmp_path,
        module_map=set(),
        orphan_traces={},
        context_builder=types.SimpleNamespace(),
        settings=types.SimpleNamespace(orphan_modules_reintroduction=True),
    )

    cycle.include_orphan_modules(ctx)

    assert calls["mods"] == ["helper.py"]
