import sandbox_runner.generative_stub_provider as gsp


def sample_func(name: str, count: int, active: bool) -> None:
    """Sample function used for stub generation."""
    return None


def test_rule_based_stub_generation(monkeypatch):
    """Stub generation falls back to deterministic rule-based values."""
    async def fake_aload_generator():
        return None

    # Avoid loading optional model backends
    monkeypatch.setattr(gsp, "_aload_generator", fake_aload_generator)
    gsp._CACHE.clear()

    stubs = gsp.generate_stubs([{}], {"target": sample_func})[0]
    assert gsp._type_matches(stubs["name"], str)
    assert gsp._type_matches(stubs["count"], int)
    assert gsp._type_matches(stubs["active"], bool)
