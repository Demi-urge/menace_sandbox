import pytest

orchestration = pytest.importorskip("menace.self_improvement.orchestration")


def test_orchestration_reexports_function():
    with pytest.raises(RuntimeError):
        orchestration.integrate_orphans()
