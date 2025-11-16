import pytest
import self_improvement_policy as sip

def test_rl_strategy_cannot_instantiate():
    with pytest.raises(TypeError):
        sip.RLStrategy()
