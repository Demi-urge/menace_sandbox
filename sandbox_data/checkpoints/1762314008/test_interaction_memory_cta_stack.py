from neurosales.interaction_memory import InteractionMemory, CTAEvent


def test_cta_event_stack_lifecycle():
    mem = InteractionMemory()
    mem.push_event("hello", escalation_level=1)
    mem.push_event("follow up", escalation_level=2, success=True)
    chain = mem.current_chain()
    assert len(chain) == 2
    assert chain[0].message == "hello" and chain[0].escalation_level == 1
    assert chain[1].success is True
    popped = mem.pop_chain()
    assert len(popped) == 2
    assert mem.current_chain() == []
