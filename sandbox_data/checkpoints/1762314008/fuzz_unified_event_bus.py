from hypothesis import given, strategies as st, settings
from menace.unified_event_bus import UnifiedEventBus


@settings(max_examples=50)
@given(st.lists(st.dictionaries(st.text(min_size=1, max_size=10),
                                st.one_of(st.integers(), st.text(), st.floats(allow_nan=False)),
                                max_size=5)))
def test_event_bus_roundtrip(events):
    """Publish arbitrary events and ensure all are received."""
    bus = UnifiedEventBus()
    received = []
    bus.subscribe("topic", lambda t, e: received.append(e))
    for ev in events:
        bus.publish("topic", ev)
    assert received == events
