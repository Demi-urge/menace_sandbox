import universal_retriever as ur_mod


def test_retriever_reload_on_feedback(monkeypatch):
    monkeypatch.setattr(
        ur_mod.UniversalRetriever, "_load_reliability_stats", lambda self: None
    )

    class DummyBus:
        def __init__(self) -> None:
            self.cb = None

        def subscribe(self, topic, callback):
            self.cb = callback

        def publish(self, topic, event):
            if self.cb:
                self.cb(topic, event)

    bus = DummyBus()
    ur = ur_mod.UniversalRetriever(
        event_bus=bus,
        enable_model_ranking=False,
        enable_reliability_bias=False,
        code_db=object(),
    )

    called = []
    monkeypatch.setattr(ur, "reload_reliability_scores", lambda: called.append(True))

    bus.publish("retrieval:feedback", {"db": "x"})
    assert called

