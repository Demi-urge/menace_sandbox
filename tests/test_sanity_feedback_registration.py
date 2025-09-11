import menace.sanity_feedback as sf


class DummyEngine:
    pass


class DummyRegistry:
    def __init__(self):
        self.names = []

    def register_bot(self, name):
        self.names.append(name)


class DummyMetricsDB:
    def __init__(self):
        self.records = []

    def log_eval(self, name, metric, value):
        self.records.append((name, metric, value))


class DummyDataBot:
    def __init__(self):
        self.db = DummyMetricsDB()

    def roi(self, _name):
        return 0.0


class DummyManager:
    def __init__(self, engine, registry, data_bot):
        self.engine = engine
        self.bot_registry = registry
        self.data_bot = data_bot
    def run_patch(self, *a, **k):
        return None


def test_sanity_feedback_registers():
    registry = DummyRegistry()
    data_bot = DummyDataBot()
    manager = DummyManager(DummyEngine(), registry, data_bot)
    sf.SanityFeedback(manager, outcome_db=object())
    assert "SanityFeedback" in registry.names
    assert data_bot.db.records
