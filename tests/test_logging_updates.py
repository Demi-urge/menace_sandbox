import os
import logging
import types
import sys
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def _stub_deps():
    if "jinja2" not in sys.modules:
        mod = types.ModuleType("jinja2")
        mod.Template = type("T", (), {"render": lambda self, *a, **k: ""})
        sys.modules["jinja2"] = mod
    for name in ["yaml", "numpy", "matplotlib", "matplotlib.pyplot", "cryptography", "cryptography.hazmat", "cryptography.hazmat.primitives", "cryptography.hazmat.primitives.asymmetric"]:  # path-ignore  # noqa: E501
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules.pop('menace.database_steward_bot', None)
    sys.modules.pop('menace.capital_management_bot', None)
    sa = types.ModuleType("sqlalchemy")
    sa.create_engine = lambda *a, **k: types.SimpleNamespace()
    sa.MetaData = lambda: types.SimpleNamespace(create_all=lambda eng: None)
    sa.Table = lambda *a, **k: None
    sa.Column = lambda *a, **k: None
    sa.Integer = object
    sa.String = object
    sa.orm = types.SimpleNamespace(sessionmaker=lambda **k: lambda: None)
    sa.exc = types.SimpleNamespace(SQLAlchemyError=Exception)
    sys.modules["sqlalchemy"] = sa
    sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))
    if "cryptography.hazmat.primitives.asymmetric.ed25519" not in sys.modules:
        ed = types.ModuleType("ed25519")
        ed.Ed25519PrivateKey = type("P", (), {})
        ed.Ed25519PublicKey = type("U", (), {})
        sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"] = ed
    sys.modules.setdefault("cryptography.hazmat.primitives.serialization", types.ModuleType("serialization"))  # noqa: E501


def test_offer_bot_logs_memory_errors(tmp_path, caplog):
    _stub_deps()
    import menace.offer_testing_bot as ot
    db = ot.OfferDB(tmp_path / "o.db")

    class BadMM:
        def subscribe(self, *a, **k):
            raise RuntimeError("boom")

        def store(self, *a, **k):
            raise RuntimeError("boom")

    caplog.set_level(logging.ERROR)
    ot.OfferTestingBot(db, memory_mgr=BadMM())
    assert "memory manager subscribe failed" in caplog.text

    bot = ot.OfferTestingBot(db)
    caplog.clear()
    bot.memory_mgr = BadMM()
    bot._on_variant_event("t", {})
    assert "memory store failed" in caplog.text


def test_database_steward_safe_mode_logs(tmp_path, caplog):
    pytest.skip("dependencies missing")

    class DummyErrDB:
        def is_safe_mode(self, module):
            return True

    err_bot = eb.ErrorBot(eb.ErrorDB(tmp_path / "e.db"))  # noqa: F821
    err_bot.db = DummyErrDB()

    class Conv:
        def notify(self, msg):
            raise RuntimeError("boom")

    bot = dsb.DatabaseStewardBot(  # noqa: F821
        sql_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        conversation_bot=Conv(),
        error_bot=err_bot,
    )
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        bot._check_safe()
    assert "notify failed" in caplog.text


def test_capital_prediction_logs(caplog):
    _stub_deps()
    import menace.capital_management_bot as cmb

    class BadPred:
        def predict(self, feats):
            raise RuntimeError("boom")

    class DummyPM:
        def __init__(self):
            self.registry = {"b": types.SimpleNamespace(bot=BadPred())}

        def assign_prediction_bots(self, bot):
            return ["b"]

    pm = DummyPM()
    bot = cmb.CapitalManagementBot(prediction_manager=pm)
    bot.assigned_prediction_bots = ["b"]
    caplog.set_level(logging.ERROR)
    bot._apply_prediction_bots(1.0, [1.0])
    assert "prediction bot BadPred failed" in caplog.text


@pytest.mark.skipif(
    'menace.model_automation_pipeline' not in sys.modules
    and os.getenv('SKIP_PIPELINE_TEST'),
    reason="pipeline unavailable",
)
def test_pipeline_prime_logging(caplog):
    _stub_deps()
    try:
        import menace.model_automation_pipeline as mapl
        from menace.research_aggregator_bot import ResearchAggregatorBot
    except Exception:
        pytest.skip("pipeline import failed")
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    agg = ResearchAggregatorBot([], context_builder=builder)
    pipeline = mapl.ModelAutomationPipeline(aggregator=agg, context_builder=builder)

    class BadBot:
        def prime(self):
            raise RuntimeError("boom")
    pipeline._bots = [BadBot()]
    caplog.set_level(logging.ERROR)
    pipeline._prime_bots()
    assert "prime() failed" in caplog.text
