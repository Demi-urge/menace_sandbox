import sys
import types
import yaml


stub_cbi = types.ModuleType("menace.coding_bot_interface")
stub_cbi.self_coding_managed = lambda *a, **k: (lambda cls: cls)
stub_cbi.manager_generate_helper = lambda *_a, **_k: None
sys.modules["menace.coding_bot_interface"] = stub_cbi


class UnifiedEventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list] = {}

    def subscribe(self, topic: str, handler):
        self._subs.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload):
        for h in self._subs.get(topic, []):
            h(topic, payload)


def test_threshold_event_published(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={
            "bot": types.SimpleNamespace(roi_drop=-5.0, error_threshold=5.0)
        },
    )
    bus = UnifiedEventBus()
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(
        mdb,
        event_bus=bus,
        settings=settings,
        roi_drop_threshold=-5.0,
        error_threshold=5.0,
    )
    events: list[dict] = []
    degraded: list[dict] = []
    bus.subscribe("bot:degraded", lambda _t, e: degraded.append(e))
    bot.subscribe_threshold_breaches(lambda e: events.append(e))
    bot.collect("bot", revenue=10.0, expense=0.0, errors=0)
    bot.collect("bot", revenue=0.0, expense=0.0, errors=10)
    assert len(events) == 1
    assert events[0]["roi_breach"] and events[0]["error_breach"]
    assert not events[0]["test_failure_breach"]
    assert degraded and degraded[0]["roi_breach"]
    # Degradation events provide summary metrics and deltas for downstream consumers
    assert "roi_drop" in degraded[0]
    assert "error_rate" in degraded[0]
    assert "tests_failed" in degraded[0]
    assert "delta_roi" in degraded[0]
    assert "delta_errors" in degraded[0]
    assert "delta_tests_failed" in degraded[0]


def test_check_degradation_callback(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={},
    )
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(
        mdb,
        settings=settings,
        roi_drop_threshold=-0.1,
        error_threshold=1.0,
    )
    bot.event_bus = None
    events: list[dict] = []
    bot.check_degradation("bot", roi=10.0, errors=0.0)
    bot.check_degradation("bot", roi=0.0, errors=10.0, callback=lambda e: events.append(e))
    assert events and events[0]["roi_breach"] and events[0]["error_breach"]
    assert "bot" in bot._thresholds


def test_forecasting_model_detection(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={},
    )
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(
        mdb,
        settings=settings,
        roi_drop_threshold=-0.1,
        error_threshold=1.0,
    )
    for _ in range(5):
        assert not bot.check_degradation("bot", roi=10.0, errors=0.0, test_failures=0.0)
    # Slight changes stay within the model's confidence interval
    assert not bot.check_degradation("bot", roi=9.95, errors=0.1, test_failures=0.0)
    # Significant deviation triggers degradation
    assert bot.check_degradation("bot", roi=0.0, errors=5.0, test_failures=3.0)
    hist = bot._forecast_history["bot"]
    assert len(hist["roi"]) >= 7


def test_forecast_threshold_persist(monkeypatch, tmp_path):
    """Thresholds derived from forecasts are persisted and broadcast."""
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)

    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={},
    )
    mdb = db.MetricsDB(tmp_path / "m.db")
    bus = UnifiedEventBus()
    events: list[dict] = []
    bus.subscribe("data:thresholds_refreshed", lambda _t, e: events.append(e))

    bot = db.DataBot(mdb, event_bus=bus, settings=settings)

    class Model:
        def __init__(self) -> None:
            self.count = 0

        def fit(self, hist):
            self.count = len(hist)
            return self

        def forecast(self):
            # Widen confidence interval on each call to force threshold updates
            return 5.0, 5.0 - self.count, 5.0 + self.count

    bot.threshold_service = types.SimpleNamespace(
        get=lambda _b, _s: db.ROIThresholds(-0.1, 1.0, 0.0),
        update=lambda *a, **k: None,
    )
    bot._forecast_models["bot"] = {
        "roi": Model(),
        "errors": Model(),
        "tests_failed": Model(),
    }
    bot._forecast_meta["bot"] = {"model": "m", "confidence": 0.9, "params": {}}

    calls: list[tuple] = []
    monkeypatch.setattr(
        db,
        "persist_sc_thresholds",
        lambda *a, **k: calls.append((a, k)),
    )

    for _ in range(3):
        bot.check_degradation("bot", roi=5.0, errors=0.0, test_failures=0.0)

    assert len(calls) > 1
    assert len(events) > 1


def test_internalize_persists_defaults(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    import importlib
    sys.modules.pop("menace.sandbox_settings", None)
    real_settings = importlib.import_module("menace.sandbox_settings")
    monkeypatch.setitem(sys.modules, "sandbox_settings", real_settings)
    import menace.chunking as chunking  # noqa: WPS433
    chunking._SETTINGS = real_settings.SandboxSettings()
    import menace.self_coding_thresholds as sct  # noqa: WPS433
    cfg_path = tmp_path / "sc.yaml"
    monkeypatch.setattr(sct, "_CONFIG_PATH", cfg_path)
    import menace.data_bot as db  # noqa: WPS433
    monkeypatch.setattr(db, "psutil", None)
    stub_th = types.ModuleType("menace.sandbox_runner.test_harness")
    stub_th.run_tests = lambda *a, **k: None
    stub_th.TestHarnessResult = object
    monkeypatch.setitem(sys.modules, "menace.sandbox_runner.test_harness", stub_th)
    stub_cd = types.ModuleType("menace.code_database")
    stub_cd.PatchRecord = object
    monkeypatch.setitem(sys.modules, "menace.code_database", stub_cd)
    stub_engine = types.ModuleType("menace.self_coding_engine")
    stub_engine.SelfCodingEngine = object
    monkeypatch.setitem(sys.modules, "menace.self_coding_engine", stub_engine)
    stub_ra = types.ModuleType("menace.research_aggregator_bot")
    stub_ra.ResearchAggregatorBot = object
    stub_ra.ResearchItem = object
    monkeypatch.setitem(sys.modules, "menace.research_aggregator_bot", stub_ra)
    import menace.self_coding_manager as scm  # noqa: WPS433
    monkeypatch.setattr(
        scm,
        "persist_sc_thresholds",
        lambda bot, roi_drop=None, error_increase=None, test_failure_increase=None, **_: (
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "bots": {
                            bot: {
                                "roi_drop": roi_drop,
                                "error_increase": error_increase,
                                "test_failure_increase": test_failure_increase,
                            }
                        }
                    },
                    sort_keys=False,
                )
            )
        ),
    )

    class DummyManager:
        def __init__(self, *_a, **_k):
            self.quick_fix = object()
            self.logger = types.SimpleNamespace(exception=lambda *a, **k: None)

    monkeypatch.setattr(scm, "SelfCodingManager", DummyManager)

    class DummyRegistry:
        def register_bot(self, *_a, **_k):
            return None

    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.0,
        bot_thresholds={},
    )
    data_bot = types.SimpleNamespace(settings=settings, event_bus=None)

    scm.internalize_coding_bot(
        "bot",
        engine=object(),
        pipeline=object(),
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )

    data = yaml.safe_load(cfg_path.read_text())
    bot_cfg = data["bots"]["bot"]
    assert bot_cfg["roi_drop"] == -0.1
    assert bot_cfg["error_increase"] == 1.0
    assert bot_cfg["test_failure_increase"] == 0.0

    thresh = db.load_sc_thresholds("bot", settings, path=cfg_path)
    assert thresh.roi_drop == -0.1
    assert thresh.error_increase == 1.0
    assert thresh.test_failure_increase == 0.0


def test_internalize_records_thresholds_and_emits_test_failure(monkeypatch, tmp_path):
    stub = types.ModuleType("vector_metrics_db")
    monkeypatch.setitem(sys.modules, "menace.vector_metrics_db", stub)
    sys.modules.setdefault("sandbox_settings", sys.modules["menace.sandbox_settings"])
    from menace.data_bot import DataBot, MetricsDB  # noqa: WPS433
    from menace.bot_registry import BotRegistry  # noqa: WPS433

    bus = UnifiedEventBus()
    settings = types.SimpleNamespace(
        self_coding_roi_drop=-0.1,
        self_coding_error_increase=1.0,
        self_coding_test_failure_increase=0.5,
        bot_thresholds={},
    )
    mdb = MetricsDB(tmp_path / "m.db")
    data_bot = DataBot(
        mdb,
        event_bus=bus,
        settings=settings,
        roi_drop_threshold=-0.1,
        error_threshold=1.0,
        test_failure_threshold=0.5,
    )
    registry = BotRegistry(event_bus=bus)
    manager = types.SimpleNamespace()
    registry.register_bot(
        "sample",
        roi_threshold=-0.1,
        error_threshold=1.0,
        test_failure_threshold=0.5,
        manager=manager,
        data_bot=data_bot,
        is_coding_bot=True,
    )
    node = registry.graph.nodes["sample"]
    assert node["roi_threshold"] == -0.1
    assert node["error_threshold"] == 1.0
    assert node["test_failure_threshold"] == 0.5

    events: list[dict] = []
    bus.subscribe("bot:degraded", lambda _t, e: events.append(e))
    data_bot.check_degradation("sample", roi=-1.0, errors=0.0, test_failures=1.0)
    assert events and events[0]["test_failure_breach"]
