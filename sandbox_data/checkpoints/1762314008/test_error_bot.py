import pytest
import menace.error_bot as eb
import menace.data_bot as db
import menace.prediction_manager_bot as pmb
import menace.capital_management_bot as cmb
import menace.bot_database as bdbm
import menace.menace as mn
import menace.knowledge_graph as kg
import menace.chatgpt_enhancement_bot as ceb
import menace.chatgpt_idea_bot as cib
import os
import json
import logging
from pathlib import Path

pytest.importorskip("sqlalchemy")


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build(self, *a, **k):
        return ""


def make_metrics(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord(
        bot="a", cpu=1.0, memory=2.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=5
    )
    mdb.add(rec)
    return mdb


def test_add_error_duplicate(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(eb.ErrorDB, "add_embedding", lambda *a, **k: None)
    db = eb.ErrorDB(tmp_path / "e.db")

    captured: dict[str, int | None] = {"id": None}
    orig = eb.insert_if_unique

    def wrapper(*args, **kwargs):
        res = orig(*args, **kwargs)
        captured["id"] = res
        return res

    monkeypatch.setattr(eb, "insert_if_unique", wrapper)

    first = db.add_error("dup", type_="t", description="d", resolution="r")
    captured["id"] = None
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        second = db.add_error("dup", type_="t", description="d", resolution="r")
    assert first == second
    assert captured["id"] == first
    assert "duplicate" in caplog.text.lower()


def test_error_content_hash_unique_index(tmp_path, monkeypatch):
    monkeypatch.setattr(eb.ErrorDB, "add_embedding", lambda *a, **k: None)
    old = eb.GLOBAL_ROUTER
    db = eb.ErrorDB(tmp_path / "e.db")
    indexes = {row[1]: row[2] for row in db.conn.execute("PRAGMA index_list('errors')").fetchall()}
    assert indexes.get("idx_errors_content_hash") == 1
    eb.GLOBAL_ROUTER = old


def test_add_error_duplicate_different_message(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(eb.ErrorDB, "add_embedding", lambda *a, **k: None)
    db = eb.ErrorDB(tmp_path / "e.db")

    captured: dict[str, int | None] = {"id": None}
    orig = eb.insert_if_unique

    def wrapper(*args, **kwargs):
        res = orig(*args, **kwargs)
        captured["id"] = res
        return res

    monkeypatch.setattr(eb, "insert_if_unique", wrapper)

    first = db.add_error("dup1", type_="t", description="d", resolution="r")
    captured["id"] = None
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        second = db.add_error("dup2", type_="t", description="d", resolution="r")
    assert first == second
    assert captured["id"] == first
    assert "duplicate" in caplog.text.lower()


def test_handle_known(tmp_path):
    e = eb.ErrorDB(tmp_path / "e.db")
    e.add_known("err", "fix")
    bot = eb.ErrorBot(e, make_metrics(tmp_path), context_builder=DummyBuilder())
    res = bot.handle_error("err")
    assert res == "fix"
    assert e.discrepancies().empty


def test_handle_unknown(tmp_path):
    e = eb.ErrorDB(tmp_path / "e.db")
    bot = eb.ErrorBot(e, make_metrics(tmp_path), context_builder=DummyBuilder())
    bot.handle_error("unknown")
    df = e.discrepancies()
    assert len(df) == 1


def test_discrepancies_scope(tmp_path):
    e = eb.ErrorDB(tmp_path / "e.db")
    e.log_discrepancy("local", source_menace_id="a")
    e.log_discrepancy("remote", source_menace_id="b")
    df_local = e.discrepancies(source_menace_id="a", scope="local")
    assert list(df_local["message"]) == ["local"]
    df_global = e.discrepancies(source_menace_id="a", scope="global")
    assert list(df_global["message"]) == ["remote"]
    df_all = e.discrepancies(source_menace_id="a", scope="all")
    assert set(df_all["message"]) == {"local", "remote"}


def test_top_error_module_scope(tmp_path):
    err_db = eb.ErrorDB(tmp_path / "e.db")
    # Insert telemetry for two menace instances
    err_db.conn.execute(
        "INSERT INTO telemetry(bot_id, error_type, module_counts, resolution_status, source_menace_id) VALUES (?,?,?,?,?)",  # noqa: E501
        ("b1", "E", json.dumps({"m1": 2}), "unresolved", "a"),
    )
    err_db.conn.execute(
        "INSERT INTO telemetry(bot_id, error_type, module_counts, resolution_status, source_menace_id) VALUES (?,?,?,?,?)",  # noqa: E501
        ("b2", "E", json.dumps({"m2": 3}), "unresolved", "b"),
    )
    err_db.conn.commit()

    local = err_db.top_error_module(scope="local", source_menace_id="a")
    assert local and local[1] == "m1"

    global_scope = err_db.top_error_module(scope="global", source_menace_id="a")
    assert global_scope and global_scope[1] == "m2"

    all_scope = err_db.top_error_module(scope="all", source_menace_id="a")
    assert all_scope and all_scope[1] == "m2" and all_scope[3] == 3


def test_summarize_telemetry_scope(tmp_path):
    err_db = eb.ErrorDB(tmp_path / "e.db")
    metrics = make_metrics(tmp_path)
    bot = eb.ErrorBot(err_db, metrics, context_builder=DummyBuilder())
    # Insert telemetry with different menace ids
    err_db.conn.execute(
        "INSERT INTO telemetry(error_type, resolution_status, source_menace_id) VALUES (?,?,?)",
        ("L", "successful", "a"),
    )
    err_db.conn.execute(
        "INSERT INTO telemetry(error_type, resolution_status, source_menace_id) VALUES (?,?,?)",
        ("G", "unresolved", "b"),
    )
    err_db.conn.commit()

    local = bot.summarize_telemetry(limit=10, scope="local", source_menace_id="a")
    assert local == [{"error_type": "L", "count": 1.0, "success_rate": 1.0}]

    global_res = bot.summarize_telemetry(limit=10, scope="global", source_menace_id="a")
    assert global_res == [{"error_type": "G", "count": 1.0, "success_rate": 0.0}]

    all_res = bot.summarize_telemetry(limit=10, scope="all", source_menace_id="a")
    mapping = {r["error_type"]: r for r in all_res}
    assert set(mapping) == {"L", "G"}


def test_monitor_logs(tmp_path):
    mdb = make_metrics(tmp_path)
    e = eb.ErrorDB(tmp_path / "e.db")
    bot = eb.ErrorBot(e, mdb, context_builder=DummyBuilder())
    bot.monitor()
    df = e.discrepancies()
    assert not df.empty


class DummyPredictor:
    def predict(self, df):
        return ["future_err"]


def test_predict_and_roi_scan(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(
        db.MetricRecord(
            bot="b",
            cpu=1.0,
            memory=1.0,
            response_time=0.1,
            disk_io=1.0,
            net_io=1.0,
            errors=0,
            revenue=10.0,
            expense=1.0,
        )
    )
    mdb.add(
        db.MetricRecord(
            bot="b",
            cpu=1.0,
            memory=1.0,
            response_time=0.2,
            disk_io=1.0,
            net_io=1.0,
            errors=0,
            revenue=0.0,
            expense=0.0,
        )
    )
    capital = cmb.CapitalManagementBot(data_bot=db.DataBot(mdb))
    data_bot = db.DataBot(mdb, capital_bot=capital)
    pm = pmb.PredictionManager(tmp_path / "reg.json")
    pm.register_bot(DummyPredictor(), {"scope": ["errors"], "risk": ["low"]})
    bot = eb.ErrorBot(
        eb.ErrorDB(tmp_path / "e2.db"),
        mdb,
        prediction_manager=pm,
        data_bot=data_bot,
        context_builder=DummyBuilder(),
    )
    assert bot.prediction_ids
    preds = bot.predict_errors()
    assert "future_err" in preds
    bot.scan_roi_discrepancies(threshold=0.1)
    df = bot.db.discrepancies()
    assert not df.empty


def test_roi_discrepancy_links(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(
        db.MetricRecord(
            bot="b",
            cpu=1.0,
            memory=1.0,
            response_time=0.1,
            disk_io=1.0,
            net_io=1.0,
            errors=0,
            revenue=15.0,
            expense=5.0,
        )
    )
    mdb.add(
        db.MetricRecord(
            bot="b",
            cpu=1.0,
            memory=1.0,
            response_time=0.2,
            disk_io=1.0,
            net_io=1.0,
            errors=0,
            revenue=0.0,
            expense=0.0,
        )
    )
    capital = cmb.CapitalManagementBot(data_bot=db.DataBot(mdb))
    data_bot = db.DataBot(mdb, capital_bot=capital)

    menace = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'men.db'}")
    with menace.engine.begin() as conn:
        conn.execute(menace.models.insert().values(model_id=1, model_name="m"))
        conn.execute(menace.bots.insert().values(bot_id=1, bot_name="b"))
        conn.execute(menace.workflows.insert().values(workflow_id=1, workflow_name="w"))
        conn.execute(
            menace.enhancements.insert().values(
                enhancement_id=1,
                description_of_change="e",
                reason_for_change="",
                performance_delta=0.0,
                timestamp="t",
                triggered_by="x",
                source_menace_id="",
            )
        )

    bdb = bdbm.BotDB(tmp_path / "b.db")
    rec = bdbm.BotRecord(name="b", bid=1, hierarchy_level="L1")
    bid = bdb.add_bot(rec)
    bdb.link_model(bid, 1)
    bdb.link_workflow(bid, 1)
    bdb.link_enhancement(bid, 1)

    err_db = eb.ErrorDB(tmp_path / "e.db")
    bot = eb.ErrorBot(
        err_db,
        mdb,
        data_bot=data_bot,
        menace_db=menace,
        bot_db=bdb,
        context_builder=DummyBuilder(),
    )
    bot.scan_roi_discrepancies(threshold=5.0)

    with menace.engine.connect() as conn:
        disc = conn.execute(menace.discrepancies.select()).fetchall()
        bots = conn.execute(menace.discrepancy_bots.select()).fetchall()
        models = conn.execute(menace.discrepancy_models.select()).fetchall()
        wfs = conn.execute(menace.discrepancy_workflows.select()).fetchall()
        enhs = conn.execute(menace.discrepancy_enhancements.select()).fetchall()

    assert disc and bots and models and wfs and enhs


def test_prompt_rewriter_daemon(tmp_path):
    err_db = eb.ErrorDB(tmp_path / "e.db")
    enh_db = ceb.EnhancementDB(tmp_path / "enh.db")
    mdb = make_metrics(tmp_path)
    bot = eb.ErrorBot(err_db, mdb, enhancement_db=enh_db, context_builder=DummyBuilder())
    bot.record_runtime_error("boom")
    import time
    time.sleep(0.1)
    with enh_db._connect() as conn:
        row = conn.execute("SELECT prompt FROM prompt_history").fetchone()
    assert row is not None


def test_safe_mode_activation(tmp_path):
    import types
    import sys

    sys.modules.setdefault(
        "neurosales",
        types.SimpleNamespace(
            add_message=lambda *a, **k: None,
            get_recent_messages=lambda *a, **k: [],
            push_chain=lambda *a, **k: None,
            peek_chain=lambda *a, **k: None,
            MessageEntry=object,
            CTAChain=object,
        ),
    )
    from menace.conversation_manager_bot import ConversationManagerBot

    err_db = eb.ErrorDB(tmp_path / "e.db")

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

    client = cib.ChatGPTClient("key", context_builder=DummyBuilder())
    conv = ConversationManagerBot(client)
    metrics = make_metrics(tmp_path)
    bot = eb.ErrorBot(err_db, metrics, conversation_bot=conv, context_builder=DummyBuilder())
    bot.record_runtime_error("fail", bot_ids=["DatabaseStewardBot"])
    assert err_db.is_safe_mode("DatabaseStewardBot")
    notes = conv.get_notifications()
    assert notes and "safe mode" in notes[0]


def test_root_cause_graph(tmp_path):
    err_db = eb.ErrorDB(tmp_path / "e.db")
    mdb = make_metrics(tmp_path)
    bot = eb.ErrorBot(err_db, mdb, context_builder=DummyBuilder())
    bot.record_runtime_error(
        "boom",
        model_id=1,
        bot_ids=["Alpha"],
        code_ids=[2],
        stack_trace="trace",
    )
    causes = bot.root_causes("Alpha")
    assert any(c.startswith("error:") for c in causes)


class DummyForecaster:
    def train(self):
        pass

    def predict_error_prob(self, bot: str, steps: int = 1):
        return [0.9 for _ in range(steps)]


def test_forecaster_safe_mode(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(db.MetricRecord(bot="X", cpu=0.0, memory=0.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=1))  # noqa: E501
    fc = DummyForecaster()
    err_db = eb.ErrorDB(tmp_path / "e.db")
    bot = eb.ErrorBot(err_db, mdb, forecaster=fc, context_builder=DummyBuilder())
    bot.predict_errors()
    assert err_db.is_safe_mode("X")


def test_forecaster_cascade(tmp_path):
    pytest.importorskip("networkx")
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(db.MetricRecord(bot="A", cpu=0.0, memory=0.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=1))  # noqa: E501
    g = kg.KnowledgeGraph()
    g.graph.add_node("bot:A")
    g.graph.add_edge("bot:A", "model:1")
    g.graph.add_edge("model:1", "code:foo")

    class DummyPM:
        def assign_prediction_bots(self, _):
            return []

        def get_prediction_bots_for(self, _):
            return []

    fc = DummyForecaster()
    err_db = eb.ErrorDB(tmp_path / "e.db")
    bot = eb.ErrorBot(
        err_db,
        mdb,
        forecaster=fc,
        graph=g,
        prediction_manager=DummyPM(),
        context_builder=DummyBuilder(),
    )
    preds = bot.predict_errors()
    chain = "bot:A -> model:1 -> code:foo"
    assert chain in preds
    assert bot.last_forecast_chains["A"] == ["bot:A", "model:1", "code:foo"]


def test_forecaster_patch_rollback(tmp_path, monkeypatch):
    class DummyForecaster:
        def train(self):
            pass

        def predict_error_prob(self, bot: str, steps: int = 1):
            return [0.95 for _ in range(steps)]

    class DummyEngine:
        def __init__(self):
            self.called = False

        def apply_patch(self, path: Path, desc: str, **_: object):
            self.called = True
            original = path.read_text()
            path.write_text(original + "#patch\n")
            path.write_text(original)
            return 1, True, 0.0

    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(db.MetricRecord(bot="Z", cpu=0.0, memory=0.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=1))  # noqa: E501
    fc = DummyForecaster()
    engine = DummyEngine()
    err_db = eb.ErrorDB(tmp_path / "e.db")
    bot_file = tmp_path / "Z.py"  # path-ignore
    bot_file.write_text("def z():\n    pass\n")
    monkeypatch.chdir(tmp_path)

    bot = eb.ErrorBot(
        err_db,
        mdb,
        forecaster=fc,
        self_coding_engine=engine,
        context_builder=DummyBuilder(),
    )
    bot.predict_errors()
    assert engine.called
    assert "#patch" not in bot_file.read_text()


def test_runbook_generation(tmp_path):
    class DummyForecaster:
        def train(self):
            pass

        def predict_error_prob(self, bot: str, steps: int = 1):
            return [0.9 for _ in range(steps)]

    class DummyPM:
        def assign_prediction_bots(self, _):
            return []

        def get_prediction_bots_for(self, _):
            return []

    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(db.MetricRecord(bot="R", cpu=0.0, memory=0.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=1))  # noqa: E501
    g = kg.KnowledgeGraph()
    g.graph.add_node("bot:R")
    g.graph.add_edge("bot:R", "module:X")
    fc = DummyForecaster()
    err_db = eb.ErrorDB(tmp_path / "e.db")
    bot = eb.ErrorBot(
        err_db,
        mdb,
        forecaster=fc,
        graph=g,
        prediction_manager=DummyPM(),
        context_builder=DummyBuilder(),
    )
    preds = bot.predict_errors()
    assert any(p.endswith(".json") for p in preds)
    path = bot.generated_runbooks.get("R")
    assert path and os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data.get("mitigation")
