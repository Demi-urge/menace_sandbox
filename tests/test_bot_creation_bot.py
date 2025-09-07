import asyncio
import logging

import pytest

import menace.bot_creation_bot as bcb
import menace.data_bot as db
import menace.bot_planning_bot as bp
import menace.bot_development_bot as bd
import menace.bot_testing_bot as bt
import menace.deployment_bot as dep
import menace.error_bot as eb
import menace.task_handoff_bot as thb
import menace.research_aggregator_bot as rab
import menace.chatgpt_enhancement_bot as ceb
import menace.database_manager as dm
import menace.contrarian_db as cdb
import menace.menace as mn
import menace.trending_scraper as ts
import menace.bot_database as bdb
from menace.db_router import init_db_router

pytest.importorskip("sqlalchemy")


def _metrics(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(
        db.MetricRecord(
            bot="a",
            cpu=80.0,
            memory=90.0,
            response_time=0.1,
            disk_io=1.0,
            net_io=1.0,
            errors=0,
        )
    )
    return mdb


def _ctx_builder():
    return bd.ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")


def test_needs_new_bot(tmp_path):
    bot = bcb.BotCreationBot(
        metrics_db=_metrics(tmp_path),
        context_builder=_ctx_builder(),
    )
    assert bot.needs_new_bot()


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, feats):
        self.called = True
        return 0.0


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _):
        return ["p"]


def test_needs_new_bot_with_prediction(tmp_path):
    manager = StubManager(DummyPred())
    bot = bcb.BotCreationBot(
        metrics_db=_metrics(tmp_path),
        prediction_manager=manager,
        context_builder=_ctx_builder(),
    )
    assert bot.assigned_prediction_bots == ["p"]
    assert not bot.needs_new_bot()
    assert manager.registry["p"].bot.called


def test_create_bots(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()
    router = init_db_router("tcb", str(tmp_path / "l.db"), str(tmp_path / "s.db"))
    bdb.router = router

    class DummyClusterer:
        pass

    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        info_db=rab.InfoDB(tmp_path / "info.db"),
        db_router=router,
    )
    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        intent_clusterer=DummyClusterer(),
        context_builder=developer.context_builder,
    )
    task = bp.PlanningTask(
        description="do", complexity=1, frequency=1, expected_time=0.1, actions=["run"]
    )
    ids = asyncio.run(bot.create_bots([task]))
    assert ids and deployer.db.get(ids[0])["status"] == "success"


def test_duplicate_bot_insert(tmp_path, caplog, monkeypatch):
    import menace.db_router as dbr

    old = dbr.GLOBAL_ROUTER
    init_db_router("dup", str(tmp_path / "l.db"), str(tmp_path / "s.db"))
    bdb.router = dbr.GLOBAL_ROUTER
    monkeypatch.setattr(bdb.BotDB, "_embed_record_on_write", lambda *a, **k: None)
    dbh = bdb.BotDB()
    rec1 = bdb.BotRecord(name="b1", dependencies=["d"], resources={"cpu": 1})
    first = dbh.add_bot(rec1)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rec2 = bdb.BotRecord(
            name="b1",
            dependencies=["d"],
            resources={"cpu": 1},
            tags=["x"],
        )
        second = dbh.add_bot(rec2)
    assert first == second
    assert dbh.conn.execute("SELECT COUNT(*) FROM bots").fetchone()[0] == 1
    assert "already exists" in caplog.text.lower()
    dbr.GLOBAL_ROUTER = old
    bdb.router = old


def test_trending_scraper_receives_energy(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()
    deployer = dep.DeploymentBot(dep.DeploymentDB(tmp_path / "d.db"))

    class DummyScraper:
        def __init__(self) -> None:
            self.energy = None

        def scrape_reddit(self, energy=None):
            self.energy = energy
            return []

    class DummyCapital:
        def energy_score(self, **kw):
            return 0.5

    scraper = DummyScraper()
    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        trending_scraper=scraper,
        capital_bot=DummyCapital(),
        context_builder=developer.context_builder,
    )
    task = bp.PlanningTask(
        description="do",
        complexity=1,
        frequency=1,
        expected_time=0.1,
        actions=["run"],
    )
    asyncio.run(bot.create_bots([task]))
    assert scraper.energy is not None


def test_create_bots_logs_errors(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()
    err_db = eb.ErrorDB(tmp_path / "e.db")
    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        code_db=dep.CodeDB(tmp_path / "c.db"),
        error_db=err_db,
    )
    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        error_bot=eb.ErrorBot(err_db, context_builder=developer.context_builder),
        context_builder=developer.context_builder,
    )
    developer.errors.append("integration fail")
    task = bp.PlanningTask(
        description="do",
        complexity=1,
        frequency=1,
        expected_time=0.1,
        actions=["run"],
    )
    asyncio.run(bot.create_bots([task], model_id=1))
    row = err_db.conn.execute(
        "SELECT id FROM errors WHERE message='integration fail'"
    ).fetchone()
    assert row is not None
    eid = row[0]
    linked = err_db.conn.execute(
        "SELECT model_id FROM error_model WHERE error_id=?", (eid,)
    ).fetchone()
    assert linked and linked[0] == 1


def test_create_bots_records_workflows(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    info_db = rab.InfoDB(tmp_path / "info.db")
    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        workflow_db=wf_db,
        info_db=info_db,
        enh_db=enh_db,
    )
    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        context_builder=developer.context_builder,
    )

    dm.DB_PATH = tmp_path / "models.db"
    model_id = dm.add_model("m", db_path=dm.DB_PATH)
    info_db.set_current_model(model_id)
    info_db.add(rab.ResearchItem(topic="t", content="c", timestamp=0.0))
    eid = enh_db.add(ceb.Enhancement(idea="i", rationale="r", model_ids=[model_id]))

    task = bp.PlanningTask(
        description="do", complexity=1, frequency=1, expected_time=0.1, actions=["run"]
    )
    asyncio.run(bot.create_bots([task], model_id=model_id, enhancements=[eid]))

    recs = wf_db.fetch()
    assert recs and recs[0].status == "pending"
    with dm.get_connection(db_path=dm.DB_PATH) as conn:
        row = conn.execute(
            "SELECT workflow_id FROM models WHERE id=?", (model_id,)
        ).fetchone()
    assert row and row[0] == recs[0].wid
    row = deployer.bot_db.conn.execute(
        "SELECT workflow_id FROM bot_workflow"
    ).fetchone()
    assert row and row[0] == recs[0].wid


def test_code_db_updates_on_creation(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()
    code_db = dep.CodeDB(tmp_path / "code.db")
    err_db = eb.ErrorDB(tmp_path / "err.db")
    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        code_db=code_db,
    )
    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        error_bot=eb.ErrorBot(err_db, context_builder=developer.context_builder),
        context_builder=developer.context_builder,
    )

    task = bp.PlanningTask(
        description="do",
        complexity=1,
        frequency=1,
        expected_time=0.1,
        actions=["run"],
    )
    asyncio.run(bot.create_bots([task]))
    assert code_db.fetch_all()

    developer.errors.append("boom")
    asyncio.run(bot.create_bots([task], enhancements=[1]))
    bot_id = deployer.bot_db.fetch_all()[0]["id"]
    cids = code_db.codes_for_bot(bot_id)
    with code_db._connect() as conn:
        rows_enh = conn.execute(
            "SELECT enhancement_id FROM code_enhancements WHERE code_id=?",
            (cids[0],),
        ).fetchall()
        rows_err = conn.execute(
            "SELECT error_id FROM code_errors WHERE code_id=?",
            (cids[0],),
        ).fetchall()
    assert rows_enh
    assert rows_err


def test_creation_updates_status_and_enh_links(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()

    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    eid = enh_db.add(ceb.Enhancement(idea="i", rationale="r"))

    menace = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with menace.engine.begin() as conn:
        conn.execute(
            menace.models.insert().values(
                model_id=1,
                model_name="m",
                current_status="inactive",
            )
        )

    contr_db = cdb.ContrarianDB(tmp_path / "c.db")
    cid = contr_db.add(cdb.ContrarianRecord(innovation_name="x", innovation_type="y"))

    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        workflow_db=dep.WorkflowDB(tmp_path / "wf.db"),
        enh_db=enh_db,
        menace_db=menace,
        contrarian_db=contr_db,
    )

    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        context_builder=developer.context_builder,
    )

    task = bp.PlanningTask(
        description="do", complexity=1, frequency=1, expected_time=0.1, actions=["run"]
    )
    asyncio.run(
        bot.create_bots([task], model_id=1, enhancements=[eid], contrarian_id=cid)
    )

    assert enh_db.bots_for(eid)
    with menace.engine.connect() as conn:
        row = conn.execute(menace.models.select()).mappings().fetchone()
    assert row["current_status"] == "active"
    assert contr_db.fetch()[0].status == "active"


def test_higher_order_contrarian_timestamp(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()

    contr_db = cdb.ContrarianDB(tmp_path / "c.db")
    cid = contr_db.add(
        cdb.ContrarianRecord(
            innovation_name="x", innovation_type="y", timestamp_last_evaluated="old"
        )
    )

    wf_db = dep.WorkflowDB(tmp_path / "wf.db")
    wid = wf_db.add(dep.WorkflowRecord(workflow=["step"]))

    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        workflow_db=wf_db,
        contrarian_db=contr_db,
    )

    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        context_builder=developer.context_builder,
    )

    tasks = [
        bp.PlanningTask(
            description=f"t{i}",
            complexity=1,
            frequency=1,
            expected_time=0.1,
            actions=["run"],
        )
        for i in range(3)
    ]
    asyncio.run(bot.create_bots(tasks, workflows=[wid], contrarian_id=cid))

    assert contr_db.get(cid).timestamp_last_evaluated != "old"


def test_higher_order_contrarian_new_links(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()

    contr_db = cdb.ContrarianDB(tmp_path / "c.db")
    cid = contr_db.add(cdb.ContrarianRecord(innovation_name="n", innovation_type="y"))

    wf_db = dep.WorkflowDB(tmp_path / "wf.db")
    wid = wf_db.add(dep.WorkflowRecord(workflow=["step"]))

    enh_db = ceb.EnhancementDB(tmp_path / "e.db")
    eid = enh_db.add(ceb.Enhancement(idea="i", rationale="r"))

    err_db = eb.ErrorDB(tmp_path / "err.db")
    developer.errors.append("boom")

    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        workflow_db=wf_db,
        enh_db=enh_db,
        error_db=err_db,
        contrarian_db=contr_db,
    )

    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        error_bot=eb.ErrorBot(err_db, context_builder=developer.context_builder),
        context_builder=developer.context_builder,
    )

    tasks = [
        bp.PlanningTask(
            description=f"t{i}",
            complexity=1,
            frequency=1,
            expected_time=0.1,
            actions=["run"],
        )
        for i in range(3)
    ]
    asyncio.run(
        bot.create_bots(tasks, workflows=[wid], enhancements=[eid], contrarian_id=cid)
    )

    assert contr_db.workflows_for(cid)
    assert eid in contr_db.enhancements_for(cid)
    assert contr_db.errors_for(cid)
    assert contr_db.fetch()[0].status == "active"


def test_workflow_bot_links(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()

    menace = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with menace.engine.begin() as conn:
        conn.execute(
            menace.models.insert().values(model_id=1, model_name="m")
        )

    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        workflow_db=dep.WorkflowDB(tmp_path / "wf.db"),
        menace_db=menace,
    )

    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        context_builder=developer.context_builder,
    )

    task = bp.PlanningTask(
        description="do", complexity=1, frequency=1, expected_time=0.1, actions=["run"]
    )
    asyncio.run(bot.create_bots([task], model_id=1))

    with menace.engine.connect() as conn:
        rows = conn.execute(menace.workflow_bots.select()).fetchall()
    assert rows


def test_trending_order(tmp_path):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()
    deployer = dep.DeploymentBot(dep.DeploymentDB(tmp_path / "d.db"))
    deployer.build_containers = lambda bots, model_id=None: ([], [])

    class DummyScraper:
        def scrape_reddit(self, energy=None):
            return [
                ts.TrendingItem(platform="r", product_name="bot2"),
                ts.TrendingItem(platform="r", product_name="bot0"),
            ]

    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        trending_scraper=DummyScraper(),
        context_builder=developer.context_builder,
    )

    tasks = [
        bp.PlanningTask(
            description=f"t{i}",
            complexity=1,
            frequency=1,
            expected_time=0.1,
            actions=["run"],
        )
        for i in range(3)
    ]

    asyncio.run(bot.create_bots(tasks))

    rows = deployer.db.conn.execute(
        "SELECT name FROM deployments ORDER BY id"
    ).fetchall()
    names = [r[0] for r in rows]
    assert names[:2] == ["bot2", "bot0"]


class FailingEnhDB(ceb.EnhancementDB):
    def link_bot(self, enh, bot):
        raise RuntimeError("link fail")


def test_enhancement_link_error_logged(tmp_path, caplog):
    metrics = _metrics(tmp_path)
    planner = bp.BotPlanningBot()
    developer = bd.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
    tester = bt.BotTestingBot()
    enh_db = FailingEnhDB(tmp_path / "e.db")
    deployer = dep.DeploymentBot(
        dep.DeploymentDB(tmp_path / "d.db"),
        bot_db=dep.BotDB(tmp_path / "b.db"),
        enh_db=enh_db,
    )
    bot = bcb.BotCreationBot(
        metrics_db=metrics,
        planner=planner,
        developer=developer,
        tester=tester,
        deployer=deployer,
        error_bot=eb.ErrorBot(
            eb.ErrorDB(tmp_path / "err.db"),
            context_builder=developer.context_builder,
        ),
        context_builder=developer.context_builder,
    )

    task = bp.PlanningTask(
        description="do",
        complexity=1,
        frequency=1,
        expected_time=0.1,
        actions=["run"],
    )
    caplog.set_level("ERROR")
    asyncio.run(bot.create_bots([task]))
    assert "enhancement link failed" in caplog.text
