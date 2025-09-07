import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from pathlib import Path

from db_router import GLOBAL_ROUTER, init_db_router
import menace.ipo_implementation_pipeline as ipp
import menace.ipo_bot as ipb
import menace.bot_development_bot as bdb
import menace.bot_testing_bot as btb
import menace.scalability_assessment_bot as sab
import menace.deployment_bot as dep


def _make_db(path: Path) -> Path:
    init_db_router(
        "ipo_pipeline_test", local_db_path=str(path.parent / "local.db"), shared_db_path=str(path)
    )
    conn = GLOBAL_ROUTER.get_connection("bots")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS bots (id INTEGER PRIMARY KEY, name TEXT, keywords TEXT, reuse INTEGER)"
    )
    conn.commit()
    return path


def _ctx_builder():
    return bdb.ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")


def test_full_pipeline(tmp_path: Path):
    db_path = _make_db(tmp_path / "models.db")
    ipo = ipb.IPOBot(db_path=str(db_path), enhancements_db=tmp_path / "enh.db")
    builder = _ctx_builder()
    developer = bdb.BotDevelopmentBot(
        repo_base=tmp_path / "repos", context_builder=builder
    )
    tester = btb.BotTestingBot()
    scaler = sab.ScalabilityAssessmentBot()
    deployer = dep.DeploymentBot(dep.DeploymentDB(tmp_path / "dep.db"))
    pipeline = ipp.IPOImplementationPipeline(
        ipo=ipo,
        developer=developer,
        tester=tester,
        scaler=scaler,
        deployer=deployer,
        context_builder=builder,
    )
    results = pipeline.run("BotA does things")
    assert results
    assert results[0].deployed
