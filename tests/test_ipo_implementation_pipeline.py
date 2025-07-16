import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import sqlite3
from pathlib import Path

import menace.ipo_implementation_pipeline as ipp
import menace.ipo_bot as ipb
import menace.bot_development_bot as bdb
import menace.bot_testing_bot as btb
import menace.scalability_assessment_bot as sab
import menace.deployment_bot as dep


def _make_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE bots (id INTEGER PRIMARY KEY, name TEXT, keywords TEXT, reuse INTEGER)"
    )
    conn.commit()
    conn.close()
    return path


def test_full_pipeline(tmp_path: Path):
    db_path = _make_db(tmp_path / "models.db")
    ipo = ipb.IPOBot(db_path=str(db_path), enhancements_db=tmp_path / "enh.db")
    developer = bdb.BotDevelopmentBot(repo_base=tmp_path / "repos")
    tester = btb.BotTestingBot()
    scaler = sab.ScalabilityAssessmentBot()
    deployer = dep.DeploymentBot(dep.DeploymentDB(tmp_path / "dep.db"))
    pipeline = ipp.IPOImplementationPipeline(
        ipo=ipo,
        developer=developer,
        tester=tester,
        scaler=scaler,
        deployer=deployer,
    )
    results = pipeline.run("BotA does things")
    assert results
    assert results[0].deployed
