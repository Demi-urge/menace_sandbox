import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.scalability_pipeline as sp
import menace.bot_development_bot as bdb
import menace.bot_testing_bot as btb
import menace.scalability_assessment_bot as sab
import menace.deployment_bot as dep


def test_pipeline_runs(tmp_path):
    developer = bdb.BotDevelopmentBot(repo_base=tmp_path / "repos")
    tester = btb.BotTestingBot()
    scaler = sab.ScalabilityAssessmentBot()
    deployer = dep.DeploymentBot(dep.DeploymentDB(tmp_path / "dep.db"))
    pipeline = sp.ScalabilityPipeline(
        developer=developer,
        tester=tester,
        scaler=scaler,
        deployer=deployer,
    )
    dep_id = pipeline.run(["BotX"]).deploy_id
    assert deployer.db.get(dep_id)["status"] == "success"
