# flake8: noqa
import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.scalability_pipeline as sp  # noqa: E402
import menace.bot_development_bot as bdb  # noqa: E402
import menace.bot_testing_bot as btb  # noqa: E402
import menace.scalability_assessment_bot as sab  # noqa: E402
import menace.deployment_bot as dep  # noqa: E402


def _ctx_builder():
    return bdb.ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")


def test_pipeline_runs(tmp_path):
    developer = bdb.BotDevelopmentBot(repo_base=tmp_path / "repos", context_builder=_ctx_builder())
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
