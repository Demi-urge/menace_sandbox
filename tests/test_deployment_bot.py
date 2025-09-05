import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.deployment_bot as db


def test_deploy(tmp_path):
    p = tmp_path / "a.py"  # path-ignore
    p.write_text("def run():\n    pass\n")
    import shutil, os
    shutil.copy(p, "a.py")  # path-ignore
    bot = db.DeploymentBot(
        db.DeploymentDB(tmp_path / "dep.db"),
        code_db=db.CodeDB(tmp_path / "code.db"),
    )
    spec = db.DeploymentSpec(name="test", resources={}, env={})
    dep_id = bot.deploy("run", ["a"], spec)
    rec = bot.db.get(dep_id)
    os.remove("a.py")  # path-ignore
    assert rec["status"] == "success"
    assert bot.code_db.fetch_all()


def test_rollback(tmp_path):
    bot = db.DeploymentBot(db.DeploymentDB(tmp_path / "d.db"))
    dep_id = bot.db.add("test", "failed", "{}")
    bot.rollback(dep_id)
    row = bot.db.conn.execute(
        "SELECT COUNT(*) FROM errors WHERE deploy_id = ?", (dep_id,)
    ).fetchone()
    assert row[0] == 1
