import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.deployment_bot as db


def test_obsolete_status(tmp_path):
    bot = db.DeploymentBot(
        bot_db=db.BotDB(tmp_path / "bots.db"),
        workflow_db=db.WorkflowDB(tmp_path / "wf.db"),
    )

    bot._update_bot_records(
        ["botA-v1"],
        model_id=None,
        workflows=[],
        enhancements=[],
        resources={},
        levels={},
        errors=None,
    )
    rec = bot.bot_db.find_by_name("botA-v1")
    first_id = rec["id"]
    bot.bot_db.conn.execute(
        "UPDATE bots SET last_modification_date=? WHERE id=?",
        ("2000-01-01T00:00:00", first_id),
    )
    bot.bot_db.conn.commit()

    bot._update_bot_records(
        ["botA-v1a"],
        model_id=None,
        workflows=[],
        enhancements=[],
        resources={},
        levels={},
        errors=None,
    )

    old_rec = bot.bot_db.find_by_name("botA-v1")
    new_rec = bot.bot_db.find_by_name("botA-v1a")
    assert old_rec["status"] == "obsolete"
    assert old_rec["last_modification_date"] != "2000-01-01T00:00:00"
    assert new_rec["status"] == "active"


def test_obsolete_status_reverse(tmp_path):
    bot = db.DeploymentBot(
        bot_db=db.BotDB(tmp_path / "bots.db"),
        workflow_db=db.WorkflowDB(tmp_path / "wf.db"),
    )

    bot._update_bot_records(
        ["botA-v2"],
        model_id=None,
        workflows=[],
        enhancements=[],
        resources={},
        levels={},
        errors=None,
    )
    rec = bot.bot_db.find_by_name("botA-v2")
    first_id = rec["id"]
    bot.bot_db.conn.execute(
        "UPDATE bots SET last_modification_date=? WHERE id=?",
        ("2099-01-01T00:00:00", first_id),
    )
    bot.bot_db.conn.commit()

    bot._update_bot_records(
        ["botA-v1"],
        model_id=None,
        workflows=[],
        enhancements=[],
        resources={},
        levels={},
        errors=None,
    )

    old_rec = bot.bot_db.find_by_name("botA-v2")
    new_rec = bot.bot_db.find_by_name("botA-v1")
    assert new_rec["status"] == "obsolete"
    assert old_rec["status"] == "active"
