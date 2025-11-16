import menace.mirror_bot as mb


def test_log_and_fetch(tmp_path):
    db = mb.MirrorDB(tmp_path / "m.db")
    bot = mb.MirrorBot(db)
    bot.log_interaction("hello", "hi there", feedback="great response")
    records = bot.history()
    assert len(records) == 1
    assert records[0].sentiment > 0


def test_style_and_bias(tmp_path):
    db = mb.MirrorDB(tmp_path / "m.db")
    bot = mb.MirrorBot(db)
    bot.update_style("be concise")
    bot.log_interaction("hi", "ok", feedback="good")
    resp = bot.generate_response("hello")
    assert "be concise" in resp
    assert resp.endswith(":)")
