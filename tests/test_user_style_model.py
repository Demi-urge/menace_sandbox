import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.user_style_model as usm
import menace.mirror_bot as mb


def test_train_and_generate(tmp_path):
    db = mb.MirrorDB(tmp_path / "m.db")
    bot = mb.MirrorBot(db)
    bot.log_interaction("hi", "hello", feedback="good")
    model = usm.UserStyleModel(db)
    try:
        model.train()
    except RuntimeError:
        pytest.skip("transformers not available")
    text = model.generate("test")
    assert isinstance(text, str)
