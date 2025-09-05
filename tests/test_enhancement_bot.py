import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import menace.enhancement_bot as eb
import menace.chatgpt_enhancement_bot as ceb


def test_evaluate_and_log(tmp_path):
    f = tmp_path / "a.py"  # path-ignore
    f.write_text("def run(x, y):\n    total = 0\n    for _ in range(1000):\n        total += x + y\n    return total\n")
    new_code = "def run(x, y):\n    return (x + y) * 1000\n"
    db = ceb.EnhancementDB(tmp_path / "e.db")
    bot = eb.EnhancementBot(enhancement_db=db)
    prop = eb.RefactorProposal(file_path=f, new_code=new_code, author_bot="codex")
    ok = bot.evaluate(prop)
    assert ok
    hist = db.history()
    assert hist and hist[0].file_path == str(f)
