import textwrap
from pathlib import Path

from codebase_diff_checker import generate_code_diff, flag_risky_changes


def _write(tmp, path, content):
    p = tmp / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


def test_keyword_detection(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "a.py", "def f():\n    return 1\n")
    _write(after, "a.py", "def f():\n    reward = 1\n    return reward\n")
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("reward" in f for f in flags)


def test_semantic_detection(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "b.py", "x = 1\n")
    _write(after, "b.py", "x = 1\n\neval('2')\n")
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("eval" in f.lower() for f in flags)
