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
    _write(before, "a.py", "def f():\n    return 1\n")  # path-ignore
    _write(after, "a.py", "def f():\n    reward = 1\n    return reward\n")  # path-ignore
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("reward" in f for f in flags)


def test_semantic_detection(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "b.py", "x = 1\n")  # path-ignore
    _write(after, "b.py", "x = 1\n\neval('2')\n")  # path-ignore
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("eval" in f.lower() for f in flags)


def test_exec_detection(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "c.py", "x = 1\n")  # path-ignore
    _write(after, "c.py", "x = 1\n\nexec('print(1)')\n")  # path-ignore
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("exec" in f.lower() for f in flags)


def test_weak_hash_detection(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "d.py", "x = 1\n")  # path-ignore
    _write(after, "d.py", "import hashlib\nhashlib.md5(b'd').hexdigest()\n")  # path-ignore
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("md5" in f.lower() for f in flags)


def test_network_call_detection(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "e.py", "x = 1\n")  # path-ignore
    _write(after, "e.py", "import requests\nrequests.get('http://a')\n")  # path-ignore
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("network" in f.lower() for f in flags)


def test_semantic_filter_flags_comment(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "f.py", "x = 1\n")  # path-ignore
    _write(after, "f.py", "x = 1\n# eval data\n")  # path-ignore
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert any("use of eval" in f for f in flags)


def test_semantic_filter_benign(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _write(before, "g.py", "x = 1\n")  # path-ignore
    _write(after, "g.py", "x = 1\n# just a comment\n")  # path-ignore
    diff = generate_code_diff(str(before), str(after))
    flags = flag_risky_changes(diff)
    assert flags == []
