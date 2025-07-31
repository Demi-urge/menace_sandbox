import json
from pathlib import Path

from scripts.find_orphan_modules import find_orphan_modules, main


def test_detects_orphan_module(tmp_path, monkeypatch):
    (tmp_path / "sandbox_data").mkdir()

    # create modules
    (tmp_path / "foo.py").write_text("x = 1\n")
    (tmp_path / "bar.py").write_text("y = 2\n")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_foo.py").write_text("import foo\n")

    # run detection function
    orphans = find_orphan_modules(tmp_path)
    assert Path("bar.py") in orphans
    assert Path("foo.py") not in orphans

    # run CLI and verify output file
    monkeypatch.chdir(tmp_path)
    main([])
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert str(Path("bar.py")) in data
    assert str(Path("foo.py")) not in data


def test_recursive_option(tmp_path):
    (tmp_path / "sandbox_data").mkdir()

    (tmp_path / "foo.py").write_text("x = 1\n")
    (tmp_path / "bar.py").write_text("y = 2\n")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_foo.py").write_text("import foo\n")

    non_rec = find_orphan_modules(tmp_path)
    rec = find_orphan_modules(tmp_path, recursive=True)
    assert non_rec == rec

    # CLI invocation
    import subprocess, sys

    subprocess.check_call([
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "find_orphan_modules.py"),
        "--recursive",
    ], cwd=tmp_path)
    data = json.loads((tmp_path / "sandbox_data" / "orphan_modules.json").read_text())
    assert str(Path("bar.py")) in data
