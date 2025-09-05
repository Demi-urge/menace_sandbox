from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_autonomous_uses_dynamic_setup_marker():
    src = (REPO_ROOT / "run_autonomous.py").read_text()  # path-ignore
    assert 'resolve_path(".autonomous_setup_complete")' in src
    assert 'Path(".autonomous_setup_complete")' not in src


def test_self_coding_modules_have_no_hardcoded_paths():
    pattern = re.compile(r'Path\(\"|\"[^"\n]+/[^"\n]+\.py\"')  # path-ignore
    for name in ["self_coding_manager.py", "quick_fix_engine.py"]:  # path-ignore
        text = (REPO_ROOT / name).read_text()
        matches = pattern.findall(text)
        assert not matches, f"{name} contains hard-coded paths: {matches}"
