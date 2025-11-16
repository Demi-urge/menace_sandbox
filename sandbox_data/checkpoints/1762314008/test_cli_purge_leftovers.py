import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_purge_stale(tmp_path):
    pkg = tmp_path / "pkg"
    sr = pkg / "sandbox_runner"
    sr.mkdir(parents=True)
    (sr / "__init__.py").write_text("")  # path-ignore
    cli_src = ROOT / "sandbox_runner" / "cli.py"  # path-ignore
    (sr / "cli.py").write_text(cli_src.read_text())  # path-ignore
    (sr / "environment.py").write_text(  # path-ignore
        """
import os
from pathlib import Path
SANDBOX_ENV_PRESETS = [{}]

def load_presets():
    return SANDBOX_ENV_PRESETS

def simulate_full_environment(*a, **k):
    pass

def purge_leftovers():
    Path(os.environ['OUT_FILE']).write_text('called')
"""
    )
    # menace stubs
    mn = pkg / "menace"
    mn.mkdir()
    (mn / "__init__.py").write_text("")  # path-ignore
    (mn / "metrics_dashboard.py").write_text(  # path-ignore
        """
class MetricsDashboard:
    def __init__(self, *a, **k):
        pass
"""
    )
    (mn / "environment_generator.py").write_text(  # path-ignore
        """
def generate_presets(n=None):
    return [{}]
"""
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{pkg}:{ROOT}:{env.get('PYTHONPATH', '')}"
    out_file = tmp_path / "out.txt"
    env["OUT_FILE"] = str(out_file)

    script = f"""
import sys, os
sys.path.insert(0, '{pkg}')
import sandbox_runner.cli as cli
cli.main(['--purge-stale'])
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_file.read_text() == "called"
