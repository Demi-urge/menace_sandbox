import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_check_resources(tmp_path):
    pkg = tmp_path / "pkg"
    sr = pkg / "sandbox_runner"
    sr.mkdir(parents=True)
    (sr / "__init__.py").write_text("")  # path-ignore
    cli_src = ROOT / "sandbox_runner" / "cli.py"  # path-ignore
    (sr / "cli.py").write_text(cli_src.read_text())  # path-ignore
    (sr / "environment.py").write_text(  # path-ignore
        """
import os

SANDBOX_ENV_PRESETS = [{}]

def load_presets():
    return SANDBOX_ENV_PRESETS

def simulate_full_environment(*a, **k):
    pass

class DummyLock:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc, tb):
        pass

_PURGE_FILE_LOCK = DummyLock()
_STALE_CONTAINERS_REMOVED = 0
_STALE_VMS_REMOVED = 0


def purge_leftovers():
    global _STALE_CONTAINERS_REMOVED, _STALE_VMS_REMOVED
    with open(os.environ['OUT_FILE'], 'a') as fh:
        fh.write('purge\\n')
    _STALE_CONTAINERS_REMOVED += 2
    _STALE_VMS_REMOVED += 1


def retry_failed_cleanup(progress=None):
    with open(os.environ['OUT_FILE'], 'a') as fh:
        fh.write('retry\\n')
"""
    )

    mn = pkg / "menace"
    mn.mkdir()
    (mn / "__init__.py").write_text("")  # path-ignore
    (mn / "metrics_dashboard.py").write_text("class MetricsDashboard: pass")  # path-ignore
    (mn / "environment_generator.py").write_text(  # path-ignore
        "def generate_presets(n=None):\n    return [{}]\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{pkg}:{ROOT}:{env.get('PYTHONPATH', '')}"
    out_file = tmp_path / "out.txt"
    env["OUT_FILE"] = str(out_file)

    script = (
        f"import sys, os; sys.path.insert(0, '{pkg}'); "
        f"import sandbox_runner.cli as cli; cli.main(['check-resources'])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'Removed 2 containers and 1 overlays'
    assert out_file.read_text().splitlines() == ['purge', 'retry']
