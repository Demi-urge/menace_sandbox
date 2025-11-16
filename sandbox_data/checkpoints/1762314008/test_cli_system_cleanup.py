import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_system_cleanup(tmp_path):
    pkg = tmp_path / "pkg"
    sr = pkg / "sandbox_runner"
    sr.mkdir(parents=True)
    (sr / "__init__.py").write_text("")  # path-ignore
    cli_src = ROOT / "sandbox_runner" / "cli.py"  # path-ignore
    (sr / "cli.py").write_text(cli_src.read_text())  # path-ignore
    (sr / "environment.py").write_text(  # path-ignore
        """
import os
import json
import shutil
from pathlib import Path

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


def purge_leftovers():
    with open(os.environ['OUT_FILE'], 'a') as fh:
        fh.write('purge\\n')
    cfile = Path(os.environ['CONTAINER_FILE'])
    if cfile.exists():
        cfile.write_text('[]')
    odir = Path(os.environ['OVERLAY_DIR'])
    if odir.exists():
        shutil.rmtree(odir)


def retry_failed_cleanup(progress=None):
    with open(os.environ['OUT_FILE'], 'a') as fh:
        fh.write('retry\\n')
"""
    )
    # menace stubs
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
    containers = tmp_path / "containers.json"
    containers.write_text(json.dumps(["c1", "c2"]))
    env["CONTAINER_FILE"] = str(containers)
    overlay_dir = tmp_path / "overlay"
    overlay_dir.mkdir()
    (overlay_dir / "overlay.qcow2").touch()
    env["OVERLAY_DIR"] = str(overlay_dir)

    script = (
        f"import sys, os; sys.path.insert(0, '{pkg}'); "
        f"import sandbox_runner.cli as cli; cli.main(['cleanup'])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert out_file.read_text().splitlines() == ["purge", "retry"]
    assert containers.read_text() == "[]"
    assert not overlay_dir.exists()

