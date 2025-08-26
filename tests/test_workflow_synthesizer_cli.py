import json
import os
import pty
import select
import shutil
import subprocess
import sys
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "workflow_modules"


def _prepare(tmp_path: Path) -> None:
    for mod in FIXTURES.glob("*.py"):
        shutil.copy(mod, tmp_path / mod.name)
    graph = nx.DiGraph()
    graph.add_edge("mod_a", "mod_b", weight=1.0)
    graph.add_edge("mod_a", "mod_c", weight=1.0)
    sd = tmp_path / "sandbox_data"
    sd.mkdir()
    path = sd / "module_synergy_graph.json"
    path.write_text(json.dumps(json_graph.node_link_data(graph)), encoding="utf-8")


def _run_tty(cmd, cwd: Path, text: str) -> tuple[int, str]:
    master, slave = pty.openpty()
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=slave, stdout=slave, stderr=slave, text=True)
    os.close(slave)
    os.write(master, text.encode())
    output = ""
    while True:
        r, _, _ = select.select([master], [], [], 0.1)
        if r:
            try:
                data = os.read(master, 1024).decode()
            except OSError:
                break
            if not data:
                break
            output += data
        if proc.poll() is not None and not r:
            break
    proc.wait()
    os.close(master)
    return proc.returncode, output


def test_cli_interactive_selection(tmp_path: Path):
    _prepare(tmp_path)
    out = tmp_path / "wf.workflow.json"
    cmd = [sys.executable, str(ROOT / "workflow_synthesizer_cli.py"), "mod_a", "--limit", "3", "--out", str(out)]
    rc, output = _run_tty(cmd, tmp_path, "2\n")
    assert rc == 0
    saved = tmp_path / "workflows" / out.name
    data = json.loads(saved.read_text())
    modules = [s["module"] for s in data["steps"]]
    assert modules == ["mod_a", "mod_c"]
    assert "Select workflow" in output
    assert "score=" in output


def test_cli_select_flag(tmp_path: Path):
    _prepare(tmp_path)
    out = tmp_path / "wf.workflow.json"
    cmd = [
        sys.executable,
        str(ROOT / "workflow_synthesizer_cli.py"),
        "mod_a",
        "--limit",
        "3",
        "--out",
        str(out),
        "--select",
        "2",
    ]
    proc = subprocess.run(cmd, cwd=tmp_path, text=True, capture_output=True)
    assert proc.returncode == 0
    saved = tmp_path / "workflows" / out.name
    data = json.loads(saved.read_text())
    modules = [s["module"] for s in data["steps"]]
    assert modules == ["mod_a", "mod_c"]
    assert "Select workflow" not in proc.stdout
