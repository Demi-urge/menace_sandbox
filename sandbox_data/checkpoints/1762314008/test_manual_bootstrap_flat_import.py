from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _write_stub(module_path: Path, content: str) -> None:
    module_path.write_text(content, encoding="utf-8")


def test_manual_bootstrap_flat_import(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    project_root = repo_root.parent

    stubs_dir = tmp_path / "import_stubs"
    stubs_dir.mkdir()

    _write_stub(
        stubs_dir / "quick_fix_engine.py",
        "\n".join(
            [
                "class QuickFixEngine:",
                "    def __init__(self, *args, **kwargs):",
                "        self.args = args",
                "        self.kwargs = kwargs",
                "        self.patches = []",
                "",
                "    def apply(self, *args, **kwargs):",
                "        self.patches.append((args, kwargs))",
                "        return True",
                "",
                "",
                "class ErrorDB:",
                "    def __init__(self, *args, **kwargs):",
                "        self.rows = []",
            ]
        ),
    )

    _write_stub(
        stubs_dir / "self_coding_manager.py",
        "\n".join(
            [
                "from types import SimpleNamespace",
                "",
                "",
                "class SelfCodingManager:",
                "    def __init__(self, *args, **kwargs):",
                "        layer = SimpleNamespace(context_builder=SimpleNamespace())",
                "        self.engine = SimpleNamespace(cognition_layer=layer)",
                "        self.bot_registry = kwargs.get('bot_registry')",
                "        self.data_bot = kwargs.get('data_bot')",
                "        self.quick_fix = kwargs.get('quick_fix')",
                "",
                "",
                "def _manager_generate_helper_with_builder(*args, **kwargs):",
                "    return kwargs.get('code', '')",
            ]
        ),
    )

    env = os.environ.copy()
    pythonpath_parts = [str(stubs_dir), str(project_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    result = subprocess.run(
        [
            sys.executable,
            "manual_bootstrap.py",
            "--skip-environment",
            "--skip-sandbox",
        ],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "sandbox bootstrap completed successfully" in result.stdout
