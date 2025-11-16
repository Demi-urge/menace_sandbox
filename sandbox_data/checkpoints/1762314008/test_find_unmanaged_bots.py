"""Tests for tools/find_unmanaged_bots.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _script_path(root: Path) -> Path:
    return root / "tools" / "find_unmanaged_bots.py"


def test_repo_has_no_unmanaged_bots() -> None:
    """The repository should contain only managed bots."""
    root = Path(__file__).resolve().parent.parent
    script = _script_path(root)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_detects_unmanaged_bot(tmp_path: Path) -> None:
    """The script returns a non-zero exit code when unmanaged bots exist."""
    # Prepare temporary repo with script
    root = Path(__file__).resolve().parent.parent
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    shutil.copy2(_script_path(root), tools_dir / "find_unmanaged_bots.py")

    # Create unmanaged bot
    bot_file = tmp_path / "rogue_bot.py"
    bot_file.write_text("class RogueBot:\n    pass\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(tools_dir / "find_unmanaged_bots.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "rogue_bot.py" in result.stdout
