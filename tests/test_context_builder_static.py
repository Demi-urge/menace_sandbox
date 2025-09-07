import subprocess
import sys
from pathlib import Path


def test_context_builder_static_analysis():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_context_builder_usage.py"
    subprocess.run([sys.executable, str(script)], check=True)


def test_flags_missing_context_builder(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from menace_sandbox.chatgpt_idea_bot import build_prompt\n"
        "def demo(client):\n"
        "    build_prompt(client, ['tag'])\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [(3, "build_prompt")]
