import subprocess
import sys
from pathlib import Path


def test_context_builder_static_analysis_runs():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_context_builder_usage.py"
    subprocess.run([sys.executable, str(script)])


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


def test_flags_missing_context_builder_with_memory(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "def demo(client):\n"
        "    client.build_prompt_with_memory(['tag'], 'hi')\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [(2, "build_prompt_with_memory")]


def test_flags_openai_calls(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "import openai\n"
        "def demo():\n"
        "    openai.ChatCompletion.create([])\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [(3, "openai.ChatCompletion.create")]


def test_flags_chat_completion_wrapper(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from billing.openai_wrapper import chat_completion_create\n"
        "def demo():\n"
        "    chat_completion_create([])\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [(3, "chat_completion_create")]


def test_allows_nocb_comment(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from billing.openai_wrapper import chat_completion_create\n"
        "def demo():\n"
        "    chat_completion_create([], model='x')  # nocb\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == []


def test_allows_context_builder_import(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = "from vector_service.context_builder import ContextBuilder\n"
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == []


def test_allows_context_builder_call(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo():\n"
        "    ContextBuilder(bots_db='bots.db', code_db='code.db', errors_db='errors.db', workflows_db='workflows.db')\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == []


def test_flags_getattr_context_builder(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "def demo(obj):\n"
        "    return getattr(obj, 'context_builder', None)\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [(2, "getattr context_builder default None")]


def test_allows_getattr_with_nocb(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "def demo(obj):\n"
        "    # nocb\n"
        "    return getattr(obj, 'context_builder', None)\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == []
