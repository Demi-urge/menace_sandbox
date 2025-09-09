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
    assert check_file(path) == [
        (3, "build_prompt disallowed or missing context_builder")
    ]


def test_flags_missing_context_builder_with_memory(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "def demo(client):\n"
        "    client.build_prompt_with_memory(['tag'], 'hi')\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (2, "build_prompt_with_memory disallowed or missing context_builder")
    ]


def test_flags_openai_calls(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "import openai\n"
        "def demo():\n"
        "    openai.ChatCompletion.create([])\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (3, "openai.ChatCompletion.create disallowed or missing context_builder")
    ]


def test_flags_chat_completion_wrapper(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from billing.openai_wrapper import chat_completion_create\n"
        "def demo():\n"
        "    chat_completion_create([])\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (3, "chat_completion_create disallowed or missing context_builder")
    ]


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


def test_flags_context_builder_in_function(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo():\n"
        "    ContextBuilder()\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [(
        3,
        "ContextBuilder() missing bots.db, code.db, errors.db, workflows.db",
    )]


def test_allows_context_builder_with_nocb(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo():\n"
        "    ContextBuilder()  # nocb\n"
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
    assert check_file(path) == [
        (2, "getattr context_builder default None disallowed or missing context_builder")
    ]


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


def test_flags_build_with_optional_builder(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo(builder=None):\n"
        "    builder.build()\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (3, "builder.build disallowed or missing context_builder")
    ]


def test_flags_build_with_fallback_builder(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo(builder):\n"
        "    builder = builder or ContextBuilder('bots.db', 'code.db', 'errors.db', 'workflows.db')\n"
        "    builder.build()\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (4, "builder.build disallowed or missing context_builder")
    ]


def test_allows_build_with_required_builder(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo(builder):\n"
        "    builder.build()\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == []
