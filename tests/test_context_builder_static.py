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
        "def demo(x):\n"
        "    openai.ChatCompletion.create(x)\n"
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
        "def demo(x):\n"
        "    chat_completion_create(x)\n"
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


def test_flags_manual_string_prompt(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo(llm):\n"
        "    builder = ContextBuilder('bots.db', 'code.db', 'errors.db', 'workflows.db')\n"
        "    llm.generate('hi', context_builder=builder)\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (
            4,
            "manual string prompt disallowed; use context_builder.build_prompt or SelfCodingEngine.build_enriched_prompt",
        )
    ]


def test_flags_string_concatenation_prompt(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from vector_service.context_builder import ContextBuilder\n"
        "def demo(llm):\n"
        "    builder = ContextBuilder('bots.db', 'code.db', 'errors.db', 'workflows.db')\n"
        "    llm.generate('hi' + ' there', context_builder=builder)\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (
            4,
            "manual string prompt disallowed; use context_builder.build_prompt or SelfCodingEngine.build_enriched_prompt",
        )
    ]


def test_flags_prompt_engine_build_prompt(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "from prompt_engine import PromptEngine\n"
        "def demo():\n"
        "    PromptEngine.build_prompt('x')\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (3, "PromptEngine.build_prompt disallowed; use ContextBuilder.build_prompt"),
    ]


def test_flags_context_builder_build_concatenation(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "def demo(llm, context_builder):\n"
        "    llm.generate(context_builder.build('x') + 'y', context_builder=context_builder)\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (2, "context_builder.build result concatenation disallowed; pass directly")
    ]


def test_flags_context_builder_build_list_concatenation(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "def demo(llm, context_builder):\n"
        "    llm.generate(context_builder.build('x') + ['y'], context_builder=context_builder)\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (2, "context_builder.build result concatenation disallowed; pass directly")
    ]


def test_flags_ask_with_memory(tmp_path):
    from scripts.check_context_builder_usage import check_file

    code = (
        "def demo():\n"
        "    ask_with_memory('c', 'k', 'p')\n"
    )
    path = tmp_path / "snippet.py"
    path.write_text(code)
    assert check_file(path) == [
        (2, "ask_with_memory disallowed; use ContextBuilder.build_prompt")
    ]
