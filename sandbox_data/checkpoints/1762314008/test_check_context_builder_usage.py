from __future__ import annotations

from pathlib import Path
import textwrap

from scripts import check_context_builder_usage as ccb


def _write_source(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


def _extract_messages(errors):
    return [message for _, message in errors]


def test_prompt_instantiation_flagged(tmp_path):
    bad_file = _write_source(tmp_path, "direct_prompt.py", """
    Prompt('raw prompt text')
    """)

    messages = _extract_messages(ccb.check_file(bad_file))

    assert any("Prompt instantiation disallowed" in message for message in messages)


def test_prompt_engine_build_prompt_flagged(tmp_path):
    bad_file = _write_source(tmp_path, "prompt_engine.py", """
    PromptEngine.build_prompt('raw text')
    """)

    messages = _extract_messages(ccb.check_file(bad_file))

    assert any("PromptEngine.build_prompt disallowed" in message for message in messages)


def test_prompt_alias_instantiation_flagged(tmp_path):
    bad_file = _write_source(
        tmp_path,
        "chatgpt_idea_bot.py",
        """
        from prompt_types import Prompt as IdeaPrompt

        IdeaPrompt('raw prompt text')
        """,
    )

    messages = _extract_messages(ccb.check_file(bad_file))

    assert any("Prompt instantiation disallowed" in message for message in messages)


def test_prompt_assignment_alias_flagged(tmp_path):
    bad_file = _write_source(
        tmp_path,
        "alias_assignment.py",
        """
        from prompt_types import Prompt

        IdeaPrompt = Prompt
        IdeaPrompt('raw prompt text')
        """,
    )

    messages = _extract_messages(ccb.check_file(bad_file))

    assert any("Prompt instantiation disallowed" in message for message in messages)


def test_main_returns_failure_on_banned_usage(tmp_path, monkeypatch):
    bad_file = _write_source(tmp_path, "banned_usage.py", """
    Prompt('raw prompt text')
    """)

    monkeypatch.setattr(ccb, "iter_python_files", lambda _root: [bad_file])

    assert ccb.main() == 1


def test_chatgpt_idea_bot_is_clean():
    messages = _extract_messages(ccb.check_file(Path("chatgpt_idea_bot.py")))

    assert not any("Prompt instantiation disallowed" in message for message in messages)
