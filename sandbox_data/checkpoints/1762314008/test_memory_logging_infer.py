from menace_sandbox.gpt_memory import GPTMemoryManager
from memory_logging import log_with_tags


def test_infers_tag_from_response():
    mem = GPTMemoryManager(":memory:")
    log_with_tags(mem, "p", "Bug fix applied")
    row = mem.conn.execute("SELECT tags FROM interactions").fetchone()[0]
    assert row == "error_fix"


def test_ambiguous_text_rejects_tags():
    mem = GPTMemoryManager(":memory:")
    # Contains keywords for both error_fix and improvement_path
    log_with_tags(mem, "p", "Fix applied to improve performance")
    row = mem.conn.execute("SELECT tags FROM interactions").fetchone()[0]
    assert row == ""
