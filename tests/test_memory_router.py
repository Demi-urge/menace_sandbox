from db_router import DBRouter
from menace_sandbox.gpt_memory import GPTMemoryManager


def test_gpt_memory_uses_router(tmp_path):
    router = DBRouter("test", tmp_path / "local.db", tmp_path / "shared.db")
    mgr = GPTMemoryManager(router=router)
    mgr.log_interaction("p", "r")
    # Ensure the connection comes from router
    assert mgr.conn is router.get_connection("memory")
    with router.get_connection("memory") as conn:
        rows = conn.execute("SELECT prompt, response FROM interactions").fetchall()
    assert rows == [("p", "r")]
