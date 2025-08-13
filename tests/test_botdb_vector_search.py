from types import MethodType

from menace.bot_database import BotDB, BotRecord


def test_vector_search(tmp_path):
    db = BotDB(
        path=tmp_path / "b.db",
        vector_backend="annoy",
        vector_index_path=tmp_path / "b.index",
    )

    captured: list[str] = []

    def fake_embed(self, text: str):
        captured.append(text)
        if "alpha" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    rec1 = BotRecord(name="A", purpose="alpha", tags=["x"], toolchain="tc1")
    rec2 = BotRecord(name="B", purpose="beta", tags=["y"], toolchain="tc2")

    db.add_bot(rec1)
    assert captured and "alpha" in captured[0] and "x" in captured[0] and "tc1" in captured[0]

    db.add_bot(rec2)
    assert len(captured) == 2

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0]["name"] == "A"
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0]["name"] == "B"
