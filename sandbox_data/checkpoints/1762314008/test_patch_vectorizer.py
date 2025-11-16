import importlib
import sys
from pathlib import Path
import types

from db_router import DBRouter

# Ensure relative imports resolve and stub heavy dependencies
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

# Provide a minimal ``code_database`` with ``PatchHistoryDB`` for the vectorizer
code_db_mod = types.ModuleType("code_database")


class PatchHistoryDB:  # noqa: D401 - minimal stub
    """Simple stand-in storing path and returning SQLite connection."""

    def __init__(self, path):
        self.path = Path(path)
        self.router = DBRouter("patch_history", str(self.path), str(self.path))

    def get(self, pid):
        conn = self.router.get_connection("patch_history")
        row = conn.execute(
            "SELECT description, diff, summary FROM patch_history WHERE id=?",
            (pid,),
        ).fetchone()
        if row:
            return types.SimpleNamespace(
                description=row[0], diff=row[1], summary=row[2]
            )
        return None


code_db_mod.PatchHistoryDB = PatchHistoryDB
sys.modules.setdefault("code_database", code_db_mod)
sys.modules.setdefault("menace_sandbox.code_database", code_db_mod)

# Stub ``data_bot`` with minimal ``MetricsDB`` implementation
data_bot_mod = types.ModuleType("data_bot")


class MetricsDB:  # noqa: D401 - simple stub
    """Stub metrics DB used for staleness logging."""

    def log_embedding_staleness(self, origin, rid, age):  # pragma: no cover - stub
        pass


data_bot_mod.MetricsDB = MetricsDB
sys.modules.setdefault("data_bot", data_bot_mod)
sys.modules.setdefault("menace_sandbox.data_bot", data_bot_mod)

# Finally, expose ``embeddable_db_mixin`` for direct import by vectorizers
sys.modules.setdefault(
    "embeddable_db_mixin",  # noqa: E402
    importlib.import_module("menace_sandbox.embeddable_db_mixin"),
)

from vector_service.patch_vectorizer import (  # noqa: E402
    PatchVectorizer,
    backfill_patch_embeddings,
)
from code_database import PatchHistoryDB  # noqa: E402


def test_patch_vectorizer_embeds_diff_and_summary(tmp_path, monkeypatch):
    db_path = tmp_path / "patch_history.db"
    phdb = PatchHistoryDB(db_path)
    conn = phdb.router.get_connection("patch_history")
    conn.execute(
        "CREATE TABLE patch_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "description TEXT, diff TEXT, summary TEXT)"
    )
    conn.execute(
        "INSERT INTO patch_history (description, diff, summary) VALUES (?, ?, ?)",
        ("desc text", "diff text", "summary text"),
    )
    conn.commit()

    pv = PatchVectorizer(path=db_path, index_path=tmp_path / "patch.index")

    captured = {}

    def fake_encode(self, text: str):
        captured["text"] = text
        return [float(len(text))]

    monkeypatch.setattr(PatchVectorizer, "encode_text", fake_encode, raising=False)

    pv.backfill_embeddings()

    assert "diff text" in captured["text"]
    assert "summary text" in captured["text"]

    meta = pv._metadata.get("1")
    assert meta and meta["record"]["diff"] == "diff text"
    assert meta["record"]["summary"] == "summary text"

    expected = "\n".join(["desc text", "diff text", "summary text"])
    assert pv.get_vector(1) == [float(len(expected))]


def test_backfill_helper(tmp_path, monkeypatch):
    db_path = tmp_path / "patch_history.db"
    phdb = PatchHistoryDB(db_path)
    conn = phdb.router.get_connection("patch_history")
    conn.execute(
        "CREATE TABLE patch_history (id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT, diff TEXT, summary TEXT)"
    )
    conn.execute(
        "INSERT INTO patch_history (description, diff, summary) VALUES (?, ?, ?)",
        ("desc text", "diff text", "summary text"),
    )
    conn.commit()

    captured = {}

    def fake_encode(self, text: str):
        captured["text"] = text
        return [float(len(text))]

    monkeypatch.setattr(PatchVectorizer, "encode_text", fake_encode, raising=False)

    backfill_patch_embeddings(path=db_path, index_path=tmp_path / "patch.index")

    pv = PatchVectorizer(path=db_path, index_path=tmp_path / "patch.index")
    expected = "\n".join(["desc text", "diff text", "summary text"])
    assert pv.get_vector(1) == [float(len(expected))]
