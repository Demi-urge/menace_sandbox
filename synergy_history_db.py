import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS synergy_history ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " entry TEXT NOT NULL)"
)


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute(CREATE_TABLE)
    conn.commit()
    return conn


def insert_entry(conn: sqlite3.Connection, entry: Dict[str, float]) -> None:
    conn.execute("INSERT INTO synergy_history(entry) VALUES (?)", (json.dumps(entry),))
    conn.commit()


def fetch_all(conn: sqlite3.Connection) -> List[Dict[str, float]]:
    rows = conn.execute("SELECT entry FROM synergy_history ORDER BY id").fetchall()
    out: List[Dict[str, float]] = []
    for (text,) in rows:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                out.append({str(k): float(v) for k, v in data.items()})
        except Exception:
            continue
    return out


def fetch_after(conn: sqlite3.Connection, last_id: int) -> List[Tuple[int, Dict[str, float]]]:
    """Return entries with ``id`` greater than ``last_id``."""
    rows = conn.execute(
        "SELECT id, entry FROM synergy_history WHERE id > ? ORDER BY id",
        (int(last_id),),
    ).fetchall()
    out: List[Tuple[int, Dict[str, float]]] = []
    for row_id, text in rows:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                out.append((int(row_id), {str(k): float(v) for k, v in data.items()}))
        except Exception:
            continue
    return out


def fetch_latest(conn: sqlite3.Connection) -> Dict[str, float]:
    row = conn.execute(
        "SELECT entry FROM synergy_history ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {}
    try:
        data = json.loads(row[0])
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def migrate_json_to_db(json_path: str | Path, db_path: str | Path) -> None:
    jp = Path(json_path)
    if not jp.exists():
        return
    try:
        data = json.loads(jp.read_text())
        if not isinstance(data, list):
            return
    except Exception:
        return
    conn = connect(db_path)
    try:
        for entry in data:
            if isinstance(entry, dict):
                insert_entry(conn, {str(k): float(v) for k, v in entry.items()})
    finally:
        conn.close()
