from __future__ import annotations

from pathlib import Path


def test_no_direct_sqlite_connect_usage() -> None:
    """Ensure direct sqlite3.connect calls are confined to approved files.

    The project aims to centralise SQLite access through ``DBRouter``.  This
    test scans all Python sources and flags any unapproved ``sqlite3.connect``
    usage.  Existing legitimate usages are recorded in
    ``approved_sqlite3_usage.txt``.  Adding a new direct connection requires
    updating that allow list intentionally.
    """

    repo_root = Path(__file__).resolve().parents[1]
    allow_file = Path(__file__).with_name("approved_sqlite3_usage.txt")
    allowed = {
        line.strip()
        for line in allow_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }

    offenders: list[str] = []
    this_test = Path(__file__).relative_to(repo_root).as_posix()
    pattern = "sqlite3." + "connect("
    for path in repo_root.rglob("*.py"):  # path-ignore
        rel = path.relative_to(repo_root).as_posix()
        if rel == this_test:
            continue  # ignore this test file which intentionally mentions the pattern
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern in text and rel not in allowed:
            offenders.append(rel)

    assert not offenders, "Disallowed sqlite3.connect usage:\n" + "\n".join(offenders)
