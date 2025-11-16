import os
import subprocess
import sys
import json
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_module(tmp_path: Path, rel: str, content: str) -> None:
    file_path = tmp_path / rel
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(textwrap.dedent(content))


@pytest.fixture()
def cli_env(tmp_path: Path):
    modules = {
        "vector_service/__init__.py": """  # path-ignore
class ContextBuilder:
    def __init__(self, retriever=None, **kwargs):
        self.retriever = retriever

    def refresh_db_weights(self):
        return None
""",
        "vector_service/retriever.py": """  # path-ignore
import os, types
if os.environ.get('RETRIEVER_IMPORT_ERROR') == '1':
    raise ImportError('fail retriever')
class Retriever:
    def __init__(self, cache=None):
        pass
    def search(self, query, top_k=5, dbs=None):
        if os.environ.get('TEST_RETRIEVE_SEARCH_FAIL') == '1':
            raise RuntimeError('search fail')
        hit = types.SimpleNamespace(origin_db='code', record_id=1, score=0.99, text='vec')
        return [hit]
class FallbackResult(list):
    pass
def fts_search(q, dbs=None, limit=None):
    return [{'origin_db': 'code', 'record_id': 2, 'score': 0.5, 'text': 'fts'}]
""",
        "vector_service/embedding_backfill.py": """  # path-ignore
import os


class DummyDB:
    def __init__(self, vector_backend=None):
        mismatch = os.environ.get('TEST_VECTOR_MISMATCH') == '1'
        self.records = ['a', 'b']
        self._id_map = ['a'] if mismatch else ['a', 'b']

    def iter_records(self):
        for i, r in enumerate(self.records):
            yield i, r, 'text'

    def needs_refresh(self, record_id, record):
        return True

    def add_embedding(self, record_id, record, kind):
        if os.environ.get('TEST_EMBED_RECORD_FAIL') == '1' and record_id == 0:
            raise RuntimeError('record fail')


class EmbeddingBackfill:
    def __init__(self, batch_size: int = 100, backend: str = "annoy"):
        self.batch_size = batch_size
        self.backend = backend

    def run(self, session_id=None, dbs=None, batch_size=None, backend=None):
        if os.environ.get('TEST_EMBED_FAIL') == '1':
            from vector_service.exceptions import VectorServiceError
            raise VectorServiceError('embed failure')
        self._process_db(DummyDB(), batch_size=batch_size or self.batch_size, session_id=session_id)

    def _load_known_dbs(self, names=None):
        return [DummyDB]


_RUN_SKIPPED = type(
    "G",
    (),
    {"labels": lambda *a, **k: type("L", (), {"inc": lambda *a, **k: None})()},
)()


def _log_violation(*args, **kwargs):
    pass
""",
        "vector_service/exceptions.py": """  # path-ignore
class VectorServiceError(Exception):
    pass
""",
        "universal_retriever.py": """  # path-ignore
import os, types
if os.environ.get('UNIVERSAL_IMPORT_ERROR') == '1':
    raise ImportError('fail universal')
class UniversalRetriever:
    def retrieve(self, query, top_k=5, dbs=None):
        hit = types.SimpleNamespace(origin_db='uni', record_id=3, score=0.1, text='uni')
        return [hit], '', []
""",
        "quick_fix_engine.py": """  # path-ignore
import os


def generate_patch(
    module,
    manager,
    *,
    context_builder,
    engine=None,
    description=None,
    patch_logger=None,
    context=None,
):
    if os.environ.get('TEST_PATCH_FAIL') == '1':
        return None
    return 7
""",
        "code_database.py": """  # path-ignore
class PatchHistoryDB:
    def __init__(self):
        self.records = {7: type('R', (), {'filename': 'mod.py'})()}  # path-ignore
    def get(self, pid):
        return self.records.get(pid)
""",
        "patch_provenance.py": """  # path-ignore
class PatchLogger:
    def __init__(self, patch_db=None):
        pass
def build_chain(*args, **kwargs):
    return []
def search_patches_by_vector(*args, **kwargs):
    return []
def search_patches_by_license(*args, **kwargs):
    return []
""",
        "cache_utils.py": """  # path-ignore
def _get_cache():
    class C: pass
    return C()
def get_cached_chain(query, dbs):
    return None
def set_cached_chain(query, dbs, results):
    pass
def clear_cache():
    pass
def show_cache():
    return {}
def cache_stats():
    return {}
""",
        "scripts/new_db_template.py": """  # path-ignore
import os, sys
if os.environ.get('TEST_NEW_DB_FAIL') == '1':
    sys.exit(1)
with open('created.txt', 'w') as f:
    f.write(sys.argv[1])
""",
    }
    for rel, content in modules.items():
        _write_module(tmp_path, rel, content)
    env = {"PYTHONPATH": str(REPO_ROOT)}
    return tmp_path, env


def _run(tmp_path: Path, env: dict, *args: str, extra_env: dict | None = None):
    full_env = os.environ.copy()
    full_env.update(env)
    if extra_env:
        full_env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "menace_cli", *args],
        cwd=tmp_path,
        env=full_env,
        capture_output=True,
        text=True,
    )


def test_retrieve_success(cli_env):
    tmp, env = cli_env
    res = _run(tmp, env, "retrieve", "query", "--json", "--no-cache")
    assert res.returncode == 0
    assert json.loads(res.stdout) == [
        {"origin_db": "code", "record_id": 1, "score": 0.99, "snippet": "vec"}
    ]


def test_retrieve_failure(cli_env):
    tmp, env = cli_env
    res = _run(
        tmp,
        env,
        "retrieve",
        "query",
        extra_env={"RETRIEVER_IMPORT_ERROR": "1"},
    )
    assert res.returncode == 1
    assert "vector retriever unavailable" in res.stderr


def test_patch_success(cli_env):
    tmp, env = cli_env
    res = _run(tmp, env, "patch", "mod", "--desc", "fix")
    assert res.returncode == 0
    assert json.loads(res.stdout) == {"patch_id": 7, "files": ["mod.py"]}  # path-ignore


def test_patch_failure(cli_env):
    tmp, env = cli_env
    res = _run(tmp, env, "patch", "mod", "--desc", "fix", extra_env={"TEST_PATCH_FAIL": "1"})
    assert res.returncode == 1


def test_embed_success(cli_env):
    tmp, env = cli_env
    res = _run(tmp, env, "embed", "--db", "code")
    assert res.returncode == 0


def test_embed_failure(cli_env):
    tmp, env = cli_env
    res = _run(tmp, env, "embed", "--db", "code", extra_env={"TEST_EMBED_FAIL": "1"})
    assert res.returncode == 1
    assert "embed failure" in res.stderr


def test_embed_verify_mismatch(cli_env):
    tmp, env = cli_env
    res = _run(
        tmp,
        env,
        "embed",
        "--verify",
        extra_env={"TEST_VECTOR_MISMATCH": "1"},
    )
    assert res.returncode == 0
    assert "vectors for" in res.stderr


def test_embed_log_file(cli_env):
    tmp, env = cli_env
    log = tmp / "skip.log"
    res = _run(
        tmp,
        env,
        "embed",
        "--log-file",
        str(log),
        extra_env={"TEST_EMBED_RECORD_FAIL": "1"},
    )
    assert res.returncode == 0
    assert "embedding error" in log.read_text()


def test_new_db_success(cli_env):
    tmp, env = cli_env
    res = _run(tmp, env, "new-db", "mydb")
    assert res.returncode == 0
    assert (tmp / "created.txt").read_text() == "mydb"


def test_new_db_failure(cli_env):
    tmp, env = cli_env
    res = _run(tmp, env, "new-db", "mydb", extra_env={"TEST_NEW_DB_FAIL": "1"})
    assert res.returncode == 1
    assert not (tmp / "created.txt").exists()
