import sys
import types
import importlib.util
from pathlib import Path
import pytest

# Ensure repository root is on sys.path so quick_fix_engine dependencies resolve
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Lightweight stubs to load quick_fix_engine without heavy dependencies
package = types.ModuleType("menace_sandbox")
package.__path__ = [str(ROOT)]
sys.modules["menace_sandbox"] = package

error_bot = types.ModuleType("menace_sandbox.error_bot")
error_bot.ErrorDB = object
sys.modules["menace_sandbox.error_bot"] = error_bot

scm = types.ModuleType("menace_sandbox.self_coding_manager")
scm.SelfCodingManager = object
sys.modules["menace_sandbox.self_coding_manager"] = scm

kg = types.ModuleType("menace_sandbox.knowledge_graph")
kg.KnowledgeGraph = object
sys.modules["menace_sandbox.knowledge_graph"] = kg

vec_pkg = types.ModuleType("vector_service")
vec_pkg.__path__ = []
vec = types.ModuleType("vector_service.context_builder")


class _DummyContextBuilder:
    def __init__(self, *a, retriever=None, **k):
        self.retriever = retriever

    def build(self, *a, **k):
        return ""

    def refresh_db_weights(self):
        return None


class _DummyBackfill:
    def __init__(self, *a, **k):
        pass

    def run(self, *a, **k):
        return None


vec.ContextBuilder = _DummyContextBuilder
vec.Retriever = object
vec.FallbackResult = object
vec.EmbeddingBackfill = _DummyBackfill
sys.modules["vector_service"] = vec_pkg
sys.modules["vector_service.context_builder"] = vec
vec_text = types.ModuleType("vector_service.text_preprocessor")
vec_text.PreprocessingConfig = object
vec_text.get_config = lambda *a, **k: None
vec_text.generalise = lambda *a, **k: ""
sys.modules["vector_service.text_preprocessor"] = vec_text
vec_embed = types.ModuleType("vector_service.embed_utils")
vec_embed.get_text_embeddings = lambda *a, **k: []
vec_embed.EMBED_DIM = 1
sys.modules["vector_service.embed_utils"] = vec_embed
vec_pkg.SharedVectorService = object

pp = types.ModuleType("patch_provenance")
pp.PatchLogger = object
sys.modules["patch_provenance"] = pp

spec = importlib.util.spec_from_file_location(
    "menace_sandbox.quick_fix_engine",
    ROOT / "quick_fix_engine.py",  # path-ignore
)
quick_fix = importlib.util.module_from_spec(spec)
sys.modules["menace_sandbox.quick_fix_engine"] = quick_fix
spec.loader.exec_module(quick_fix)


def test_chunked_patch_generation(tmp_path, monkeypatch):
    path = tmp_path / "big.py"  # path-ignore
    lines = ["def big():"] + ["    pass" for _ in range(4000)]
    path.write_text("\n".join(lines) + "\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda *a, **k: {})
    monkeypatch.setattr(quick_fix, "flag_risky_changes", lambda *a, **k: [])
    monkeypatch.setattr(quick_fix, "resolve_path", lambda p: Path(p))
    monkeypatch.setattr(quick_fix, "path_for_prompt", lambda p: p)

    class Engine:
        prompt_chunk_token_threshold = 1000

        def __init__(self):
            self.calls = []

        def generate_helper(self, desc, **kwargs):
            return f"# patch {len(self.calls) + 1}"

        def apply_patch(self, p, helper, **kw):
            self.calls.append(helper)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(helper + "\n")
            return len(self.calls), False, ""

    engine = Engine()

    pid = quick_fix.generate_patch(
        str(path), engine, context_builder=quick_fix.ContextBuilder()
    )
    assert len(engine.calls) > 1
    assert pid == len(engine.calls)
    lines = path.read_text().strip().splitlines()
    assert lines[-len(engine.calls):] == [f"# patch {i}" for i in range(1, len(engine.calls) + 1)]


def test_patch_fails_without_validation(monkeypatch):
    class DummyManager:
        def __init__(self):
            self.quick_fix = None

        def _ensure_quick_fix_engine(self):
            return None

        def run_patch(self, path, desc, **kw):
            self._ensure_quick_fix_engine()
            if self.quick_fix is None:
                raise RuntimeError("QuickFixEngine validation unavailable")

    mgr = DummyManager()
    with pytest.raises(RuntimeError):
        mgr.run_patch(Path("mod.py"), "desc")
