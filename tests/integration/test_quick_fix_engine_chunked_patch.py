import sys
import types
import importlib.util
from pathlib import Path
import pytest
from menace.coding_bot_interface import manager_generate_helper

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
# Stub out coding bot interface to avoid heavy self-coding engine imports
cbi = types.ModuleType("menace_sandbox.coding_bot_interface")
cbi.self_coding_managed = lambda cls: cls
cbi.manager_generate_helper = manager_generate_helper
sys.modules["menace_sandbox.coding_bot_interface"] = cbi

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
# Provide minimal vector_service attributes so quick_fix_engine import succeeds
class _ErrorResult(Exception):
    pass
vec_pkg.ErrorResult = _ErrorResult
vec_pkg.ContextBuilder = _DummyContextBuilder
vec_pkg.CognitionLayer = object
vec_pkg.SharedVectorService = object
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
    manager = types.SimpleNamespace(
        engine=engine, register_patch_cycle=lambda *a, **k: None
    )

    pid = quick_fix.generate_patch(
        str(path), manager, engine, context_builder=quick_fix.ContextBuilder()
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


def test_helper_generation_retries_and_logging(tmp_path, monkeypatch, caplog):
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("def f():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda *a, **k: {})
    monkeypatch.setattr(quick_fix, "flag_risky_changes", lambda *a, **k: [])
    monkeypatch.setattr(quick_fix, "resolve_path", lambda p: Path(p))
    monkeypatch.setattr(quick_fix, "path_for_prompt", lambda p: p)

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, topic, payload):
            events.append((topic, payload))

    calls: list[tuple[str, str, bool, list[str] | None]] = []

    class DataBot:
        def record_validation(self, bot, module, passed, flags=None):
            calls.append((bot, module, passed, flags))

    class Engine:
        helper_retry_attempts = 2
        helper_retry_delay = 0
        bot_name = "bot"

        def __init__(self):
            self.event_bus = Bus()
            self.data_bot = DataBot()

        def generate_helper(self, desc, **kw):
            raise RuntimeError("fail")

        def apply_patch(self, *a, **kw):
            return None, False, ""

    engine = Engine()
    manager = types.SimpleNamespace(
        engine=engine, register_patch_cycle=lambda *a, **k: None
    )

    with caplog.at_level("ERROR"):
        quick_fix.generate_patch(
            str(path), manager, engine, context_builder=quick_fix.ContextBuilder()
        )

    assert len(events) == engine.helper_retry_attempts
    assert calls == [("bot", "mod.py", False, ["helper_generation_failed"])]
    assert "helper generation failed" in caplog.text


def test_risk_flags_emitted_for_large_diff(tmp_path, monkeypatch):
    path = tmp_path / "mod.py"  # path-ignore
    path.write_text("def f():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quick_fix, "resolve_path", lambda p: Path(p))
    monkeypatch.setattr(quick_fix, "path_for_prompt", lambda p: p)
    diff_data = {
        "mod.py": {
            "status": "modified",
            "changes": {"added": {"f": ["+x"] * 60}, "removed": {}, "modified": {}},
        }
    }
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda *a, **k: diff_data)

    class Engine:
        def generate_helper(self, desc, **kw):
            return "# fix"

        def apply_patch(self, p, helper, **kw):
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(helper + "\n")
            return 1, False, ""

    engine = Engine()
    manager = types.SimpleNamespace(
        engine=engine, register_patch_cycle=lambda *a, **k: None
    )

    pid, flags = quick_fix.generate_patch(
        str(path), manager, engine, context_builder=quick_fix.ContextBuilder(), return_flags=True
    )
    assert pid == 1
    assert any("large diff" in f for f in flags)


def test_risk_flags_emitted_for_sensitive_module(tmp_path, monkeypatch):
    path = tmp_path / "security_mod.py"  # path-ignore
    path.write_text("def f():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quick_fix, "resolve_path", lambda p: Path(p))
    monkeypatch.setattr(quick_fix, "path_for_prompt", lambda p: p)
    diff_data = {
        "security_mod.py": {
            "status": "modified",
            "changes": {"modified": {"f": ["+x"]}},
        }
    }
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda *a, **k: diff_data)

    class Engine:
        def generate_helper(self, desc, **kw):
            return "# fix"

        def apply_patch(self, p, helper, **kw):
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(helper + "\n")
            return 1, False, ""

    engine = Engine()
    manager = types.SimpleNamespace(
        engine=engine, register_patch_cycle=lambda *a, **k: None
    )

    pid, flags = quick_fix.generate_patch(
        str(path), manager, engine, context_builder=quick_fix.ContextBuilder(), return_flags=True
    )
    assert pid == 1
    assert any("critical file modified" in f for f in flags)
