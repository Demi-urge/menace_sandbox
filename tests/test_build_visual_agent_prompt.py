import types
import importlib.util
import importlib.machinery
import os
import sys

# stub heavy optional deps
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
jinja_mod.__spec__ = importlib.machinery.ModuleSpec(
    "jinja2", loader=None, is_package=True
)
jinja_mod.__path__ = []  # mark as package
sys.modules.setdefault("jinja2", jinja_mod)
jinja_ext_mod = types.ModuleType("jinja2.ext")
jinja_ext_mod.Extension = object
sys.modules.setdefault("jinja2.ext", jinja_ext_mod)
jinja_sandbox_mod = types.ModuleType("jinja2.sandbox")
jinja_sandbox_mod.ImmutableSandboxedEnvironment = object
sys.modules.setdefault("jinja2.sandbox", jinja_sandbox_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault(
    "gpt_memory",
    types.SimpleNamespace(
        GPTMemoryManager=object,
        STANDARD_TAGS=[],
        INSIGHT="insight",
        _summarise_text=lambda *a, **k: "",
    ),
)
sys.modules.setdefault(
    "menace.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
)
sys.modules.setdefault(
    "menace.shared_knowledge_module",
    types.SimpleNamespace(LOCAL_KNOWLEDGE_MODULE=None),
)
sys.modules.setdefault(
    "menace.local_knowledge_module",
    types.SimpleNamespace(
        LocalKnowledgeModule=type(
            "LocalKnowledgeModule", (), {"__init__": lambda self, *a, **k: None}
        ),
        init_local_knowledge=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "menace.gpt_knowledge_service", types.SimpleNamespace(GPTKnowledgeService=object)
)
for name in [
    "shared_gpt_memory",
    "shared_knowledge_module",
    "local_knowledge_module",
    "gpt_knowledge_service",
]:
    sys.modules.setdefault(name, sys.modules[f"menace.{name}"])
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
# additional stubs to avoid heavy imports
vec_mod = types.ModuleType("vector_service")
vec_mod.__path__ = []
vec_mod.CognitionLayer = object
vec_mod.PatchLogger = object
vec_mod.VectorServiceError = Exception
vec_mod.EmbeddableDBMixin = object
vec_mod.SharedVectorService = object
sys.modules.setdefault("vector_service", vec_mod)
retr_mod = types.ModuleType("vector_service.retriever")
retr_mod.Retriever = lambda: None
retr_mod.FallbackResult = list
sys.modules.setdefault("vector_service.retriever", retr_mod)
sys.modules.setdefault(
    "vector_service.decorators", types.ModuleType("vector_service.decorators")
)
sys.modules.setdefault(
    "vector_service.context_builder", types.ModuleType("vector_service.context_builder")
)
roi_mod = types.ModuleType("vector_service.roi_tags")

class _RoiTag:
    HIGH_ROI = types.SimpleNamespace(value="high-ROI")
    SUCCESS = types.SimpleNamespace(value="success")
    LOW_ROI = types.SimpleNamespace(value="low-ROI")
    NEEDS_REVIEW = types.SimpleNamespace(value="needs-review")
    BUG_INTRODUCED = types.SimpleNamespace(value="bug-introduced")
    BLOCKED = types.SimpleNamespace(value="blocked")

    @classmethod
    def validate(cls, value):
        return cls.SUCCESS

roi_mod.RoiTag = _RoiTag
sys.modules.setdefault("vector_service.roi_tags", roi_mod)
data_bot_mod = types.ModuleType("data_bot")
data_bot_mod.MetricsDB = object
sys.modules.setdefault("data_bot", data_bot_mod)
pkg_path = os.path.join(os.path.dirname(__file__), "..")
pkg_spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(pkg_path, "__init__.py"), submodule_search_locations=[pkg_path]
)
menace_pkg = importlib.util.module_from_spec(pkg_spec)
sys.modules["menace"] = menace_pkg
pkg_spec.loader.exec_module(menace_pkg)

import menace.self_coding_engine as sce  # noqa: E402


def test_build_visual_agent_prompt_basic(monkeypatch):
    captured: dict[str, str | None] = {}

    def fake_build_prompt(
        self, description, *, context="", retrieval_context="", retry_trace=None
    ):
        captured.update(
            description=description,
            context=context,
            retrieval_context=retrieval_context,
            retry_trace=retry_trace,
        )
        return "PROMPT"

    monkeypatch.setattr(sce.PromptEngine, "build_prompt", fake_build_prompt)
    prompt = sce.SelfCodingEngine(None, None).build_visual_agent_prompt(
        "helper.py", "print hello", "def hello():\n    pass"
    )
    assert prompt == "PROMPT"
    assert captured["description"] == "print hello"
    assert "def hello()" in captured["context"]
    assert captured["retrieval_context"] == ""
    assert captured["retry_trace"] is None


def test_build_visual_agent_prompt_env(monkeypatch, tmp_path):
    tpl = tmp_path / "va.tmpl"
    tpl.write_text("FUNC {func} DESC {description} CONT {context} PATH {path}\n")
    monkeypatch.setenv("VA_PROMPT_TEMPLATE", str(tpl))
    monkeypatch.setenv("VA_PROMPT_PREFIX", "NOTE: ")
    import importlib
    importlib.reload(sce)
    monkeypatch.setattr(sce.PromptEngine, "build_prompt", lambda self, d, **k: "PROMPT")
    prompt = sce.SelfCodingEngine(None, None).build_visual_agent_prompt(
        "a.py", "do things", "ctx"
    )
    assert prompt.startswith("NOTE: ")
    assert "FUNC auto_do_things DESC do things CONT ctx PATH a.py" in prompt
    assert prompt.strip().endswith("PROMPT")


def test_build_visual_agent_prompt_layout(monkeypatch):
    monkeypatch.setenv("VA_REPO_LAYOUT_LINES", "2")
    import importlib
    importlib.reload(sce)
    captured = {}

    def fake_build_prompt(
        self, description, *, context="", retrieval_context="", retry_trace=None
    ):
        captured["context"] = context
        return "PROMPT"

    monkeypatch.setattr(sce.PromptEngine, "build_prompt", fake_build_prompt)
    eng = sce.SelfCodingEngine(None, None)
    expected = eng._get_repo_layout(2)
    eng.build_visual_agent_prompt("a.py", "desc", "ctx")
    for line in expected.splitlines():
        assert line in captured["context"]


def test_build_visual_agent_prompt_retrieval_context(monkeypatch):
    captured = {}

    def fake_build_prompt(
        self, description, *, context="", retrieval_context="", retry_trace=None
    ):
        captured["retrieval_context"] = retrieval_context
        return "PROMPT"

    monkeypatch.setattr(sce.PromptEngine, "build_prompt", fake_build_prompt)
    eng = sce.SelfCodingEngine(None, None)
    rc = "{\"bots\": []}"
    eng.build_visual_agent_prompt("a.py", "desc", "ctx", rc)
    assert captured["retrieval_context"] == rc
