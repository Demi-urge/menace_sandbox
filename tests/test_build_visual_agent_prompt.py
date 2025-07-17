import os
import sys
import types
import importlib.util

# stub heavy optional deps
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
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
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
pkg_path = os.path.join(os.path.dirname(__file__), "..")
pkg_spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(pkg_path, "__init__.py"), submodule_search_locations=[pkg_path]
)
menace_pkg = importlib.util.module_from_spec(pkg_spec)
sys.modules["menace"] = menace_pkg
pkg_spec.loader.exec_module(menace_pkg)

import menace.self_coding_engine as sce  # noqa: E402


def test_build_visual_agent_prompt_basic():
    prompt = sce.SelfCodingEngine(None, None).build_visual_agent_prompt(
        "helper.py", "print hello", "def hello():\n    pass"
    )
    assert "### Introduction" in prompt
    assert "helper.py" in prompt
    assert "print hello" in prompt
    assert "def hello()" in prompt


def test_build_visual_agent_prompt_env(monkeypatch, tmp_path):
    tpl = tmp_path / "va.tmpl"
    tpl.write_text("FUNC {func} DESC {description} CONT {context} PATH {path}\n")
    monkeypatch.setenv("VA_PROMPT_TEMPLATE", str(tpl))
    monkeypatch.setenv("VA_PROMPT_PREFIX", "NOTE: ")
    import importlib
    importlib.reload(sce)
    prompt = sce.SelfCodingEngine(None, None).build_visual_agent_prompt(
        "a.py", "do things", "ctx"
    )
    assert prompt.startswith("NOTE: ")
    assert "do things" in prompt
    assert "ctx" in prompt


def test_build_visual_agent_prompt_sections():
    import importlib
    importlib.reload(sce)
    snippet = "def add(x, y):\n    return x + y"
    prompt = sce.SelfCodingEngine(None, None).build_visual_agent_prompt(
        "utils.py", "add numbers", snippet
    )
    sections = [
        "### Introduction",
        "### Functions",
        "### Dependencies",
        "### Coding standards",
        "### Repository layout",
        "### Metadata",
        "### Version control",
        "### Testing",
        "### Snippet context",
    ]
    for sec in sections:
        assert sec in prompt
    assert snippet in prompt


def test_build_visual_agent_prompt_layout(monkeypatch):
    monkeypatch.setenv("VA_REPO_LAYOUT_LINES", "2")
    import importlib
    importlib.reload(sce)
    eng = sce.SelfCodingEngine(None, None)
    expected = eng._get_repo_layout(2)
    prompt = eng.build_visual_agent_prompt("a.py", "desc", "ctx")
    for line in expected.splitlines():
        assert line in prompt
