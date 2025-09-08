import runpy
import sys
import os

from tests.test_pytest_stub_signatures import load_self_test_service

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

sts = load_self_test_service()


def test_stub_handles_typed_annotations(tmp_path, monkeypatch):
    mod_code = (
        "from dataclasses import dataclass\n"
        "from enum import Enum\n\n"
        "@dataclass\n"
        "class Point:\n"
        "    x: int\n"
        "    y: int\n\n"
        "class Colour(Enum):\n"
        "    RED = 1\n"
        "    BLUE = 2\n\n"
        "called = []\n\n"
        "def handle(pt: Point, colour: Colour, nums: list[int], mapping: dict[str, int]):\n"
        "    called.append((pt, colour, nums, mapping))\n\n"
        "def needs_custom(a: int):\n"
        "    called.append(a)\n\n"
        "class C:\n"
        "    def __init__(self, colour: Colour, pts: list[Point]):\n"
        "        called.append((colour, pts))\n"
    )

    mod_path = tmp_path / "typed_mod.py"  # path-ignore
    mod_path.write_text(mod_code, encoding="utf-8")

    hook_mod = tmp_path / "hook_mod.py"  # path-ignore
    hook_mod.write_text(
        "def gen(p):\n    if p.name == 'a':\n        return 42\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *a, **k):
            if k.get("return_metadata"):
                return "", {}
            return ""

    svc = sts.SelfTestService(fixture_hook="hook_mod:gen", context_builder=DummyBuilder())
    stub_path = svc._generate_pytest_stub(str(mod_path))
    ns = runpy.run_path(stub_path)
    ns["test_stub"]()

    mod = sys.modules["typed_mod"]

    c_colour, pts = mod.called[0]
    assert isinstance(c_colour, mod.Colour)
    assert isinstance(pts, list) and isinstance(pts[0], mod.Point)

    pt, colour, nums, mapping = mod.called[1]
    assert isinstance(pt, mod.Point)
    assert isinstance(colour, mod.Colour)
    assert isinstance(nums, list) and isinstance(nums[0], int)
    assert isinstance(mapping, dict) and isinstance(next(iter(mapping)), str)

    assert mod.called[2] == 42

    stub_dir = stub_path.parent
    stub_path.unlink()
    stub_dir.rmdir()
    stub_dir.parent.rmdir()
    stub_dir.parent.parent.rmdir()
    sys.path.remove(str(tmp_path))
    sys.modules.pop("typed_mod", None)
    sys.modules.pop("hook_mod", None)
