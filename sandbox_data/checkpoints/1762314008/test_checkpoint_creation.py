from pathlib import Path

import importlib
import types
import sys

from menace_sandbox.dynamic_path_router import resolve_path

pkg = types.ModuleType("menace_sandbox.self_improvement")
pkg.__path__ = [str(Path(resolve_path("self_improvement")))]
sys.modules["menace_sandbox.self_improvement"] = pkg
snapshot_tracker = importlib.import_module(
    "menace_sandbox.self_improvement.snapshot_tracker"
)


def test_save_checkpoint_creates_file(tmp_path):
    module = tmp_path / Path("mod").with_suffix(".py")  # path-ignore
    module.write_text("print('hi')\n")

    dest = snapshot_tracker.save_checkpoint(module, "cycle1")
    expected = (
        Path(resolve_path("sandbox_data"))
        / "checkpoints"
        / "mod"
        / Path("cycle1").with_suffix(".py")  # path-ignore
    )
    assert dest == expected
    assert dest.exists()
    assert dest.read_text() == module.read_text()
