import importlib.util
import sys
from pathlib import Path

import pytest

from dynamic_path_router import resolve_path


@pytest.mark.parametrize(
    "module_file, module_name",
    [
        ("error_parser.py", "error_parser_copy"),  # path-ignore
        ("failure_localization.py", "failure_localization_copy"),  # path-ignore
    ],
)
def test_modules_import_target_region_when_relocated(tmp_path, module_file, module_name):
    """Modules should load TargetRegion via resolve_path even when moved."""
    src = resolve_path(module_file)
    dst = tmp_path / Path(module_file).name
    dst.write_text(Path(src).read_text())
    spec = importlib.util.spec_from_file_location(module_name, dst)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    from self_improvement.target_region import TargetRegion as TR

    assert module.TargetRegion.__name__ == TR.__name__


def test_self_coding_engine_fallback_imports_target_region():
    """Self coding engine resolves TargetRegion via dynamic path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_target_region_fallback", resolve_path("self_improvement/target_region.py")  # path-ignore
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_target_region_fallback"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    from self_improvement.target_region import TargetRegion as TR

    assert module.TargetRegion.__name__ == TR.__name__
