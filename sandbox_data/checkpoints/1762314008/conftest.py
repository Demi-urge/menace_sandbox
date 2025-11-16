import sys
from pathlib import Path
import types

sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(
        resolve_path=lambda p: Path(p),
        resolve_dir=lambda p: Path(p),
        path_for_prompt=lambda p: Path(p).as_posix(),
    ),
)
