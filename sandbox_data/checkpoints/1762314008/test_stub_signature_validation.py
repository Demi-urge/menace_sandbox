import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import sandbox_runner.environment as env  # noqa: E402


def sample(a, b, c):
    return a + b + c


def test_generated_stubs_match_signature():
    stubs = env._validate_stubs([
        {"a": 1, "b": 2, "c": 3, "extra": 4}
    ], sample)
    assert stubs == [{"a": 1, "b": 2, "c": 3}]
