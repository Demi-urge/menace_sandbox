from pathlib import Path
from mypy import api


def test_mypy_clean():
    file = Path(__file__).resolve().parents[1] / "capital_management_bot.py"  # path-ignore
    result = api.run([
        str(file),
        "--strict",
        "--ignore-missing-imports",
        "--follow-imports=skip",
    ])
    # mypy writes output to stdout when successful and errors to stderr
    assert result[1] == ""
