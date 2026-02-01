from pathlib import Path

from menace_sandbox.mvp_self_debug import _loop, _roi_delta_total


def _run_loop_on_copy(tmp_path: Path, *, max_attempts: int) -> tuple[Path, object]:
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "toy.py"
    temp_path = tmp_path / "toy.py"
    temp_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return temp_path, _loop(temp_path, max_attempts=max_attempts)


def test_mvp_self_debug_loop_toy_determinism(tmp_path: Path) -> None:
    max_attempts = 1
    original_path = Path(__file__).resolve().parents[1] / "toy.py"
    original_content = original_path.read_text(encoding="utf-8")

    first_path, first_result = _run_loop_on_copy(tmp_path, max_attempts=max_attempts)
    assert first_result.attempts <= max_attempts
    assert first_result.exit_reason in {
        "success",
        "no_patch_generated",
        "invalid_patch",
        "non_positive_roi_delta",
        "max_attempts_reached",
        "unknown",
    }
    if first_result.final_run.returncode != 0:
        assert first_result.classification
        assert (
            first_result.classification.get("status")
            or first_result.classification.get("matched_rule_id")
        )
        assert first_result.final_run.stderr or first_result.final_run.stdout
    assert first_result.patch_validity is not None
    assert "flags" in first_result.patch_validity

    second_path, second_result = _run_loop_on_copy(tmp_path, max_attempts=max_attempts)

    assert first_result.exit_reason == second_result.exit_reason
    assert first_result.patch_validity["flags"] == second_result.patch_validity["flags"]
    assert _roi_delta_total(first_result.roi_delta or {}) == _roi_delta_total(
        second_result.roi_delta or {}
    )

    assert first_path.read_text(encoding="utf-8") != ""
    assert second_path.read_text(encoding="utf-8") != ""
    assert original_path.read_text(encoding="utf-8") == original_content
