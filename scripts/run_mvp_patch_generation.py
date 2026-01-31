"""Generate MVP patch output using the LLM client interface."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from error_ontology import classify_error
from llm_interface import LLMBackend, LLMClient, LLMResult
from menace_sandbox.stabilization.logging_wrapper import (
    StabilizationLoggingWrapper,
    wrap_with_logging,
)
from menace_sandbox.stabilization.patch_validator import validate_patch_text
from mvp_evaluator import evaluate_roi
from prompt_types import Prompt


@dataclass
class _HeuristicBackend(LLMBackend):
    """Deterministic backend that derives a patch from the prompt text."""

    model: str = "heuristic"

    def generate(self, prompt: Prompt, *, context_builder: object) -> LLMResult:  # type: ignore[override]
        _ = context_builder
        patch_text = _derive_patch(prompt.text)
        return LLMResult(raw={"backend": "heuristic"}, text=patch_text)


def _derive_patch(prompt_text: str) -> str:
    match = re.search(
        r"Toy file contents:\n(?P<contents>.+?)\nCaptured traceback:\n",
        prompt_text,
        flags=re.DOTALL,
    )
    contents = match.group("contents") if match else ""
    if "return a - b" in contents:
        return _build_diff()
    return _build_diff()


def _build_diff() -> str:
    return textwrap.dedent(
        """\
        diff --git a/toy.py b/toy.py
        index 1111111..2222222 100644
        --- a/toy.py
        +++ b/toy.py
        @@ -1,6 +1,6 @@
         def add(a, b):
        -    return a - b
        +    return a + b


         if __name__ == "__main__":
             print(add(2, 3))
        """
    ).rstrip() + "\n"


def _build_prompt(objective: str) -> Prompt:
    return Prompt(
        user=objective,
        metadata={"vector_confidences": [1.0], "intent_tags": ["mvp_codegen"]},
        origin="context_builder",
    )


def _build_client() -> LLMClient:
    backend = os.getenv("MVP_LLM_BACKEND", "heuristic").lower()
    if backend == "heuristic":
        return LLMClient(model="heuristic", backends=[_HeuristicBackend()])
    if backend == "openai":
        from llm_interface import OpenAIProvider

        return OpenAIProvider()
    if backend == "ollama":
        from local_client import OllamaClient

        return OllamaClient(
            model=os.getenv("MVP_LLM_MODEL", "mistral"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if backend == "vllm":
        from local_client import VLLMClient

        return VLLMClient(
            model=os.getenv("MVP_LLM_MODEL", "facebook/opt-125m"),
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
        )
    raise ValueError(f"Unknown MVP_LLM_BACKEND: {backend}")


def _normalize_patch_text(raw_text: str) -> tuple[str, str, str]:
    normalized_text = raw_text.strip()
    if normalized_text.startswith("```"):
        lines = normalized_text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        normalized_text = "\n".join(lines).strip()
    sanitized_text = normalized_text.replace("\r\n", "\n")
    return normalized_text, sanitized_text, sanitized_text


def _generate_patch(output_path: Path) -> str:
    toy_file = """def add(a, b):
    return a - b


if __name__ == "__main__":
    print(add(2, 3))
"""
    traceback = """Traceback (most recent call last):
  File "toy.py", line 6, in <module>
    assert add(2, 3) == 5
AssertionError: expected 5, got -1
"""
    objective = (
        "Generate a unified diff patch to fix the bug in the toy file.\n\n"
        "Toy file contents:\n"
        f"{toy_file}\n"
        "Captured traceback:\n"
        f"{traceback}"
    )
    constraints = [
        "Return only a unified diff patch.",
        "Target file path is toy.py.",
    ]

    backend = os.getenv("MVP_LLM_BACKEND", "heuristic").lower()
    model = os.getenv("MVP_LLM_MODEL", "heuristic")
    client = _build_client()
    constraint_block = "\n".join(f"- {item}" for item in constraints) if constraints else "- none"
    prompt_text = f"{objective}\n\nConstraints:\n{constraint_block}\n"
    prompt = _build_prompt(prompt_text)
    print(f"LLMClient.generate start backend={backend} model={model}")
    result = client.generate(prompt, context_builder=object())
    print(f"LLMClient.generate end backend={backend} model={model}")
    patch_text = result.text
    normalized_text, sanitized_text, final_text = _normalize_patch_text(patch_text)
    output_path.write_text(final_text.strip() + "\n", encoding="utf-8")
    return final_text


def _apply_patch_to_temp(patch_text: str, source_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="mvp_patch_"))
    temp_file = temp_dir / source_path.name
    temp_file.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    patch_file = temp_dir / "patch.diff"
    patch_file.write_text(patch_text, encoding="utf-8")
    result = subprocess.run(
        ["patch", "-p1", "--input", str(patch_file)],
        cwd=temp_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Patch application failed",
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
    return temp_file


def _run_toy(path: Path) -> tuple[str, str, int]:
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout, result.stderr, result.returncode


_apply_patch_to_temp = wrap_with_logging(_apply_patch_to_temp)
_run_toy = wrap_with_logging(_run_toy)


def _classify_error(error: object) -> str | None:
    classification = classify_error(error)
    status = classification.get("status")
    if isinstance(status, str):
        return status
    return None


def _extract_validation_summary(validation: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    valid = bool(validation.get("valid"))
    flags = list(validation.get("flags") or [])
    context = validation.get("context")
    if not isinstance(context, dict):
        context = {}
    return valid, flags, context


def main() -> None:
    output_path = Path(os.getenv("MVP_PATCH_OUTPUT", "mvp_patch_output.diff"))
    logger = StabilizationLoggingWrapper.start(source="run_mvp_patch_generation")
    try:
        patch_text = _generate_patch(output_path)
        raw_text = patch_text
        normalized_text, sanitized_text, final_text = _normalize_patch_text(raw_text)
        logger.log_normalization(
            raw_text=raw_text,
            normalized_text=normalized_text,
            sanitized_text=sanitized_text,
            error=None,
        )

        validation = validate_patch_text(final_text)
        valid, flags, context = _extract_validation_summary(validation)
        reason = ", ".join(flags) if flags else None
        error_category = _classify_error(flags if flags else reason) if not valid else None
        logger.log_validation(
            ok=valid,
            reason=reason,
            diagnostics_count=len(flags),
            error_category=error_category,
        )
        if not valid:
            raise RuntimeError(f"Patch validation failed: {flags}", context)

        temp_toy = _apply_patch_to_temp(final_text, REPO_ROOT / "toy.py")
        logger.log_handoff(
            ok=True,
            error=None,
            reason="patch_applied",
            error_category=None,
        )

        stdout, stderr, returncode = _run_toy(temp_toy)
        roi = evaluate_roi(stdout, stderr)
        exec_error = stderr.strip() if returncode else None
        exec_category = _classify_error(exec_error) if exec_error else None
        logger.log_handoff(
            ok=returncode == 0,
            error=exec_error,
            reason="toy_execution",
            error_category=exec_category,
        )
        logger._log(
            "stabilization.roi",
            ok=returncode == 0,
            roi=roi,
            stdout_length=len(stdout),
            stderr_length=len(stderr),
            returncode=returncode,
        )
    except Exception as exc:
        category = _classify_error(exc)
        logger.log_handoff(
            ok=False,
            error=str(exc),
            reason="mvp_patch_generation_failed",
            error_category=category,
        )
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
