"""Generate MVP patch output using the LLM client interface."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import textwrap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_interface import LLMBackend, LLMClient, LLMResult
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


def _generate_patch(output_path: Path) -> None:
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
    patch_text = result.text.strip()
    if patch_text.startswith("```"):
        lines = patch_text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        patch_text = "\n".join(lines).strip()
    output_path.write_text(patch_text + "\n", encoding="utf-8")


def main() -> None:
    output_path = Path(os.getenv("MVP_PATCH_OUTPUT", "mvp_patch_output.diff"))
    _generate_patch(output_path)


if __name__ == "__main__":
    main()
