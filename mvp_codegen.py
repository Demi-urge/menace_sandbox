"""Single-pass code generation with safety constraints."""

from __future__ import annotations

from typing import Any

import local_model_wrapper


def run_generation(task: dict[str, object]) -> str:
    """Generate safe Python code from a task payload using a single model call.

    The task must include an ``objective`` string and may include optional
    ``constraints`` (string or list). The function is deterministic, defensive,
    and side-effect-free: it validates inputs, builds a strict safety system
    prompt, invokes the model wrapper exactly once, and sanitizes/truncates the
    output to safe Python code. On any failure, it returns a minimal placeholder
    script that prints an internal error message.
    """
    fallback_script = (
        "\"\"\"Internal error: code generation failed.\"\"\"\n"
        "def main() -> None:\n"
        "    print('Internal error: code generation failed.')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    if not isinstance(task, dict):
        return fallback_script

    objective_value = task.get("objective")
    if not isinstance(objective_value, str):
        return fallback_script
    objective = objective_value.strip()
    if not objective:
        return fallback_script

    constraints_value = task.get("constraints")
    constraints_list: list[str] = []
    if isinstance(constraints_value, str):
        constraints_list = [line.strip() for line in constraints_value.splitlines() if line.strip()]
    elif isinstance(constraints_value, list):
        constraints_list = [item.strip() for item in constraints_value if isinstance(item, str) and item.strip()]
    elif constraints_value is not None:
        constraints_list = [str(constraints_value).strip()] if str(constraints_value).strip() else []

    banned_imports = (
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shlex",
        "shutil",
        "tempfile",
        "ctypes",
        "importlib",
        "inspect",
        "signal",
        "asyncio",
        "multiprocessing",
        "threading",
        "resource",
        "builtins",
        "platform",
        "selectors",
        "ssl",
        "http",
        "urllib",
        "requests",
    )

    safety_prompt = (
        "You must output only plain Python code. "
        "Never import or use dangerous modules or system interfaces. "
        "Forbidden imports include: "
        + ", ".join(banned_imports)
        + ". "
        "Do not access the filesystem, environment variables, or network. "
        "Do not spawn processes, execute shell/system calls, or use reflection. "
        "Do not use eval/exec/compile, __import__, or dynamic imports. "
        "Only use safe, deterministic in-memory logic." 
    )

    constraints_section = "\n".join(f"- {item}" for item in constraints_list) if constraints_list else "- None provided."
    prompt_text = (
        f"Objective:\n{objective}\n\n"
        f"Constraints:\n{constraints_section}\n\n"
        "Respond with only valid Python code."
    )

    model = task.get("model")
    tokenizer = task.get("tokenizer")
    if model is None or tokenizer is None:
        return fallback_script

    output_text = ""
    try:
        wrapper = local_model_wrapper.LocalModelWrapper(model, tokenizer)
        prompt_obj = local_model_wrapper.Prompt(
            user=prompt_text,
            system=safety_prompt,
            metadata={"origin": "mvp_codegen"},
        )
        raw_output: Any = wrapper.generate(
            prompt_obj,
            context_builder=None,
            max_new_tokens=256,
            do_sample=False,
        )
        if isinstance(raw_output, list):
            output_text = str(raw_output[0]) if raw_output else ""
        else:
            output_text = str(raw_output)
    except Exception:
        return fallback_script

    output_text = str(output_text)
    output_text = output_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    output_text = output_text.replace("\r\n", "\n").replace("\r", "\n")
    output_text = "".join(char for char in output_text if char.isprintable() or char in {"\n", "\t"})

    forbidden_modules = set(banned_imports)
    forbidden_builtins = {"open", "exec", "eval", "compile", "__import__"}
    risky_call_snippets = [
        "os.",
        "sys.",
        "subprocess.",
        "socket.",
        "pathlib.",
        "shlex.",
        "shutil.",
        "tempfile.",
        "ctypes.",
        "importlib.",
        "inspect.",
        "signal.",
        "asyncio.",
        "multiprocessing.",
        "threading.",
        "resource.",
        "platform.",
        "selectors.",
        "ssl.",
        "http.",
        "urllib.",
        "requests.",
        "popen(",
        "system(",
        "shell=true",
        "__import__(",
    ]

    cleaned_lines: list[str] = []
    for line in output_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        lowered = stripped.lower()
        if stripped.startswith("import "):
            imported = stripped[len("import ") :].split("#", 1)[0]
            modules = [part.strip().split(" as ")[0] for part in imported.split(",")]
            if any(module in forbidden_modules for module in modules):
                cleaned_lines.append(f"{line[: len(line) - len(line.lstrip())]}pass")
                continue
        if stripped.startswith("from "):
            module_part = stripped[len("from ") :].split(" import ", 1)[0].strip()
            root_module = module_part.split(".", 1)[0]
            if root_module in forbidden_modules:
                cleaned_lines.append(f"{line[: len(line) - len(line.lstrip())]}pass")
                continue
        if any(bad in lowered for bad in risky_call_snippets):
            cleaned_lines.append(f"{line[: len(line) - len(line.lstrip())]}pass")
            continue
        if any(f"{builtin}(" in lowered for builtin in forbidden_builtins):
            cleaned_lines.append(f"{line[: len(line) - len(line.lstrip())]}pass")
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned or not any(char.isalnum() for char in cleaned):
        return fallback_script

    max_chars = 4000
    if len(cleaned) > max_chars:
        cutoff = cleaned.rfind("\n", 0, max_chars)
        cleaned = cleaned[: cutoff if cutoff != -1 else max_chars].rstrip()

    try:
        compile(cleaned, "<generated>", "exec")
    except Exception:
        return fallback_script

    return cleaned
