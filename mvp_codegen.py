"""Single-pass code generation with strict safety constraints."""


def run_generation(task: dict[str, object]) -> str:
    """Generate a safe Python script from a task payload with strict safeguards.

    The task payload must include an objective and may include constraints and a
    callable model wrapper. The function returns a deterministic fallback script
    whenever the payload is invalid, the wrapper is missing, or validation fails.
    The function is side-effect-free and always returns a Python code string.

    Args:
        task: Task payload containing the objective, optional constraints, and optional
            model wrapper.

    Returns:
        A safe Python script derived from the task payload or a deterministic fallback.
    """
    import ast
    import sys

    fallback_objective = "unspecified objective"
    fallback_constraints_line = "none"
    objective = fallback_objective
    constraints: list[str] = []
    constraints_section = "- none"
    use_fallback = True

    if isinstance(task, dict):
        raw_objective = task.get("objective")
        if isinstance(raw_objective, str):
            objective_candidate = raw_objective.strip()
            if objective_candidate:
                objective = objective_candidate
                constraints_value = task.get("constraints")
                if isinstance(constraints_value, list):
                    for item in constraints_value:
                        if isinstance(item, str):
                            text = item.strip()
                        elif item is None:
                            text = ""
                        else:
                            try:
                                text = str(item).strip()
                            except Exception:
                                text = ""
                        if text:
                            constraints.append(text)
                elif constraints_value is not None:
                    if isinstance(constraints_value, str):
                        text = constraints_value.strip()
                    else:
                        try:
                            text = str(constraints_value).strip()
                        except Exception:
                            text = ""
                    if text:
                        constraints.append(text)
                constraint_texts = [
                    item.strip()
                    for item in constraints
                    if isinstance(item, str) and item.strip()
                ]
                constraints_section = (
                    "\n".join(f"- {item}" for item in constraints) if constraints else "- none"
                )
                fallback_constraints_line = (
                    ", ".join(constraint_texts) if constraint_texts else "none"
                )
                use_fallback = False

    lines = [
        '"""Deterministic placeholder script generated without a model wrapper."""',
        "",
        "def main() -> None:",
        f"    objective = {objective!r}",
        f"    constraints = {fallback_constraints_line!r}",
        "    summary_lines = [",
        '        "Placeholder execution: no model wrapper provided.",',
        '        f"Objective: {objective}",',
        '        f"Constraints: {constraints}",',
        '        "Status: completed safe placeholder run.",',
        "    ]",
        "    print(\"\\n\".join(summary_lines))",
        "",
        "if __name__ == \"__main__\":",
        "    main()",
    ]
    fallback_code = "\n".join(lines)

    if use_fallback:
        return fallback_code

    wrapper = task.get("model_wrapper") if isinstance(task, dict) else None
    if wrapper is None or not callable(wrapper):
        return fallback_code

    safety_prompt = (
        "System safety rules (mandatory):\n"
        "- Single-pass, non-recursive output only.\n"
        "- Do not use or suggest dangerous imports (os, sys, subprocess, socket, pathlib, shutil, "
        "requests, http, urllib, openai, etc.).\n"
        "- No system calls, subprocess usage, filesystem access (including io/open), network access, "
        "or shell commands.\n"
        "- Do not use open, eval, exec, compile, __import__, or input.\n"
        "Return only runnable Python code.\n"
    )

    prompt_text = (
        f"{safety_prompt}\n"
        f"Objective:\n{objective}\n\n"
        f"Constraints:\n{constraints_section}\n"
    )

    timeout_value = task.get("timeout_s") if isinstance(task, dict) else None

    try:
        if timeout_value is not None:
            try:
                raw_output = wrapper(prompt_text, timeout_s=timeout_value)
            except TypeError:
                raw_output = wrapper(prompt_text)
        else:
            raw_output = wrapper(prompt_text)
    except TimeoutError:
        return fallback_code
    except Exception:
        return fallback_code

    if isinstance(raw_output, bytes):
        try:
            output_text = raw_output.decode("utf-8")
        except Exception:
            return fallback_code
    else:
        try:
            output_text = str(raw_output)
        except Exception:
            return fallback_code

    output_text = output_text.replace("\x00", "")
    output_text = output_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if output_text.startswith("```"):
        lines = output_text.split("\n")
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        output_text = "\n".join(lines).strip()

    if not output_text:
        return fallback_code

    max_length = 4000
    if len(output_text) > max_length:
        cut = output_text.rfind("\n", 0, max_length + 1)
        if cut == -1:
            output_text = output_text[:max_length].rstrip()
        else:
            output_text = output_text[:cut].rstrip()

    if not output_text:
        return fallback_code

    if len(output_text) > max_length:
        return fallback_code

    try:
        parsed = ast.parse(output_text)
    except Exception:
        return fallback_code

    stdlib_allowlist = getattr(sys, "stdlib_module_names", None)
    if stdlib_allowlist is None:
        stdlib_allowlist = {
            "__future__",
            "_ast",
            "_thread",
            "abc",
            "argparse",
            "array",
            "ast",
            "base64",
            "binascii",
            "bisect",
            "builtins",
            "calendar",
            "collections",
            "cmath",
            "contextlib",
            "copy",
            "csv",
            "dataclasses",
            "datetime",
            "decimal",
            "difflib",
            "dis",
            "enum",
            "errno",
            "fnmatch",
            "fractions",
            "functools",
            "gc",
            "getopt",
            "getpass",
            "gettext",
            "glob",
            "hashlib",
            "heapq",
            "hmac",
            "html",
            "inspect",
            "io",
            "itertools",
            "json",
            "linecache",
            "locale",
            "logging",
            "math",
            "operator",
            "os",
            "pathlib",
            "pickle",
            "platform",
            "pprint",
            "random",
            "re",
            "reprlib",
            "secrets",
            "shlex",
            "shutil",
            "signal",
            "statistics",
            "string",
            "struct",
            "tempfile",
            "subprocess",
            "sys",
            "textwrap",
            "threading",
            "time",
            "traceback",
            "types",
            "typing",
            "uuid",
            "warnings",
            "weakref",
            "asyncio",
            "ftplib",
            "http",
            "imaplib",
            "poplib",
            "selectors",
            "smtplib",
            "socket",
            "ssl",
            "telnetlib",
            "urllib",
        }

    filesystem_modules = {
        "glob",
        "io",
        "os",
        "pathlib",
        "shutil",
        "tempfile",
        # These modules open or persist files on disk.
        "dbm",
        "logging",
        "shelve",
        "sqlite3",
    }
    network_modules = {
        "asyncio",
        "ftplib",
        "http",
        "imaplib",
        "poplib",
        "requests",
        "selectors",
        "smtplib",
        "socket",
        "ssl",
        "telnetlib",
        "urllib",
    }
    extra_denied_modules = {
        "ctypes",
        "importlib",
        "subprocess",
        "sys",
        "xmlrpc",
    }
    denied_modules = filesystem_modules | network_modules | extra_denied_modules
    banned_builtins = {"open", "eval", "exec", "compile", "__import__", "input"}
    banned_base_names = {"builtins", "__builtins__"}
    banned_dynamic_call_names = {"__import__", "import_module"}
    banned_module_attrs = {"import_module", "reload"}
    banned_symbol_names = banned_builtins | banned_dynamic_call_names | banned_module_attrs
    banned_reflection_calls = {"getattr", "setattr", "globals", "locals", "vars", "dir"}
    banned_aliases: set[str] = set()
    banned_module_aliases: set[str] = set()
    io_aliases: set[str] = {"io"}
    io_file_calls = {"open", "open_code", "FileIO", "BufferedWriter", "BufferedReader", "BufferedRandom", "TextIOWrapper"}
    banned_builtin_attrs = {"__dict__", "__getattribute__"}

    banned_call_paths = {
        "logging.FileHandler",
        "logging.handlers.FileHandler",
        "logging.handlers.RotatingFileHandler",
        "logging.handlers.TimedRotatingFileHandler",
        "logging.handlers.WatchedFileHandler",
        "sqlite3.connect",
    }

    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_name = alias.name.split(".")[0]
                if base_name not in stdlib_allowlist:
                    return fallback_code
                if base_name in denied_modules:
                    return fallback_code
                if base_name == "io" and alias.asname:
                    io_aliases.add(alias.asname)
                if alias.asname and base_name in banned_base_names:
                    banned_aliases.add(alias.asname)
                    banned_module_aliases.add(alias.asname)
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            module_base = module_name.split(".")[0]
            if module_base not in stdlib_allowlist:
                return fallback_code
            if module_base in denied_modules:
                return fallback_code
            for alias in node.names:
                if alias.name in banned_symbol_names:
                    return fallback_code
                if alias.asname and alias.asname in banned_symbol_names:
                    return fallback_code
                if module_name in banned_base_names and alias.name in banned_builtins:
                    return fallback_code
                if alias.asname and module_base in banned_base_names:
                    banned_aliases.add(alias.asname)
                    banned_module_aliases.add(alias.asname)
        if isinstance(node, ast.Name) and node.id in banned_builtins:
            return fallback_code
        if isinstance(node, ast.Name) and node.id in banned_aliases:
            return fallback_code
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in denied_modules:
                    return fallback_code
                if node.value.id in banned_base_names:
                    if node.attr in banned_builtins:
                        return fallback_code
                    if node.attr in banned_builtin_attrs:
                        return fallback_code
                if node.value.id in banned_aliases and node.attr in banned_builtin_attrs:
                    return fallback_code
                if node.value.id in banned_module_aliases:
                    return fallback_code
        if isinstance(node, ast.Subscript):
            key = None
            slice_node = node.slice
            if isinstance(slice_node, ast.Index):
                slice_node = slice_node.value
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                key = slice_node.value
            elif isinstance(slice_node, ast.Str):
                key = slice_node.s
            if isinstance(node.value, ast.Name) and node.value.id in banned_base_names:
                if key in banned_builtins:
                    return fallback_code
            if isinstance(node.value, ast.Attribute) and node.value.attr == "__dict__":
                if isinstance(node.value.value, ast.Name):
                    base_name = node.value.value.id
                    if base_name in banned_base_names or base_name in banned_aliases or base_name in banned_module_aliases:
                        if key in banned_builtins or key in denied_modules:
                            return fallback_code
            if isinstance(node.value, ast.Subscript):
                parent = node.value
                parent_key = None
                parent_slice_node = parent.slice
                if isinstance(parent_slice_node, ast.Index):
                    parent_slice_node = parent_slice_node.value
                if isinstance(parent_slice_node, ast.Constant) and isinstance(parent_slice_node.value, str):
                    parent_key = parent_slice_node.value
                elif isinstance(parent_slice_node, ast.Str):
                    parent_key = parent_slice_node.s
                if parent_key == "__builtins__":
                    if isinstance(parent.value, ast.Call):
                        call = parent.value
                        if isinstance(call.func, ast.Name) and call.func.id in {"globals", "locals", "vars"}:
                            if key in banned_builtins:
                                return fallback_code
            if isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id in {"globals", "locals", "vars"}:
                    if key in banned_builtins or key in denied_modules:
                        return fallback_code
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in banned_symbol_names:
                    return fallback_code
                if func.id in banned_aliases:
                    return fallback_code
                if func.id in banned_reflection_calls:
                    literal = None
                    if func.id in {"getattr", "setattr"} and len(node.args) >= 2:
                        literal_node = node.args[1]
                        if isinstance(literal_node, ast.Index):
                            literal_node = literal_node.value
                        if isinstance(literal_node, ast.Constant) and isinstance(
                            literal_node.value, str
                        ):
                            literal = literal_node.value
                        elif isinstance(literal_node, ast.Str):
                            literal = literal_node.s
                        target = node.args[0]
                        if literal in banned_builtins and isinstance(target, ast.Name):
                            if target.id in banned_base_names or target.id in banned_aliases:
                                return fallback_code
                        if literal in denied_modules and isinstance(target, ast.Name):
                            if target.id in denied_modules or target.id in banned_module_aliases:
                                return fallback_code
                    if func.id == "dir" and node.args:
                        target = node.args[0]
                        if isinstance(target, ast.Name):
                            if target.id in banned_base_names or target.id in denied_modules:
                                return fallback_code
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    dotted_name = f"{func.value.id}.{func.attr}"
                    if dotted_name in banned_call_paths:
                        return fallback_code
                    if (
                        dotted_name.startswith("logging.handlers.")
                        and dotted_name.endswith("FileHandler")
                    ):
                        return fallback_code
                    if func.value.id in io_aliases and func.attr in io_file_calls:
                        return fallback_code
                    if func.value.id in banned_module_aliases and func.attr in banned_module_attrs:
                        return fallback_code

    return output_text
