"""Utilities for flagging potential human-alignment issues in code patches.

This module exposes :class:`HumanAlignmentFlagger` which analyses unified
diff strings and returns structured reports describing any detected
alignment concerns.  The checker is intentionally conservative – it never
raises an exception and only relies on lightweight heuristics.

Checks cover a range of maintainability and safety heuristics including:

* comment density changes and explicit comment removals
* introduction of single-character identifiers and linter suppressions
* removal of type hints or absence of annotations in new code
* direct network, ``exec``/``eval`` calls or other risky patterns

Each reported issue includes a numeric ``severity`` along with a ``tier``
(``info``, ``warn`` or ``critical``) describing how the Menace integrity model
expects reviewers to act:

* ``info`` entries are logged for transparency only.
* ``warn`` entries should be reviewed and either fixed or explicitly waived.
* ``critical`` entries normally block integration until resolved.

Example
-------

>>> diff = '''--- a/util.py\n+++ b/util.py\n@@\n-def add(a, b):\n-    return a+b\n+def add(a: int, b: int) -> int:\n+    return a + b\n'''
>>> flagger = HumanAlignmentFlagger()
>>> report = flagger.flag_patch(diff, {"author": "bot"})
>>> report["tiers"]
{'info': 1}

To run the checker from the command line provide a pair of directories
representing the "before" and "after" trees:

``$ python human_alignment_flagger.py old/ new/``

The module only relies on lightweight parsing and never raises.
"""

# flake8: noqa
from __future__ import annotations

import ast
import difflib
import json
import yaml
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import re
from collections import Counter
from importlib import import_module, util

from dynamic_path_router import resolve_path
from ethics_violation_detector import flag_violations, scan_log_entry
from risk_domain_classifier import classify_action
from reward_sanity_checker import check_risk_reward_alignment
from sandbox_settings import SandboxSettings
from secret_redactor import redact_secrets
from license_detector import detect as detect_license
from analysis.semantic_diff_filter import find_semantic_risks


class Rule(Protocol):
    def __call__(self, path: str, added: list[str], removed: list[str]) -> list[dict]:
        ...


def _parse_diff_paths(diff: str) -> Dict[str, Dict[str, Any]]:
    """Return mapping of file paths to their added/removed lines and metrics.

    In addition to raw added/removed line lists this helper also tracks how
    many of those lines are comments and how many single-character identifiers
    are introduced or removed.  This enables downstream checks to reason about
    comment density and potential obfuscation.
    """

    files: Dict[str, Dict[str, Any]] = {}
    current: Dict[str, Any] | None = None
    single_char_re = re.compile(r"\b[A-Za-z]\b")

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            ext = Path(path).suffix.lower()
            # Determine comment markers based on file type so that we can
            # track comment density in non-Python files such as YAML/JSON and
            # shell scripts.
            comment_markers = ["#"]
            if ext in {".json"}:
                comment_markers = ["//", "#"]
            current = files.setdefault(
                path,
                {
                    "added": [],
                    "removed": [],
                    "comments_added": 0,
                    "comments_removed": 0,
                    "removed_comment_lines": [],
                    "single_char_added": 0,
                    "single_char_removed": 0,
                    "comment_markers": comment_markers,
                },
            )
        elif current is not None:
            markers = current.get("comment_markers", ["#"])
            if line.startswith("+") and not line.startswith("+++"):
                content = line[1:]
                current["added"].append(content)
                if any(content.lstrip().startswith(m) for m in markers):
                    current["comments_added"] += 1
                else:
                    current["single_char_added"] += len(single_char_re.findall(content))
            elif line.startswith("-") and not line.startswith("---"):
                content = line[1:]
                current["removed"].append(content)
                if any(content.lstrip().startswith(m) for m in markers):
                    current["comments_removed"] += 1
                    current["removed_comment_lines"].append(content)
                else:
                    current["single_char_removed"] += len(
                        single_char_re.findall(content)
                    )
    return files


def _secret_diff_rule(path: str, added: list[str], removed: list[str]) -> list[dict]:
    """Flag potential secrets introduced in ``added`` lines using secret_redactor."""
    issues: list[dict] = []
    for line in added:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            redacted = redact_secrets(stripped)
        except Exception:
            continue
        if redacted != stripped:
            issues.append(
                {
                    "severity": 4,
                    "tier": "critical",
                    "message": f"Secret detected in {path}: {stripped}",
                }
            )
    return issues


def _license_diff_rule(path: str, added: list[str], removed: list[str]) -> list[dict]:
    """Flag disallowed licenses in added text using license_detector."""
    text = "\n".join(added)
    try:
        lic = detect_license(text)
    except Exception:
        lic = None
    if lic:
        return [
            {
                "severity": 4,
                "tier": "critical",
                "message": f"Restricted license ({lic}) detected in {path}",
            }
        ]
    return []


class HumanAlignmentFlagger:
    """Analyse diffs for alignment risks and maintainability issues."""

    def __init__(
        self,
        settings: Optional[SandboxSettings] = None,
        rules: Optional[List[Rule]] = None,
    ) -> None:
        self.settings = settings
        cfg = getattr(settings, "alignment_rules", None) if settings else None
        self.max_complexity = getattr(cfg, "max_complexity_score", 10)
        self.max_comment_drop = getattr(cfg, "max_comment_density_drop", 0.1)
        self.allowed_network_calls = getattr(cfg, "allowed_network_calls", 0)
        self.comment_density_severity = getattr(cfg, "comment_density_severity", 2)
        self.network_call_severity = getattr(cfg, "network_call_severity", 3)

        self.rules: List[Rule] = list(rules or [])
        # Built-in diff rules for secrets and license checks
        self.rules.extend([_secret_diff_rule, _license_diff_rule])
        module_paths = getattr(cfg, "rule_modules", []) if cfg else []
        for mod in module_paths:
            module = None
            try:
                module = import_module(mod)
            except Exception:
                try:
                    if "" not in sys.path:
                        sys.path.append("")
                    module = import_module(mod)
                except Exception:
                    try:
                        rel = Path(mod.replace(".", "/") + ".py")
                        search_paths = [Path.cwd(), resolve_path(".")]
                        search_paths.extend(Path(p) for p in sys.path if p)
                        if rel.is_absolute():
                            candidate = rel
                        else:
                            candidate = None
                            for base in search_paths:
                                try:
                                    candidate = resolve_path(str(base / rel))
                                    break
                                except FileNotFoundError:
                                    continue
                        if candidate and candidate.exists():
                            spec = util.spec_from_file_location(mod, candidate)
                            module = util.module_from_spec(spec)
                            assert spec and spec.loader
                            spec.loader.exec_module(module)
                    except Exception:
                        module = None
            if not module:
                continue
            rule = getattr(module, "rule", None)
            if callable(rule):
                self.rules.append(rule)  # type: ignore[arg-type]

    def flag_patch(self, diff: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured report for *diff* with optional *context*.

        The report contains two high level metrics, ``lines_added`` and
        ``lines_removed``, along with a list of ``issues``.  Each entry in
        ``issues`` stores a ``severity`` score and a human readable
        ``message``.
        """

        files = _parse_diff_paths(diff)

        issues: List[Dict[str, Any]] = []
        lines_added = 0
        lines_removed = 0

        fs_mutation_calls = (
            "os.remove(",
            "os.unlink(",
            "os.rename(",
            "os.rmdir(",
            "shutil.rmtree(",
            "shutil.move(",
            "shutil.copyfile(",
        )
        fs_open_write_re = re.compile(r"open\([^,]+,[^)]*['\"](?:w|a|x)['\"]")
        abs_path_re = re.compile(r"['\"](/[^'\"]*)")
        linter_suppress_re = re.compile(
            r"#\s*(?:noqa|pylint:\s*disable|pragma:\s*no\s*cover)"
        )
        complexity_tokens = ("if", "for", "while", "and", "or", "try", "except", "elif")
        credential_re = re.compile(
            r"['\"]?(?:password|secret|token|api[_-]?key)['\"]?\s*[:=]\s*['\"]?\S+",
            re.IGNORECASE,
        )
        security_disable_re = re.compile(
            r"(?:allow[_-]?privilege[_-]?escalation|privileged|enable[_-]?security)\s*[:=]\s*(?:true|false)",
            re.IGNORECASE,
        )
        exec_flag_re = re.compile(
            r"chmod\s+(?:\+x|[0-7]*7[0-7]*|[0-7]{3})|set\s+\+e",
            re.IGNORECASE,
        )
        remote_exec_re = re.compile(r"(?:curl|wget).+\|\s*sh", re.IGNORECASE)

        def _tier(severity: int) -> str:
            return "critical" if severity >= 3 else "warn" if severity >= 2 else "info"

        for path, info in files.items():
            added, removed = info["added"], info["removed"]
            lines_added += len(added)
            lines_removed += len(removed)

            # Opacity checks: comment density and identifier obfuscation
            removed_comments = info.get("comments_removed", 0)
            added_comments = info.get("comments_added", 0)
            if removed_comments > 0:
                removed_ratio = removed_comments / max(len(removed), 1)
                added_ratio = added_comments / max(len(added), 1)
                drop = removed_ratio - added_ratio
                if drop > self.max_comment_drop:
                    sev = self.comment_density_severity
                    issues.append(
                        {
                            "severity": sev,
                            "tier": _tier(sev),
                            "message": f"Opacity: comment density decreased in {path}",
                        }
                    )
                for comment in info.get("removed_comment_lines", []):
                    sev = self.comment_density_severity
                    issues.append(
                        {
                            "severity": sev,
                            "tier": _tier(sev),
                            "message": f"Comment removed in {path}: {comment.strip()}",
                        }
                    )

            if removed and info.get("single_char_added", 0) > info.get("single_char_removed", 0):
                sev = 2
                issues.append(
                    {
                        "severity": sev,
                        "tier": _tier(sev),
                        "message": f"Opacity: single-character identifiers introduced in {path}",
                    }
                )

            # Docstring removal or absence
            if any('"""' in line or "'''" in line for line in removed):
                sev = 3
                issues.append({
                    "severity": sev,
                    "tier": _tier(sev),
                    "message": f"Docstring removed in {path}",
                })
            if path.endswith(".py") and len(added) > 1 and not any(
                '"""' in line or "'''" in line for line in added[:5]
            ):
                sev = 1
                issues.append({
                    "severity": sev,
                    "tier": _tier(sev),
                    "message": f"{path} may lack module docstring",
                })

            # Logging statements removed
            if any("logging." in line or "logger." in line for line in removed):
                sev = 2
                issues.append({
                    "severity": sev,
                    "tier": _tier(sev),
                    "message": f"Logging removed in {path}",
                })

            # Test code removed
            path_obj = Path(path)
            if path_obj.parts and (path_obj.parts[0] == "tests" or "test" in path_obj.name):
                if removed:
                    sev = 4
                    issues.append({
                        "severity": sev,
                        "tier": _tier(sev),
                        "message": f"Test code removed in {path}",
                    })
            elif any("assert" in line for line in removed):
                sev = 4
                issues.append({
                    "severity": sev,
                    "tier": _tier(sev),
                    "message": f"Test assertion removed in {path}",
                })

            # Configuration and script specific heuristics -----------------
            ext = path_obj.suffix.lower()
            if ext in {".yaml", ".yml", ".json"}:
                for line in added:
                    stripped = line.strip()
                    if security_disable_re.search(stripped):
                        lower = stripped.lower()
                        if (
                            ("allow" in lower and "true" in lower)
                            or ("privileged" in lower and "true" in lower)
                            or ("enable" in lower and "false" in lower)
                        ):
                            sev = 3
                            issues.append(
                                {
                                    "severity": sev,
                                    "tier": _tier(sev),
                                    "message": f"Security feature disabled in {path}: {stripped}",
                                }
                            )
                    if credential_re.search(stripped):
                        sev = 3
                        issues.append(
                            {
                                "severity": sev,
                                "tier": _tier(sev),
                                "message": f"Embedded credential in {path}: {stripped}",
                            }
                        )
            elif ext in {".sh", ".bash"}:
                for line in added:
                    stripped = line.strip()
                    if credential_re.search(stripped):
                        sev = 3
                        issues.append(
                            {
                                "severity": sev,
                                "tier": _tier(sev),
                                "message": f"Embedded credential in {path}: {stripped}",
                            }
                        )
                    if remote_exec_re.search(stripped):
                        sev = 3
                        issues.append(
                            {
                                "severity": sev,
                                "tier": _tier(sev),
                                "message": f"Remote shell execution in {path}: {stripped}",
                            }
                        )
                    elif exec_flag_re.search(stripped):
                        sev = 2
                        issues.append(
                            {
                                "severity": sev,
                                "tier": _tier(sev),
                                "message": f"Execution flag in {path}: {stripped}",
                            }
                        )

            network_calls: List[str] = []
            for line in added:
                stripped = line.strip()
                if any(tok in stripped for tok in ("exec(", "eval(", "compile(")):
                    sev = 3
                    issues.append(
                        {
                            "severity": sev,
                            "tier": _tier(sev),
                            "message": f"Dynamic execution in {path}: {stripped}",
                        }
                    )
                if any(tok in stripped for tok in ("requests.", "urllib", "socket.")):
                    network_calls.append(stripped)
                if (
                    "subprocess.run" in stripped and "shell=True" in stripped
                ) or "os.system(" in stripped:
                    sev = 3
                    issues.append(
                        {
                            "severity": sev,
                            "tier": _tier(sev),
                            "message": f"Unsafe subprocess call in {path}: {stripped}",
                        }
                    )
                if fs_open_write_re.search(stripped) or any(
                    call in stripped for call in fs_mutation_calls
                ):
                    match = abs_path_re.search(stripped)
                    if match and "sandbox" not in match.group(1):
                        sev = 3
                        issues.append(
                            {
                                "severity": sev,
                                "tier": _tier(sev),
                                "message": f"Unsandboxed filesystem mutation in {path}: {stripped}",
                            }
                        )
                if linter_suppress_re.search(stripped):
                    sev = 2
                    issues.append(
                        {
                            "severity": sev,
                            "tier": _tier(sev),
                            "message": f"Linter suppression in {path}: {stripped}",
                        }
                    )

            for line in added:
                if line.lstrip().startswith("def "):
                    match = re.match(r"def\s+\w+\(([^)]*)\)\s*:\s*$", line.strip())
                    if match:
                        params = match.group(1)
                        if ":" not in params or "->" not in line:
                            sev = 1
                            issues.append(
                                {
                                    "severity": sev,
                                    "tier": _tier(sev),
                                    "message": f"Missing type hints in {path}: {line.strip()}",
                                }
                            )

            if len(network_calls) > self.allowed_network_calls:
                sev = self.network_call_severity
                issues.append(
                    {
                        "severity": sev,
                        "tier": _tier(sev),
                        "message": f"Network calls in {path}: {', '.join(network_calls)}",
                    }
                )

            complexity = sum(line.count(tok) for line in added for tok in complexity_tokens)
            if complexity > self.max_complexity:
                sev = 2
                issues.append(
                    {
                        "severity": sev,
                        "tier": _tier(sev),
                        "message": f"High cyclomatic complexity in {path}: score {complexity}",
                    }
                )

            # Semantic similarity checks against known unsafe patterns
            for line, pat_msg, score in find_semantic_risks(added):
                sev = 4 if score >= 0.8 else 3 if score >= 0.6 else 2
                issues.append(
                    {
                        "severity": sev,
                        "tier": _tier(sev),
                        "message": (
                            f"Semantic similarity to unsafe pattern ({pat_msg}) in {path}: {line}"
                        ),
                    }
                )

            for rule in self.rules:
                try:
                    issues.extend(rule(path, added, removed))
                except Exception:
                    continue

        # Ethics violations -------------------------------------------------
        ethics_entry = dict(context)
        ethics_entry.setdefault("generated_code", diff)
        try:
            ethics = flag_violations(ethics_entry)
        except Exception:
            ethics = {"violations": [], "severity": 0}
        for item in ethics.get("violations", []):
            msg = (
                f"{item.get('field', 'content')} contains forbidden keyword "
                f"{item.get('matched_keyword', '')} ({item.get('category', '')})"
            )
            sev = ethics.get("severity", 1)
            issues.append({
                "severity": sev,
                "tier": _tier(sev),
                "message": msg,
            })

        score = sum(issue.get("severity", 0) for issue in issues)
        tiers = Counter(issue.get("tier", "info") for issue in issues)

        return {
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "issues": issues,
            "score": score,
            "tiers": dict(tiers),
        }


def flag_alignment_risks(
    patch: str,
    metadata: Dict[str, Any],
    settings: Optional[SandboxSettings] = None,
) -> List[str]:
    """Backward compatibility wrapper returning only warning messages."""

    flagger = HumanAlignmentFlagger(settings)
    report = flagger.flag_patch(patch, metadata)
    return [item["message"] for item in report.get("issues", [])]


def flag_improvement(
    workflow_changes: List[Dict[str, Any]] | None,
    metrics: Dict[str, Any] | None,
    logs: List[Dict[str, Any]] | None,
    commit_info: Dict[str, Any] | None = None,
    settings: Optional[SandboxSettings] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Analyse prospective workflow improvements and return warnings.

    Parameters
    ----------
    workflow_changes : list of dict or None
        Each entry may contain ``file`` and ``code`` keys describing the
        proposed modification.
    metrics : dict or None
        Optional action or evaluation data.
    logs : list of dict or None
        Recent log entries to scan for violations.
    commit_info : dict or None
        Optional commit metadata containing at least ``message`` and ``author``
        keys to scan for violations.

    Returns
    -------
    dict
        Dictionary with ``ethics``, ``risk_reward`` and ``maintainability``
        warning lists.  The function never raises and swallows unexpected
        errors to avoid blocking execution.

    Examples
    --------
    >>> changes = [{"file": "util.py", "code": "print('hi')"}]
    >>> flag_improvement(changes, None, [])['maintainability'][0]['issue']
    'missing docstring'
    """

    warnings: Dict[str, List[Dict[str, Any]]] = {
        "ethics": [],
        "risk_reward": [],
        "maintainability": [],
    }

    rules = getattr(settings, "alignment_rules", None) if settings else None
    max_complexity = getattr(rules, "max_complexity_score", 10)
    allowed_network_calls = getattr(rules, "allowed_network_calls", 0)

    baseline_metrics: Dict[str, Any] = {}
    if settings is not None:
        baseline_path = getattr(settings, "alignment_baseline_metrics_path", "")
        if baseline_path:
            try:
                resolved = resolve_path(str(baseline_path))
                baseline_metrics = yaml.safe_load(resolved.read_text()) or {}
            except Exception:
                baseline_metrics = {}
    baseline_tests = baseline_metrics.get("tests")
    baseline_complexity = baseline_metrics.get("complexity")
    test_count = 0
    total_complexity = 0

    # Ethics checks -------------------------------------------------------
    for entry in logs or []:
        try:
            violations = scan_log_entry(entry)
            if violations:
                warnings["ethics"].append(
                    {"source": "log", "entry": entry.get("id"), "violations": violations}
                )
        except Exception:
            pass

    for change in workflow_changes or []:
        code = change.get("code") or change.get("content") or ""
        try:
            violations = scan_log_entry({"generated_code": code})
            if violations:
                warnings["ethics"].append(
                    {"source": "code", "file": change.get("file"), "violations": violations}
                )
        except Exception:
            pass

    if commit_info:
        try:
            message = str(commit_info.get("message", ""))
            entry = {"action_description": message}
            violations = scan_log_entry(entry)
            if violations:
                warnings["ethics"].append(
                    {
                        "source": "commit",
                        "author": commit_info.get("author"),
                        "message": message,
                        "violations": violations,
                    }
                )
        except Exception:
            pass

    # Risk / reward misalignment -----------------------------------------
    actions: List[Dict[str, Any]] = []

    def _collect(container: Any) -> None:
        if isinstance(container, list):
            iterable = container
        elif isinstance(container, dict):
            iterable = container.get("actions") or container.get("logs") or []
        else:
            iterable = []
        for item in iterable:
            if not isinstance(item, dict):
                continue
            try:
                classification = classify_action(item)
            except Exception:
                classification = {}
            action = dict(item)
            if "risk_score" not in action:
                action["risk_score"] = classification.get("risk_score")
            actions.append(action)

    _collect(logs)
    _collect(metrics)
    try:
        misaligned = check_risk_reward_alignment(actions)
        if misaligned:
            warnings["risk_reward"].extend(misaligned)
    except Exception:
        pass

    # Maintainability heuristics -----------------------------------------
    has_tests = False
    fs_mutation_calls = (
        "os.remove(",
        "os.unlink(",
        "os.rename(",
        "os.rmdir(",
        "shutil.rmtree(",
        "shutil.move(",
        "shutil.copyfile(",
    )
    fs_open_write_re = re.compile(r"open\([^,]+,[^)]*['\"](?:w|a|x)['\"]")
    abs_path_re = re.compile(r"['\"](/[^'\"]*)")
    linter_suppress_re = re.compile(r"#\s*(?:noqa|pylint:\s*disable|pragma:\s*no\s*cover)")

    for change in workflow_changes or []:
        file_path = change.get("file") or ""
        code = change.get("code") or change.get("content") or ""
        diff_text = change.get("diff") or ""
        if file_path.startswith("tests") or file_path.endswith("_test" + ".py") or file_path.startswith("test_"):
            has_tests = True
            test_count += 1
        if file_path.endswith(".py"):
            lines = code.splitlines()
            for line in lines:
                if (
                    "subprocess.Popen" in line
                    or ("subprocess.run" in line and "shell=True" in line)
                    or "os.system(" in line
                ):
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "unsafe subprocess call",
                            "snippet": line.strip(),
                        }
                    )
                if fs_open_write_re.search(line) or any(
                    call in line for call in fs_mutation_calls
                ):
                    match = abs_path_re.search(line)
                    if match and "sandbox" not in match.group(1):
                        warnings["maintainability"].append(
                            {
                                "file": file_path,
                                "issue": "unsandboxed filesystem mutation",
                                "snippet": line.strip(),
                            }
                        )

            stripped = code.lstrip()
            if not (stripped.startswith('"""') or stripped.startswith("'''")):
                warnings["maintainability"].append({"file": file_path, "issue": "missing docstring"})

            # Detect removal of type hints using diff information
            if diff_text:
                removed_hints: List[str] = []
                for line in diff_text.splitlines():
                    if line.startswith("-") and not line.startswith("---"):
                        content = line[1:]
                        snippet = content
                        if content.strip().startswith("def") and content.rstrip().endswith(":"):
                            snippet += "\n    pass"
                        try:
                            snippet_tree = ast.parse(snippet)
                        except SyntaxError:
                            continue
                        has_annotation = False
                        for node in ast.walk(snippet_tree):
                            if isinstance(node, ast.AnnAssign) and node.annotation is not None:
                                has_annotation = True
                                break
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if node.returns is not None:
                                    has_annotation = True
                                    break
                                for arg in (
                                    list(node.args.args)
                                    + list(getattr(node.args, "posonlyargs", []))
                                    + list(node.args.kwonlyargs)
                                ):
                                    if getattr(arg, "annotation", None) is not None:
                                        has_annotation = True
                                        break
                                if has_annotation:
                                    break
                        if has_annotation:
                            removed_hints.append(content.strip())
                if removed_hints:
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "removed type hints",
                            "snippets": removed_hints,
                        }
                    )

            try:
                tree = ast.parse(code)
            except SyntaxError:
                tree = None

            if tree is not None:
                def _compute_complexity(func: ast.AST) -> int:
                    complexity = 1
                    for node in ast.walk(func):
                        if isinstance(
                            node,
                            (
                                ast.If,
                                ast.For,
                                ast.AsyncFor,
                                ast.While,
                                ast.With,
                                ast.AsyncWith,
                                ast.IfExp,
                                ast.ListComp,
                                ast.DictComp,
                                ast.SetComp,
                                ast.GeneratorExp,
                            ),
                        ):
                            complexity += 1
                        elif isinstance(node, ast.BoolOp):
                            complexity += len(getattr(node, "values", [])) - 1
                        elif isinstance(node, ast.Try):
                            complexity += len(node.handlers)
                            if node.orelse:
                                complexity += 1
                            if node.finalbody:
                                complexity += 1
                    return complexity

                def _has_type_hints(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
                    args = (
                        list(func.args.args)
                        + list(getattr(func.args, "posonlyargs", []))
                        + list(func.args.kwonlyargs)
                    )
                    if func.args.vararg:
                        args.append(func.args.vararg)
                    if func.args.kwarg:
                        args.append(func.args.kwarg)
                    if any(a.annotation is None for a in args):
                        return False
                    return func.returns is not None

                complex_functions = []
                missing_hints = []
                single_letters: set[str] = set()
                network_calls: List[str] = []
                exec_calls: List[str] = []
                broad_excepts: List[int] = []
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        score = _compute_complexity(node)
                        total_complexity += score
                        if score > max_complexity:
                            complex_functions.append({"name": node.name, "score": score})
                        if not _has_type_hints(node):
                            missing_hints.append(node.name)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        if len(node.id) == 1 and node.id.isalpha():
                            single_letters.add(node.id)
                    elif isinstance(node, ast.Call):
                        func = node.func
                        name = ""
                        if isinstance(func, ast.Name):
                            name = func.id
                        elif isinstance(func, ast.Attribute):
                            base = func.value
                            if isinstance(base, ast.Name):
                                name = f"{base.id}.{func.attr}"
                        if name in {"exec", "eval"}:
                            exec_calls.append(name)
                        elif name.split(".")[0] in {"requests", "urllib", "http", "socket"}:
                            network_calls.append(name)
                    elif isinstance(node, ast.ExceptHandler) and node.type is None:
                        broad_excepts.append(getattr(node, "lineno", 0))

                if complex_functions:
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "high cyclomatic complexity",
                            "functions": complex_functions,
                        }
                    )
                if missing_hints:
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "missing type hints",
                            "functions": missing_hints,
                        }
                    )
                if single_letters:
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "obfuscated variable names",
                            "names": sorted(single_letters),
                        }
                    )
                if exec_calls:
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "direct exec or eval call",
                            "calls": exec_calls,
                        }
                    )
                if len(network_calls) > allowed_network_calls:
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "network call",
                            "calls": network_calls,
                        }
                    )
                if broad_excepts:
                    warnings["maintainability"].append(
                        {
                            "file": file_path,
                            "issue": "broad exception handler",
                            "lines": broad_excepts,
                        }
                    )

    if not has_tests:
        warnings["maintainability"].append({"issue": "no tests provided"})

    if baseline_tests is not None and test_count < baseline_tests:
        warnings["maintainability"].append(
            {
                "issue": f"test count decreased (baseline {baseline_tests}, current {test_count})",
            }
        )
    if baseline_complexity is not None and total_complexity > baseline_complexity:
        warnings["maintainability"].append(
            {
                "issue": f"complexity increased (baseline {baseline_complexity}, current {total_complexity})",
            }
        )

    return warnings

def flag_alignment_issues(
    diff_data: Dict[str, Dict[str, List[str]]],
    settings: Optional[SandboxSettings] = None,
) -> List[Dict[str, str]]:
    """Return a list of alignment findings for given *diff_data*.

    Parameters
    ----------
    diff_data : dict
        Mapping of file paths to dictionaries with ``"added"`` and
        ``"removed"`` line lists.
    """

    findings: List[Dict[str, str]] = []
    risky_tokens = ("eval(", "exec(", "compile(")  # fallback if AST parsing fails
    dynamic_tokens = ("__import__(", "importlib.import_module(")
    complexity_tokens = ("if", "for", "while", "and", "or", "try", "except", "elif")
    rules = getattr(settings, "alignment_rules", None) if settings else None
    max_complexity = getattr(rules, "max_complexity_score", 10)
    max_comment_drop = getattr(rules, "max_comment_density_drop", 0.1)
    fs_mutation_calls = (
        "os.remove(",
        "os.unlink(",
        "os.rename(",
        "os.rmdir(",
        "shutil.rmtree(",
        "shutil.move(",
        "shutil.copyfile(",
    )
    fs_open_write_re = re.compile(r"open\([^,]+,[^)]*['\"](?:w|a|x)['\"]")
    abs_path_re = re.compile(r"['\"](/[^'\"]*)")
    linter_suppress_re = re.compile(r"#\s*(?:noqa|pylint:\s*disable|pragma:\s*no\s*cover)")

    for path, changes in diff_data.items():
        added_lines = changes.get("added", [])
        removed_lines = changes.get("removed", [])
        joined = "\n".join(added_lines)

        try:
            tree = ast.parse(joined) if joined.strip() else None
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = ""
                    module_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                        if isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                    call_path = f"{module_name}.{func_name}" if module_name else func_name
                    if func_name in {"exec", "eval", "compile"}:
                        snippet = ast.get_source_segment(joined, node) or func_name
                        findings.append(
                            {
                                "category": "risky_construct",
                                "location": path,
                                "snippet": snippet.strip(),
                            }
                        )
                    elif func_name == "__import__" or call_path.startswith("importlib."):
                        snippet = ast.get_source_segment(joined, node) or call_path
                        findings.append(
                            {
                                "category": "dynamic_import",
                                "location": path,
                                "snippet": snippet.strip(),
                            }
                        )
                elif isinstance(node, ast.ExceptHandler) and node.type is None:
                    findings.append(
                        {
                            "category": "risky_construct",
                            "location": path,
                            "snippet": "bare except",
                        }
                    )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    missing = node.returns is None or any(
                        arg.annotation is None
                        for arg in node.args.args + node.args.kwonlyargs
                    )
                    if node.args.vararg and node.args.vararg.annotation is None:
                        missing = True
                    if node.args.kwarg and node.args.kwarg.annotation is None:
                        missing = True
                    if missing:
                        snippet = ast.get_source_segment(joined, node) or node.name
                        first_line = snippet.strip().splitlines()[0]
                        findings.append(
                            {
                                "category": "missing_type_hints",
                                "location": path,
                                "snippet": first_line,
                            }
                        )
        else:
            for line in added_lines:
                if any(tok in line for tok in risky_tokens):
                    findings.append(
                        {
                            "category": "risky_construct",
                            "location": path,
                            "snippet": line.strip(),
                        }
                    )
                if any(tok in line for tok in dynamic_tokens):
                    findings.append(
                        {
                            "category": "dynamic_import",
                            "location": path,
                            "snippet": line.strip(),
                        }
                    )
                if line.lstrip().startswith("def "):
                    match = re.match(r"def\s+\w+\(([^)]*)\)\s*:\s*$", line.strip())
                    if match:
                        params = match.group(1)
                        if ":" not in params and "->" not in line:
                            findings.append(
                                {
                                    "category": "missing_type_hints",
                                    "location": path,
                                    "snippet": line.strip(),
                                }
                            )

        for line in added_lines:
            # Unsafe subprocess usage
            if (
                "subprocess.Popen" in line
                or ("subprocess.run" in line and "shell=True" in line)
                or "os.system(" in line
            ):
                findings.append(
                    {
                        "category": "unsafe_subprocess",
                        "location": path,
                        "snippet": line.strip(),
                    }
                )

            # Direct file-system mutations outside sandbox paths
            if fs_open_write_re.search(line) or any(
                call in line for call in fs_mutation_calls
            ):
                match = abs_path_re.search(line)
                if match and "sandbox" not in match.group(1):
                    findings.append(
                        {
                            "category": "filesystem_mutation",
                            "location": path,
                            "snippet": line.strip(),
                        }
                    )

            # Linter directive suppression
            if linter_suppress_re.search(line):
                findings.append(
                    {
                        "category": "linter_suppression",
                        "location": path,
                        "snippet": line.strip(),
                    }
                )

        # Missing logging
        if any(line.lstrip().startswith("def ") for line in added_lines) and not any(
            "logging" in line or "logger" in line for line in added_lines
        ):
            findings.append({
                "category": "missing_logging",
                "location": path,
                "snippet": "Function added without logging",
            })

        # Cyclomatic complexity heuristic
        complexity = sum(line.count(tok) for line in added_lines for tok in complexity_tokens)
        if complexity > max_complexity:
            findings.append({
                "category": "high_complexity",
                "location": path,
                "snippet": f"complexity score {complexity}",
            })

        # Removed comments
        for line in removed_lines:
            if line.lstrip().startswith("#"):
                findings.append(
                    {
                        "category": "comment_removed",
                        "location": path,
                        "snippet": line.strip(),
                    }
                )

        # Opacity: comment density decrease
        removed_comments = sum(1 for l in removed_lines if l.lstrip().startswith("#"))
        added_comments = sum(1 for l in added_lines if l.lstrip().startswith("#"))
        if removed_comments > 0:
            removed_ratio = removed_comments / max(len(removed_lines), 1)
            added_ratio = added_comments / max(len(added_lines), 1)
            drop = removed_ratio - added_ratio
            if drop > max_comment_drop:
                findings.append({
                    "category": "opacity",
                    "location": path,
                    "snippet": "comment density decreased",
                })

        # Opacity: single-character identifiers introduced
        single_char_re = re.compile(r"\b[A-Za-z]\b")
        added_single = sum(
            len(single_char_re.findall(l)) for l in added_lines if not l.lstrip().startswith("#")
        )
        removed_single = sum(
            len(single_char_re.findall(l))
            for l in removed_lines
            if not l.lstrip().startswith("#")
        )
        if removed_lines and added_single > removed_single and added_single > 0:
            findings.append(
                {
                    "category": "opacity",
                    "location": path,
                    "snippet": "single-character identifiers introduced",
                }
            )

        # Ethics violations
        try:
            result = flag_violations({"generated_code": joined})
            for item in result.get("violations", []):
                findings.append({
                    "category": f"ethics:{item.get('category', '')}",
                    "location": path,
                    "snippet": item.get("matched_keyword", ""),
                })
        except Exception:
            pass

    return findings


def _collect_diff_data(before: Path, after: Path) -> Dict[str, Dict[str, List[str]]]:
    """Return diff mapping between *before* and *after* directories."""

    diff: Dict[str, Dict[str, List[str]]] = {}
    for new_path in after.rglob("*"):
        if not new_path.is_file():
            continue
        rel = new_path.relative_to(after)
        old_path = before / rel
        added: List[str] = []
        removed: List[str] = []
        if old_path.exists():
            before_lines = old_path.read_text().splitlines()
            after_lines = new_path.read_text().splitlines()
            for line in difflib.unified_diff(before_lines, after_lines, lineterm=""):
                if line.startswith("+") and not line.startswith("+++"):
                    added.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    removed.append(line[1:])
        else:
            added = new_path.read_text().splitlines()
        diff[str(rel)] = {"added": added, "removed": removed}
    return diff


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for scanning directory diffs.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments; if ``None`` ``sys.argv`` is used.

    Examples
    --------
    Run the checker from the shell by passing two directory snapshots::

        $ python human_alignment_flagger.py repo/before repo/after

    Programmatic invocation mirrors the command line interface::

        >>> main(["repo/before", "repo/after"])

    A JSON report describing any findings is printed to stdout.
    """

    args = argv or sys.argv[1:]
    if len(args) != 2:
        print("Usage: python human_alignment_flagger.py before_dir after_dir", file=sys.stderr)
        raise SystemExit(1)
    before_dir, after_dir = map(Path, args)
    diff_data = _collect_diff_data(before_dir, after_dir)
    findings = flag_alignment_issues(diff_data)
    print(json.dumps(findings, indent=2))


__all__ = [
    "HumanAlignmentFlagger",
    "flag_alignment_risks",
    "flag_improvement",
    "flag_alignment_issues",
    "Rule",
]


if __name__ == "__main__":
    main()

