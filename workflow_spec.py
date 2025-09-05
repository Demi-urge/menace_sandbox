from __future__ import annotations

"""Utilities for serialising lightweight workflow specifications.

The helpers in this module are intentionally small.  They convert a sequence of
``WorkflowStep`` objects – simple containers describing a module and its IO –
into a mapping that can readily be dumped to JSON or YAML and persisted on
disk.

``to_spec``
    Transform a list of steps into ``{"steps": [...]}`` where each step
    contains ``module``, ``inputs``, ``outputs``, ``files`` and ``globals``
    fields.

``save_spec``
    Write such a specification to ``<name>.workflow.json`` or
    ``<name>.workflow.yaml`` inside a ``workflows`` directory.
"""

from dataclasses import asdict, is_dataclass
import json
import difflib
from pathlib import Path
from typing import Any, Iterable
from datetime import datetime, timezone
from uuid import uuid4
from dynamic_path_router import resolve_path


MAX_ID_ATTEMPTS = 5

try:  # Optional dependency for YAML output
    import yaml  # type: ignore
except Exception:  # pragma: no cover - YAML support is optional
    yaml = None  # type: ignore


def _to_list(value: Iterable[Any] | None) -> list[str]:
    """Normalise ``value`` to a list of strings."""

    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def to_spec(steps: list[Any]) -> dict:
    """Return a JSON/YAML‑serialisable specification.

    Parameters
    ----------
    steps:
        Sequence of :class:`WorkflowStep` instances or ``dict``s describing
        workflow steps.  Each step is expected to expose the attributes
        ``module``, ``inputs`` and ``outputs``.  Optional ``files`` or
        ``files_read``/``files_written`` and ``globals`` attributes are also
        honoured.
    """

    spec_steps = []

    for step in steps:
        if is_dataclass(step):
            data = asdict(step)
        elif isinstance(step, dict):
            data = dict(step)
        else:  # pragma: no cover - best effort for simple objects
            data = step.__dict__

        files = _to_list(data.get("files"))
        if not files:
            files = _to_list(data.get("files_read")) + _to_list(data.get("files_written"))

        spec_steps.append(
            {
                "module": str(data.get("module") or data.get("name", "")),
                "inputs": sorted(_to_list(data.get("inputs"))),
                "outputs": sorted(_to_list(data.get("outputs"))),
                "files": sorted(files),
                "globals": sorted(_to_list(data.get("globals"))),
            }
        )

    return {"steps": spec_steps}


def validate_metadata(metadata: dict) -> None:
    """Validate the structure of a metadata mapping.

    ``workflow_id`` and ``created_at`` must be present and strings. Optional
    fields are type‑checked when provided.
    """

    assert isinstance(metadata, dict), "metadata must be a dictionary"

    workflow_id = metadata.get("workflow_id")
    assert isinstance(workflow_id, str) and workflow_id, "workflow_id is required"

    created_at = metadata.get("created_at")
    assert isinstance(created_at, str) and created_at, "created_at is required"

    optional_fields = {
        "parent_id": (str, type(None)),
        "mutation_description": str,
        "summary_path": str,
        "diff_path": str,
    }
    for field, types in optional_fields.items():
        if field in metadata and metadata[field] is not None:
            assert isinstance(metadata[field], types), f"{field} must be {types}"


def save_spec(spec: dict, path: Path, *, summary_path: Path | str | None = None) -> Path:
    """Persist ``spec`` to ``path`` within a ``workflows`` directory.

    The extension of ``path`` determines the output format: ``.workflow.json``
    results in JSON while ``.workflow.yaml``/``.workflow.yml`` produces YAML.
    Regardless of the provided location the file is written beneath a
    ``workflows`` folder, which is created if necessary.  When a ``parent_id``
    is provided in the metadata, a unified diff against the parent workflow is
    written alongside the specification and its path recorded in the metadata
    for quick access.
    Parameters
    ----------
    spec:
        Mapping describing the workflow specification.
    path:
        Destination for the saved workflow.
    summary_path:
        Optional path to a workflow summary JSON file stored in the metadata.
    """

    # Ensure metadata block is present with required fields
    metadata = dict(spec.get("metadata") or {})
    if summary_path is not None:
        spath = Path(summary_path)
        if not spath.is_absolute():
            spath = Path(resolve_path(".")) / spath
        metadata["summary_path"] = str(spath)
    elif "summary_path" in metadata and metadata["summary_path"] is not None:
        spath = Path(str(metadata["summary_path"]))
        if not spath.is_absolute():
            spath = Path(resolve_path(".")) / spath
        metadata["summary_path"] = str(spath)
    metadata.setdefault("workflow_id", str(uuid4()))
    metadata.setdefault("parent_id", None)
    metadata.setdefault("mutation_description", "")
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    path = Path(path)
    if not path.is_absolute():
        path = Path(resolve_path(".")) / path
    parent = path.parent
    if parent.name != "workflows":
        parent = parent / "workflows"
    parent.mkdir(parents=True, exist_ok=True)

    # Ensure workflow_id uniqueness within the target directory
    existing_ids = set()
    for candidate in parent.glob("*.workflow.json"):
        try:
            data = json.loads(candidate.read_text())
            validate_metadata(data.get("metadata") or {})
        except Exception:  # pragma: no cover - ignore bad files
            continue
        existing_ids.add(data["metadata"]["workflow_id"])

    attempts = 0
    while metadata["workflow_id"] in existing_ids:
        if attempts >= MAX_ID_ATTEMPTS:
            raise RuntimeError("Unable to generate unique workflow_id")
        metadata["workflow_id"] = str(uuid4())
        attempts += 1

    validate_metadata(metadata)

    spec = dict(spec)
    spec["metadata"] = metadata

    name = path.name
    out_path = parent / name

    # Compare against parent workflow when available
    parent_id = metadata.get("parent_id")
    if parent_id:
        parent_path = parent / f"{parent_id}.workflow.json"
        if not parent_path.exists():
            for candidate in parent.glob("*.workflow.json"):
                try:
                    data = json.loads(candidate.read_text())
                    validate_metadata(data.get("metadata") or {})
                except Exception:  # pragma: no cover - ignore corrupt files
                    continue
                if data.get("metadata", {}).get("workflow_id") == parent_id:
                    parent_path = candidate
                    break
        if parent_path.exists():
            new_lines = json.dumps(spec, indent=2, sort_keys=True).splitlines()
            parent_data = json.loads(parent_path.read_text())
            validate_metadata(parent_data.get("metadata") or {})
            parent_lines = json.dumps(parent_data, indent=2, sort_keys=True).splitlines()
            diff_lines = list(
                difflib.unified_diff(
                    parent_lines,
                    new_lines,
                    fromfile=parent_path.name,
                    tofile=name,
                    lineterm="",
                )
            )
            diff_file = parent / f"{metadata['workflow_id']}.diff"
            diff_file.write_text("\n".join(diff_lines), encoding="utf-8")
            metadata["diff_path"] = str(diff_file)

    if name.endswith(".workflow.json"):
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2, sort_keys=True)
    elif name.endswith((".workflow.yaml", ".workflow.yml")):
        if yaml is None:  # pragma: no cover - YAML optional
            raise RuntimeError("PyYAML is required for YAML output")
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(spec, fh, sort_keys=False)  # type: ignore[arg-type]
    else:
        raise ValueError("path must end with .workflow.json or .workflow.yaml")

    return out_path


# Backwards compatibility -------------------------------------------------
# Some modules still import ``save``; keep it as an alias.
save = save_spec

__all__ = ["to_spec", "save_spec", "save", "validate_metadata"]
