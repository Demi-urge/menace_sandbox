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
from pathlib import Path
from typing import Any, Iterable
from datetime import datetime, timezone
from uuid import uuid4

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
        Sequence of :class:`WorkflowStep` instances or ``dict``\ s describing
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


def save_spec(spec: dict, path: Path) -> Path:
    """Persist ``spec`` to ``path`` within a ``workflows`` directory.

    The extension of ``path`` determines the output format: ``.workflow.json``
    results in JSON while ``.workflow.yaml``/``.workflow.yml`` produces YAML.
    Regardless of the provided location the file is written beneath a
    ``workflows`` folder, which is created if necessary.
    """

    # Ensure metadata block is present with required fields
    metadata = dict(spec.get("metadata") or {})
    metadata.setdefault("workflow_id", str(uuid4()))
    metadata.setdefault("parent_id", None)
    metadata.setdefault("mutation_description", "")
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    spec = dict(spec)
    spec["metadata"] = metadata

    path = Path(path)
    parent = path.parent
    if parent.name != "workflows":
        parent = parent / "workflows"
    parent.mkdir(parents=True, exist_ok=True)

    name = path.name
    if name.endswith(".workflow.json"):
        out_path = parent / name
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2, sort_keys=True)
    elif name.endswith(('.workflow.yaml', '.workflow.yml')):
        if yaml is None:  # pragma: no cover - YAML optional
            raise RuntimeError("PyYAML is required for YAML output")
        out_path = parent / name
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(spec, fh, sort_keys=False)  # type: ignore[arg-type]
    else:
        raise ValueError("path must end with .workflow.json or .workflow.yaml")

    return out_path


# Backwards compatibility -------------------------------------------------
# Some modules still import ``save``; keep it as an alias.
save = save_spec

__all__ = ["to_spec", "save_spec", "save"]

