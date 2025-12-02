"""Download and bundle the fallback embedding model.

This utility fetches a small DistilRoBERTa model from Hugging Face and packs
the minimal files into ``vector_service/minilm/tiny-distilroberta-base.tar.xz``.
Run the module as a script to refresh the archive:

```
python -m vector_service.download_model
```
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from dynamic_path_router import resolve_path

MODEL_ID = "sshleifer/tiny-distilroberta-base"
FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]

MODEL_ARCHIVE = resolve_path("vector_service/minilm") / "tiny-distilroberta-base.tar.xz"


def _check_cancelled(
    *, stop_event, budget_check, deadline: float | None
) -> None:  # pragma: no cover - trivial helper
    if stop_event is not None and stop_event.is_set():
        raise TimeoutError("embedding model download cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        if stop_event is not None:
            stop_event.set()
        raise TimeoutError("embedding model download cancelled (timeout)")
    if budget_check is not None:
        budget_check(stop_event)


def bundle(
    dest: Path,
    *,
    stop_event=None,
    budget_check=None,
    timeout: float | None = None,
) -> None:
    """Download ``MODEL_ID`` and write a compressed archive to ``dest``."""

    deadline = time.monotonic() + timeout if timeout is not None else None

    def _check() -> None:
        _check_cancelled(stop_event=stop_event, budget_check=budget_check, deadline=deadline)

    def _progress_callback(*_args: object) -> None:
        _check()

    _check()
    model_dir = Path(snapshot_download(MODEL_ID, progress_callback=_progress_callback))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in FILES:
            _check()
            shutil.copy(model_dir / name, tmp / name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w:xz") as tar:
            for name in FILES:
                _check()
                tar.add(tmp / name, arcname=name)


def ensure_model(dest: Path | None = None) -> Path:
    """Return the path to the bundled model archive.

    The archive must already exist locally. If it is missing, a descriptive
    :class:`FileNotFoundError` is raised instructing the operator to download
    it manually.
    """

    dest = dest or MODEL_ARCHIVE
    if not dest.exists():
        raise FileNotFoundError(
            f"{dest} is missing. Run `python -m vector_service.download_model` "
            "to download it from Hugging Face."
        )
    return dest


def main() -> None:  # pragma: no cover - helper script
    dest = MODEL_ARCHIVE
    bundle(dest)


if __name__ == "__main__":  # pragma: no cover
    main()

