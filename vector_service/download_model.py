"""Download and bundle the fallback embedding model.

This utility fetches a small DistilRoBERTa model from Hugging Face and packs
the minimal files into ``vector_service/minilm/tiny-distilroberta-base.tar.xz``.
Run the module as a script to refresh the archive:

```
python -m vector_service.download_model
```
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tarfile
import tempfile

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


def bundle(dest: Path) -> None:
    """Download ``MODEL_ID`` and write a compressed archive to ``dest``."""

    model_dir = Path(snapshot_download(MODEL_ID))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in FILES:
            shutil.copy(model_dir / name, tmp / name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w:xz") as tar:
            for name in FILES:
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

