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


MODEL_ID = "sshleifer/tiny-distilroberta-base"
FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]


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
    """Ensure the bundled model archive exists at ``dest``.

    If ``dest`` is ``None``, the default location inside the package is used.
    The model is downloaded from Hugging Face when missing.
    """

    dest = dest or Path(__file__).with_name("minilm") / "tiny-distilroberta-base.tar.xz"
    if not dest.exists():
        bundle(dest)
    return dest


def main() -> None:  # pragma: no cover - helper script
    dest = Path(__file__).with_name("minilm") / "tiny-distilroberta-base.tar.xz"
    bundle(dest)


if __name__ == "__main__":  # pragma: no cover
    main()

