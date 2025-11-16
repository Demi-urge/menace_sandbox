from pathlib import Path
from contextlib import contextmanager
import os
import subprocess
import shutil
import time
import logging

"""Shared models repository configuration."""

MODELS_REPO_URL = os.getenv(
    "MODELS_REPO_URL", "https://github.com/Demi-urge/models"
)
MODELS_REPO_PATH = Path(os.getenv("MODELS_REPO_PATH", "models_repo"))
MODELS_REPO_PUSH_URL = os.getenv("MODELS_REPO_PUSH_URL")

# Marker file indicating a model build is in progress
ACTIVE_MODEL_FILE = MODELS_REPO_PATH / ".active_model"

logger = logging.getLogger(__name__)


def ensure_models_repo() -> Path:
    """Clone the models repo if ``MODELS_REPO_PATH`` does not exist."""
    if not MODELS_REPO_PATH.exists():
        subprocess.run(["git", "clone", MODELS_REPO_URL, str(MODELS_REPO_PATH)], check=True)
    return MODELS_REPO_PATH


def clone_to_new_repo(model_id: int) -> Path:
    """Clone the entire models repo into ``<model_id>`` next to it.

    When ``MODELS_REPO_PUSH_URL`` is set the clone is pushed to a remote
    repository named after the model ID.
    """

    dest = MODELS_REPO_PATH.parent / str(model_id)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", str(MODELS_REPO_PATH), str(dest)], check=True)
    if MODELS_REPO_PUSH_URL:
        remote_url = f"{MODELS_REPO_PUSH_URL.rstrip('/')}/{model_id}"
        subprocess.run(
            ["git", "remote", "set-url", "origin", remote_url],
            cwd=dest,
            check=True,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", "HEAD"],
            cwd=dest,
            check=True,
        )
    return dest


@contextmanager
def model_build_lock(model_id: int, poll_interval: float = 0.1):
    """Create and remove ``.active_model`` for the build duration."""

    while ACTIVE_MODEL_FILE.exists():
        time.sleep(poll_interval)
    ACTIVE_MODEL_FILE.write_text(str(model_id))
    try:
        yield
    finally:
        try:
            if ACTIVE_MODEL_FILE.exists() and ACTIVE_MODEL_FILE.read_text() == str(model_id):
                ACTIVE_MODEL_FILE.unlink()
        except OSError as exc:
            logger.warning(
                "Failed to remove active model file",
                extra={"model_id": model_id, "path": str(ACTIVE_MODEL_FILE)},
                exc_info=exc,
            )
        except Exception:
            logger.exception(
                "Unexpected error removing active model file",
                extra={"model_id": model_id, "path": str(ACTIVE_MODEL_FILE)},
            )
            raise
