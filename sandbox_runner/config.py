"""Configuration for sandbox runner."""

from pathlib import Path

# Remote repository used by the visual agent service
SANDBOX_REPO_URL = "https://github.com/Demi-urge/menace_sandbox"

# Local checkout path of ``SANDBOX_REPO_URL``. The sandbox runner assumes this
# repository already exists and will operate directly on it instead of cloning
# a fresh copy for each run.
SANDBOX_REPO_PATH = Path(__file__).resolve().parents[1]


