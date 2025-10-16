import argparse
import subprocess
import sys
import shutil
from pathlib import Path
try:  # pragma: no cover - optional dependency in minimal envs
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully for tests
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """Stub that mirrors :func:`dotenv.load_dotenv` when dependency is absent."""

        return False


def _missing_playwright_check() -> bool:
    """Fallback when ``dynamic_harvest`` cannot be imported locally."""

    print(
        "Playwright verification skipped: neurosales.dynamic_harvest unavailable"
    )
    return False


try:  # pragma: no cover - import path differs when package installed
    from neurosales.dynamic_harvest import ensure_playwright_browsers
except ModuleNotFoundError:  # pragma: no cover - local source layout
    try:
        from neurosales.neurosales.dynamic_harvest import (  # type: ignore
            ensure_playwright_browsers,
        )
    except ModuleNotFoundError:
        ensure_playwright_browsers = _missing_playwright_check

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup heavy dependencies")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download embedding weights without installing packages",
    )
    args = parser.parse_args()

    if not args.download_only:
        root = Path(__file__).resolve().parents[1]
        reqs = root / "requirements-extra.txt"
        print(f"Installing packages from {reqs}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(reqs)]
        )

        if shutil.which("playwright"):
            print("Installing Playwright browsers...")
            subprocess.call(["playwright", "install", "--with-deps"])
            if ensure_playwright_browsers():
                print("Playwright installation verified")
            else:
                print(
                    "WARNING: Playwright browsers missing. Run 'playwright install --with-deps'"
                )
        else:
            print("Playwright not found; skipping browser install")

    try:
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import login
        import os

        login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        print("Prefetching embedding model weights...")
        SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Could not load SentenceTransformer: {exc}")

    print("Heavy dependencies are ready")


if __name__ == "__main__":
    main()
