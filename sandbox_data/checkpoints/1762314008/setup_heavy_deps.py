import argparse
import importlib
import subprocess
import sys
import shutil
from pathlib import Path
import typing
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


def _load_ensure_playwright() -> typing.Callable[[], bool]:
    """Return a best-effort Playwright verification function."""

    candidates = (
        "neurosales.dynamic_harvest",
        "neurosales.neurosales.dynamic_harvest",
        "dynamic_harvest",
    )

    for module_name in candidates:
        try:  # pragma: no cover - import paths differ between layouts
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        except Exception:  # pragma: no cover - defensive best effort
            continue

        ensure = getattr(module, "ensure_playwright_browsers", None)
        if callable(ensure):
            return typing.cast(typing.Callable[[], bool], ensure)

    return _missing_playwright_check


ensure_playwright_browsers = _load_ensure_playwright()

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
        if reqs.is_file():
            print(f"Installing packages from {reqs}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(reqs)]
            )
        else:
            print(
                f"Requirements file {reqs} not found; skipping package installation"
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
