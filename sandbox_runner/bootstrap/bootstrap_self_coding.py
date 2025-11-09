"""Compat module for running the self-coding bootstrap inside the sandbox package."""

from menace_sandbox.bootstrap_self_coding import (
    bootstrap_self_coding,
    main,
    purge_stale_files,
)

__all__ = [
    "bootstrap_self_coding",
    "purge_stale_files",
    "main",
]

if __name__ == "__main__":
    main()
