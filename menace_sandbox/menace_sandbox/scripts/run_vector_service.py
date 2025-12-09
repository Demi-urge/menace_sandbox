#!/usr/bin/env python3
"""
Standalone vector bootstrap runner invoked during Menace startup.
This does NOT run the HTTP API — it only triggers:
- db_index_load
- retriever_hydration
- vector_seeding
and writes readiness timestamps for bootstrap_readiness.
"""

from menace_sandbox.vector_service.vector_runtime import initialize_vector_service


def main() -> None:
    # Perform vector runtime bootstrap synchronously.
    initialize_vector_service()


if __name__ == "__main__":
    main()
