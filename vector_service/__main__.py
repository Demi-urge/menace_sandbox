from __future__ import annotations

"""Entry point for running the vector database service.

This module allows the service to be started via ``python -m vector_service``.
Configuration is intentionally minimal; host and port or a Unix domain socket
can be specified through environment variables.
"""

import os
import uvicorn


def main() -> None:
    target = os.environ.get("VECTOR_SERVICE_SOCKET")
    if target:
        config = uvicorn.Config(
            "vector_service.vector_database_service:app",
            uds=target,
            log_level="info",
        )
    else:
        host = os.environ.get("VECTOR_SERVICE_HOST", "127.0.0.1")
        port = int(os.environ.get("VECTOR_SERVICE_PORT", "8000"))
        config = uvicorn.Config(
            "vector_service.vector_database_service:app",
            host=host,
            port=port,
            log_level="info",
        )
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
