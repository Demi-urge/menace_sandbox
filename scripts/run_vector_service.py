#!/usr/bin/env python3
"""Run the vector service API as a standalone server."""
from __future__ import annotations

import os
import uvicorn

from context_builder_util import create_context_builder
import vector_service_api


def main() -> None:
    """Initialise the app and start Uvicorn."""
    vector_service_api.create_app(create_context_builder())
    uds = os.environ.get("VECTOR_SERVICE_SOCKET")
    if uds:
        uvicorn.run(vector_service_api.app, uds=uds, log_level="info")
    else:
        host = os.environ.get("VECTOR_SERVICE_HOST", "0.0.0.0")
        port = int(os.environ.get("VECTOR_SERVICE_PORT", "8000"))
        uvicorn.run(vector_service_api.app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
