"""Shim for legacy module path.

Older launchers still invoke
``python -m menace_sandbox.menace_sandbox.start_autonomous_sandbox``. After the
package flattening, the real entrypoint lives at
``menace_sandbox.start_autonomous_sandbox``. This module forwards imports and
``python -m`` execution to the relocated implementation so both paths continue to
work.
"""

from __future__ import annotations

from .. import start_autonomous_sandbox as _root_module

main = _root_module.main


if __name__ == "__main__":
    main()
