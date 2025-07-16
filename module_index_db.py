import json
import os
from pathlib import Path
from typing import Dict


class ModuleIndexDB:
    """Persist mapping of module names to hashed indices."""

    def __init__(self, path: Path | str = "module_map.json") -> None:
        self.path = Path(path)
        if self.path.exists():
            try:
                self._map: Dict[str, int] = json.loads(self.path.read_text())
            except Exception:
                self._map = {}
        else:
            self._map = {}

    # --------------------------------------------------------------
    def get(self, name: str) -> int:
        """Return persistent index for ``name`` creating it if needed."""
        if name not in self._map:
            self._map[name] = abs(hash(name)) % 1000
            self.save()
        return int(self._map[name])

    # --------------------------------------------------------------
    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._map, fh, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass


__all__ = ["ModuleIndexDB"]
