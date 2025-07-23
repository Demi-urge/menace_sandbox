from __future__ import annotations

"""Registry mapping bot capabilities to implementing bots."""

from collections import defaultdict
from typing import Iterable, List, Dict
import json
import os
import threading
from logging_utils import get_logger

logger = get_logger(__name__)


class CapabilityRegistry:
    """Simple lookup of bot capabilities with optional persistence."""

    def __init__(self, path: str | None = None, *, auto_persist: bool = True) -> None:
        self._caps: Dict[str, List[str]] = defaultdict(list)
        self._path = path
        self._auto_persist = auto_persist
        self._lock = threading.RLock()
        if self._path and os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for cap, bots in data.items():
                    if isinstance(bots, list):
                        self._caps[cap].extend(str(b) for b in bots)
            except Exception as exc:
                logger.error("failed to load registry from %s: %s", self._path, exc)

    def register(self, bot_name: str, capabilities: Iterable[str]) -> None:
        with self._lock:
            for cap in capabilities:
                if bot_name not in self._caps[cap]:
                    self._caps[cap].append(bot_name)
        if self._auto_persist:
            self.persist()

    def unregister(self, bot_name: str, capability: str) -> None:
        with self._lock:
            bots = self._caps.get(capability)
            if bots and bot_name in bots:
                bots.remove(bot_name)
                if not bots:
                    self._caps.pop(capability, None)
        if self._auto_persist:
            self.persist()

    def remove_capability(self, capability: str) -> None:
        with self._lock:
            self._caps.pop(capability, None)
        if self._auto_persist:
            self.persist()

    def find(self, capability: str) -> List[str]:
        with self._lock:
            return list(self._caps.get(capability, []))

    def all_capabilities(self) -> Dict[str, List[str]]:
        with self._lock:
            return {k: list(v) for k, v in self._caps.items()}

    def persist(self) -> None:
        if not self._path:
            return
        try:
            with self._lock:
                data = {k: list(v) for k, v in self._caps.items()}
            tmp_path = f"{self._path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, self._path)
        except Exception as exc:
            logger.error("failed to persist registry to %s: %s", self._path, exc)


__all__ = ["CapabilityRegistry"]
