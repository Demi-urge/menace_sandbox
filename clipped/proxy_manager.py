"""Light‑weight proxy manager with rotation and health checks.

This module originally only demonstrated a basic CLI that returned the
first active proxy from a JSON file.  It has been expanded into a small
utility that performs simple proxy rotation, optional liveness checks
and persistent status tracking.  The JSON file keeps the list of
proxies as well as their current state (``active``, ``in_use`` or
``down``).  Whenever a proxy is acquired it is marked as ``in_use`` and
moved to the end of the list to provide a very small rotation mechanism.
Proxies can be released or marked as failed which updates the stored
status.  Health checking uses the optional :mod:`requests` dependency and
is skipped when the library is unavailable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

try:  # optional dependency for proxy checking
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional
    requests = None  # type: ignore


@dataclass
class Proxy:
    ip: str
    port: str
    status: str = "active"

    @property
    def address(self) -> str:
        return f"{self.ip}:{self.port}"


def _load(path: Path) -> List[Proxy]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("proxies", [])
    proxies: List[Proxy] = []
    for entry in raw:
        proxies.append(
            Proxy(
                ip=entry.get("ip", ""),
                port=str(entry.get("port", "")),
                status=entry.get("status", "active"),
            )
        )
    return proxies


def _save(path: Path, data: List[Proxy]) -> None:
    serialized = [dict(ip=p.ip, port=p.port, status=p.status) for p in data]
    payload = {"schema_version": "1.0", "proxies": serialized}
    path.write_text(json.dumps(payload))


def _check_proxy(p: Proxy) -> bool:
    if not requests:
        return True
    proxy = f"http://{p.address}"
    try:
        requests.get(
            "http://example.com",
            proxies={"http": proxy, "https": proxy},
            timeout=5,
        )
        return True
    except Exception:
        return False


def get_available_proxy(path: Path, *, check: bool = False) -> Optional[str]:
    proxies = _load(path)
    for idx, entry in enumerate(proxies):
        if entry.status != "active":
            continue
        if check and not _check_proxy(entry):
            entry.status = "down"
            continue
        entry.status = "in_use"
        proxies.append(proxies.pop(idx))
        _save(path, proxies)
        return entry.address
    _save(path, proxies)
    return None


def release_proxy(path: Path, proxy: str) -> None:
    ip, port = proxy.split(":", 1)
    proxies = _load(path)
    for entry in proxies:
        if entry.ip == ip and entry.port == port:
            if entry.status == "in_use":
                entry.status = "active"
            break
    _save(path, proxies)


def fail_proxy(path: Path, proxy: str) -> None:
    ip, port = proxy.split(":", 1)
    proxies = _load(path)
    for entry in proxies:
        if entry.ip == ip and entry.port == port:
            entry.status = "down"
            break
    _save(path, proxies)


def list_active_proxies(path: Path) -> Iterator[str]:
    for entry in _load(path):
        if entry.status == "active":
            yield entry.address


def add_proxy(path: Path, proxy: str) -> None:
    ip, port = proxy.split(":", 1)
    proxies = _load(path)
    proxies.append(Proxy(ip=ip, port=port))
    _save(path, proxies)


def cli(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--list", action="store_true", help="list active proxies")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate proxies before returning",
    )
    parser.add_argument("--add", help="add a proxy in ip:port format")
    parser.add_argument("--release", help="mark an in-use proxy as active")
    parser.add_argument("--fail", help="mark a proxy as failed")
    args = parser.parse_args(argv)

    path = Path(args.file)

    if args.add:
        add_proxy(path, args.add)
        return
    if args.release:
        release_proxy(path, args.release)
        return
    if args.fail:
        fail_proxy(path, args.fail)
        return
    if args.list:
        for addr in list_active_proxies(path):
            print(addr)
        return

    proxy = get_available_proxy(path, check=args.check)
    if proxy:
        print(proxy)


__all__ = [
    "cli",
    "get_available_proxy",
    "release_proxy",
    "fail_proxy",
    "list_active_proxies",
    "add_proxy",
]
