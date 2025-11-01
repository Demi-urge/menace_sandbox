#!/usr/bin/env python3
"""Bootstrap the Menace environment and verify dependencies.

Run ``python scripts/bootstrap_env.py`` to install required tooling and
configuration.  Pass ``--skip-stripe-router`` to bypass the Stripe router
startup verification when working offline or without Stripe credentials.
"""
from __future__ import annotations

import argparse
import ast
import base64
import binascii
import codecs
import configparser
import hashlib
import html
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import unicodedata
from collections.abc import (
    Iterable as IterableABC,
    Mapping as MappingABC,
    MutableMapping as MutableMappingABC,
    MutableSequence as MutableSequenceABC,
    Sequence as SequenceABC,
)
from functools import lru_cache
import ntpath
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Literal,
)

from menace_sandbox import stripe_billing_router

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalize_sys_path_entry(entry: object) -> str | None:
    """Return a normalized representation of *entry* suitable for comparison."""

    if isinstance(entry, os.PathLike):
        entry = os.fspath(entry)
    if isinstance(entry, str):
        try:
            return os.path.normcase(os.path.abspath(entry))
        except OSError:
            return os.path.normcase(entry)
    return None


def _ensure_repo_root_on_path(repo_root: Path) -> None:
    """Inject *repo_root* into ``sys.path`` while avoiding duplicates."""

    target = os.path.normcase(str(repo_root.resolve()))
    normalized_entries: list[str | None] = [
        _normalize_sys_path_entry(entry) for entry in sys.path
    ]

    canonical = str(repo_root)

    try:
        existing_index = normalized_entries.index(target)  # type: ignore[arg-type]
    except ValueError:
        sys.path.insert(0, canonical)
        return

    if existing_index == 0:
        sys.path[0] = canonical
        return

    sys.path.pop(existing_index)
    sys.path.insert(0, canonical)


_ensure_repo_root_on_path(_REPO_ROOT)


LOGGER = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    """Raised when the environment bootstrap process cannot proceed."""


_DOCKER_SKIP_ENV = "MENACE_BOOTSTRAP_SKIP_DOCKER_CHECK"
_DOCKER_REQUIRE_ENV = "MENACE_REQUIRE_DOCKER"
_DOCKER_ASSUME_NO_ENV = "MENACE_BOOTSTRAP_ASSUME_NO_DOCKER"
_WSL_HOST_MOUNT_ROOT_ENV = "WSL_HOST_MOUNT_ROOT"
_WINDOWS_VISIBILITY_SKIP_ENV = "MENACE_BOOTSTRAP_NO_WINDOWS_PAUSE"


_WINDOWS_ENV_VAR_PATTERN = re.compile(r"%(?P<name>[A-Za-z0-9_]+)%")
_POSIX_ENV_VAR_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<simple>[A-Za-z_][A-Za-z0-9_]*))"
)


_WINDOWS_SYSTEM_DIRECTORY_SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("System32",),
    ("System32", "WindowsPowerShell", "v1.0"),
    ("System32", "wbem"),
    ("System32", "WindowsPowerShell", "v1.0", "Modules"),
    ("Sysnative",),
    ("Sysnative", "WindowsPowerShell", "v1.0"),
    ("SysWOW64",),
    ("SysWOW64", "WindowsPowerShell", "v1.0"),
)


_APPROX_PREFIX_PATTERN = re.compile(
    r"^(?P<prefix>about|approx(?:\.|imately)?|approximately|around|roughly|near(?:ly)?|~|≈)\s*",
    flags=re.IGNORECASE,
)

_APPROX_SUFFIX_PATTERN = re.compile(
    r"(about|approx(?:\.|imately)?|approximately|around|roughly|near(?:ly)?|~|≈)\s*$",
    flags=re.IGNORECASE,
)

_BACKOFF_INTERVAL_PATTERN = re.compile(
    r"""
    (?P<prefix>
        (?:about|approx(?:\.|imately)?|approximately|around|roughly|near(?:ly)?|~|≈)\s*
    )?
    (?P<number>[0-9]+(?:\.[0-9]+)?)
    \s*
    (?P<unit>ms|msec|milliseconds|s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_ISO_DURATION_PATTERN = re.compile(
    r"""
    ^
    (?P<sign>[-+]?)
    P
    (?=
        (?:\d+[YMWDHMS])
        |
        T(?:\d+[HMS])
    )
    (?:(?P<years>\d+(?:\.\d+)?)Y)?
    (?:(?P<months>\d+(?:\.\d+)?)M)?
    (?:(?P<weeks>\d+(?:\.\d+)?)W)?
    (?:(?P<days>\d+(?:\.\d+)?)D)?
    (?:
        T
        (?:(?P<hours>\d+(?:\.\d+)?)H)?
        (?:(?P<minutes>\d+(?:\.\d+)?)M)?
        (?:(?P<seconds>\d+(?:\.\d+)?)S)?
    )?
    $
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_GO_DURATION_PATTERN = re.compile(
    r"\b[0-9]+(?:\.[0-9]+)?[hms](?:[0-9]+(?:\.[0-9]+)?[hms]){0,2}\b",
    flags=re.IGNORECASE,
)

_GO_DURATION_COMPONENT_PATTERN = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[hms])",
    flags=re.IGNORECASE,
)

_CLOCK_DURATION_PATTERN = re.compile(r"^\d+(?::\d+){1,3}(?:\.\d+)?$")

_CLOCK_DURATION_SEARCH_PATTERN = re.compile(r"\b\d+(?::\d+){1,3}(?:\.\d+)?\b")

_CLOCK_DURATION_LAYOUTS: dict[int, tuple[str, ...]] = {
    2: ("minutes", "seconds"),
    3: ("hours", "minutes", "seconds"),
    4: ("days", "hours", "minutes", "seconds"),
}

_CLOCK_DURATION_FACTORS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}

_CLOCK_DURATION_SYMBOLS = {
    "seconds": "s",
    "minutes": "m",
    "hours": "h",
    "days": "d",
}


_FULLWIDTH_ASCII_TRANSLATION: dict[int, str | None] = {
    **{code: chr(code - 0xFEE0) for code in range(0xFF01, 0xFF5F)},
    0x3000: " ",
    ord("﹔"): ";",
    ord("﹕"): ":",
    ord("﹖"): "?",
    ord("﹗"): "!",
    ord("﹑"): ",",
    ord("﹘"): "-",
    ord("﹣"): "-",
    ord("﹦"): "=",
    ord("﹨"): "\\",
    ord("﹩"): "$",
    ord("﹪"): "%",
    ord("﹫"): "@",
    ord("〜"): "~",
    ord("〰"): "~",
    ord("‒"): "-",
    ord("–"): "-",
    ord("—"): "-",
    ord("―"): "-",
    ord("−"): "-",
}

_NON_BREAKING_WHITESPACE_CODEPOINTS: tuple[int, ...] = (
    0x00A0,  # NO-BREAK SPACE
    0x2007,  # FIGURE SPACE
    0x202F,  # NARROW NO-BREAK SPACE
    0x205F,  # MEDIUM MATHEMATICAL SPACE
    0x3000,  # IDEOGRAPHIC SPACE (full-width space)
)

_INVISIBLE_WORKER_BANNER_CODEPOINTS: tuple[int, ...] = (
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
)

_WORKER_BANNER_CHARACTER_TRANSLATION: dict[int, str | None] = {
    **_FULLWIDTH_ASCII_TRANSLATION,
    **{code: " " for code in _NON_BREAKING_WHITESPACE_CODEPOINTS},
    **{code: None for code in _INVISIBLE_WORKER_BANNER_CODEPOINTS},
}

_DURATION_UNIT_NORMALISATION = {
    "ms": "ms",
    "msec": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "seconds": "s",
    "m": "m",
    "min": "m",
    "mins": "m",
    "minutes": "m",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hours": "h",
}


_DOCKER_LOG_FIELD_PATTERN = re.compile(
    r"""
    (?P<key>[A-Za-z0-9_.-]+)
    =
    (
        "(?P<double>(?:\\.|[^"\\])*)"
        |
        '(?P<single>(?:\\.|[^'\\])*)'
        |
        (?P<bare>[^\s]+)
    )
    """,
    flags=re.VERBOSE,
)

_WORKER_ERROR_NORMALISERS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"worker\s+stalled", flags=re.IGNORECASE),
        "stalled_restart",
        "Docker Desktop automatically restarted a background worker after it stalled",
    ),
    (
        re.compile(r"restart\s+loop", flags=re.IGNORECASE),
        "restart_loop",
        "Docker Desktop detected that a background worker entered a restart loop",
    ),
    (
        re.compile(r"health\s*check\s+(?:failed|timed?\s*out)", flags=re.IGNORECASE),
        "healthcheck_failure",
        "Docker Desktop reported that the worker health check failed",
    ),
)

_WORKER_ERROR_CODE_NORMALISATION: dict[str, str] = {
    "worker_stalled": "worker_stalled",
}

for _pattern, _code, _narrative in _WORKER_ERROR_NORMALISERS:
    canonical = "worker_stalled" if _code == "stalled_restart" else _code
    _WORKER_ERROR_CODE_NORMALISATION[_code.casefold()] = canonical
    _WORKER_ERROR_CODE_NORMALISATION[_code.replace("-", "_").casefold()] = canonical
    _WORKER_ERROR_CODE_NORMALISATION[_narrative.casefold()] = canonical

_WORKER_ERROR_NARRATIVE_LOOKUP: dict[str, str] = {
    narrative.casefold(): "worker_stalled" if code == "stalled_restart" else code
    for _pattern, code, narrative in _WORKER_ERROR_NORMALISERS
}

_WORKER_STALLED_PRIMARY_CODE = _WORKER_ERROR_NORMALISERS[0][1]
_WORKER_STALLED_PRIMARY_NARRATIVE = _WORKER_ERROR_NORMALISERS[0][2]
_WORKER_STALLED_SIGNATURE_PREFIX = "worker-banner:"

#
# ``Docker Desktop`` builds for Windows occasionally vary the terminology they
# use when reporting background worker health.  Historically the primary
# symptom was the "worker stalled; restarting" banner; newer releases have
# started to surface synonymous phrasing such as "worker stuck; restarting" or
# "worker is stuck" while the recovery behaviour remains identical.  The
# sanitisation pipeline relies on collapsing all of those variants into a single
# canonical "worker stalled" form before the downstream heuristics run.  The
# ``_WORKER_STALL_ROOT_PATTERN`` consolidates the acceptable verb forms so
# regular expressions throughout this module can be generated programmatically
# and updated in a single location when Docker introduces new synonyms.
#

#
# Docker Desktop 4.32+ occasionally reports workers as "hung" or "frozen" when
# the underlying restart symptoms are identical to the long-standing
# ``worker stalled`` banner.  Treat those synonyms as equivalent so the
# sanitisation pipeline collapses every variation into the canonical narrative.
_WORKER_STALL_ROOT_PATTERN = r"""(?x:
    (?:
        stall(?:ed|ing|s)?
        |
        stuck(?:ing)?
        |
        hang(?:ed|ing|s)?
        |
        hung
        |
        freez(?:e|ing|ed|es)?
        |
        froz(?:e|en)
        |
        unrespons(?:ive|iveness)?
        |
        non[-_\s]?respons(?:ive|iveness)?
        |
        not[-_\s]?respond(?:ing|ed)?
        |
        no[-_\s]?response
        |
        timeout
        |
        timed[-_\s]?out
        |
        unreach(?:able|ability)?
        |
        no[-_\s]?heartbeat
        |
        heartbeat[-_\s]?lost
        |
        lost[-_\s]?heartbeat
        |
        off[-_\s]?line
        |
        offline
        |
        disconnect(?:ed|ing|s)?
        |
        no[-_\s]?reply
        |
        non[-_\s]?reply
    )
)"""
_WORKER_STALL_FOLLOWER_WORDS: tuple[str, ...] = (
    "be",
    "been",
    "being",
    "become",
    "becomes",
    "becoming",
    "became",
    "got",
    "gets",
    "getting",
    "gotten",
    "stay",
    "stays",
    "stayed",
    "staying",
    "remain",
    "remains",
    "remaining",
    "remained",
    "keep",
    "keeps",
    "kept",
    "keeping",
    "continue",
    "continues",
    "continued",
    "continuing",
    "to",
)
_WORKER_STALL_INTENSIFIER_WORDS: tuple[str, ...] = (
    "still",
    "yet",
    "again",
    "persistently",
    "chronically",
    "repeatedly",
    "constantly",
    "continually",
    "continuously",
    "regularly",
    "periodically",
    "sporadically",
    "intermittently",
    "frequently",
    "briefly",
    "temporarily",
    "newly",
    "recently",
    "consistently",
    "always",
    "ever",
    "just",
    "even",
    "almost",
    "nearly",
    "virtually",
)
_WORKER_STALL_FOLLOWER_PATTERN = "|".join(
    re.escape(word) for word in _WORKER_STALL_FOLLOWER_WORDS
)
_WORKER_STALL_INTENSIFIER_PATTERN = "|".join(
    re.escape(word) for word in _WORKER_STALL_INTENSIFIER_WORDS
)
_WORKER_STALL_KEYWORD_TOKENS: frozenset[str] = frozenset(
    {
        "stall",
        "stalled",
        "stalling",
        "stalls",
        "stuck",
        "sticking",
        "hang",
        "hung",
        "hangs",
        "hanging",
        "freeze",
        "freezes",
        "freezing",
        "frozen",
        "unresponsive",
        "nonresponsive",
        "non-responsive",
        "timeout",
        "timed",
        "timedout",
        "responding",
        "respond",
        "response",
        "unreachable",
        "heartbeat",
        "offline",
        "disconnected",
    }
)

_WORKER_STALLED_VARIATIONS_BODY = rf"""
    (?:
        \s+(?:has|have|had|is|was|are|were)
        |
        \s+(?:may|might|could|should|would)
        |
        \s+(?:appears?|appeared|appearing|seems?|seemed|seeming)(?:\s+to)?(?:\s+have)?(?:\s+been)?
        |
        \s+(?:remains?|remaining|remained)
        |
        \s+(?:stays?|stayed|staying)
        |
        \s+(?:keeps?|kept|keeping)
        |
        \s+(?:continues?|continued|continuing)(?:\s+to)?
        |
        \s+(?:{_WORKER_STALL_FOLLOWER_PATTERN})
        |
        \s+(?:{_WORKER_STALL_INTENSIFIER_PATTERN})
    )+
    \s+{_WORKER_STALL_ROOT_PATTERN}
"""

_BASE_WORKER_ERROR_CODE_LABELS: dict[str, str] = {
    "stalled_restart": "an automatic restart after a stall",
    "restart_loop": "a restart loop",
    "healthcheck_failure": "a health-check failure",
    "VPNKIT_HEALTHCHECK_FAILED": "a vpnkit health-check failure",
    "VPNKIT_UNRESPONSIVE": "an unresponsive vpnkit service",
    "VPNKIT_BACKGROUND_SYNC_STALLED": "a stalled vpnkit background sync worker",
    "VPNKIT_SYNC_TIMEOUT": "a vpnkit background sync timeout",
    "VPNKIT_HNS_UNAVAILABLE": "vpnkit losing contact with the Host Network Service",
    "VPNKIT_HNS_UNREACHABLE": "vpnkit losing contact with the Host Network Service",
    "VPNKIT_VSOCK_UNRESPONSIVE": "a stalled vsock channel between Windows and the Docker VM",
    "VPNKIT_VSOCK_TIMEOUT": "a vsock connection timeout between Windows and the Docker VM",
    "VPNKIT_BACKGROUND_SYNC_IO_PRESSURE": "I/O pressure impacting the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_DISK_PRESSURE": "disk pressure impacting the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_CPU_PRESSURE": "CPU pressure impacting the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_MEMORY_PRESSURE": "memory pressure impacting the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_NETWORK_PRESSURE": "network pressure impacting the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_NETWORK_JITTER": "network jitter disrupting the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_NETWORK_SATURATION": "network saturation throttling the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_IO_THROTTLED": "host I/O throttling slowing the vpnkit background sync worker",
    "VPNKIT_BACKGROUND_SYNC_NETWORK_CONGESTION": "network congestion impacting the vpnkit background sync worker",
    "VPNKIT_VSOCK_SIGNAL_LOST": "a vsock signal interruption between Windows and the Docker VM",
    "VPNKIT_VSOCK_CHANNEL_LOST": "a vsock channel interruption between Windows and the Docker VM",
    "WSL_VM_STOPPED": "a stopped WSL virtual machine",
    "WSL_VM_CRASHED": "a crashed WSL virtual machine",
    "WSL_VM_HIBERNATED": "a hibernated WSL virtual machine",
    "WSL_VM_SUSPENDED": "a suspended WSL virtual machine",
    "WSL_KERNEL_MISSING": "a missing Windows Subsystem for Linux kernel",
    "HCS_E_ACCESS_DENIED": "an access-denied error from the Host Compute Service",
}

_WSL_ERROR_CODE_LABEL_ALIASES = {
    code.replace("WSL_", "WSL2_", 1): label
    for code, label in _BASE_WORKER_ERROR_CODE_LABELS.items()
    if code.startswith("WSL_")
}

_WORKER_ERROR_CODE_LABELS: Mapping[str, str] = {
    **_BASE_WORKER_ERROR_CODE_LABELS,
    **_WSL_ERROR_CODE_LABEL_ALIASES,
}


_VIRTUALIZATION_ERROR_CODE_PREFIXES: tuple[str, ...] = (
    "WSL_",
    "WSL2_",
    "HYPERV",
    "VIRTUALIZATION",
)


_VIRTUALIZATION_ERROR_CODE_EXACT_MATCHES: frozenset[str] = frozenset(
    {
        "HCS_E_ACCESS_DENIED",
        "HCS_E_HYPERV_NOT_PRESENT",
        "HCS_E_HYPERV_NOT_RUNNING",
    }
)


#
# ``Docker Desktop`` occasionally emits benign ``worker stalled`` banners while a
# background component is recovering from transient state (for example after the
# host wakes from sleep).  Those incidents usually surface ``stalled_restart``
# style error codes and low restart counters.  Treating every occurrence as a
# warning generates noise for Windows developers and obscures genuine
# virtualization faults.  The codes below are categorised as "benign" when they
# appear alongside low restart counts and short backoff windows, allowing the
# diagnostics pipeline to downgrade the health assessment to informational
# guidance instead of a warning.
_BENIGN_WORKER_ERROR_CODES: frozenset[str] = frozenset(
    {
        "STALLED_RESTART",
        "VPNKIT_BACKGROUND_SYNC_STALLED",
        "VPNKIT_SYNC_TIMEOUT",
    }
)

# Docker Desktop exponentially increases the restart backoff when a worker keeps
# flapping.  Once the delay grows beyond roughly a minute the worker is unlikely
# to stabilise without intervention.  Values below this threshold are treated as
# transient churn and therefore eligible for informational guidance.
_SUSTAINED_BACKOFF_THRESHOLD = 45.0


@dataclass(frozen=True)
class _WorkerErrorCodeDirective:
    """Guidance describing how to remediate a specific worker error code."""

    reason: str
    detail: str | None = None
    remediation: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


_WORKER_ERROR_CODE_GUIDANCE: Mapping[str, _WorkerErrorCodeDirective] = {
    "WSL_KERNEL_OUTDATED": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop detected that the Windows Subsystem for Linux kernel is outdated"
        ),
        detail=(
            "Update the Windows Subsystem for Linux kernel so Docker Desktop can stabilise "
            "its background workers"
        ),
        remediation=(
            "Run 'wsl --update' from an elevated PowerShell session to install the latest WSL kernel",
            "Reboot Windows after updating the WSL kernel to restart Docker Desktop with the new kernel",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_kernel_outdated": (
                "Update the WSL kernel via 'wsl --update' and reboot Windows"
            ),
        },
    ),
    "WSL_DISTRIBUTION_STOPPED": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop reported that its WSL distributions are stopped or unavailable"
        ),
        detail=(
            "Ensure the 'docker-desktop' and 'docker-desktop-data' WSL distributions are running"
        ),
        remediation=(
            "Run 'wsl --shutdown' followed by launching Docker Desktop to restart the distributions",
            "Verify Docker Desktop > Settings > Resources > WSL Integration has the required distributions enabled",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_distribution_stopped": (
                "Restart the docker-desktop WSL distributions and re-enable WSL integration"
            ),
        },
    ),
    "HYPERV_DISABLED": _WorkerErrorCodeDirective(
        reason="Docker Desktop flagged that Hyper-V is disabled",
        detail="Enable Hyper-V features required for Docker Desktop's virtualization stack",
        remediation=(
            "Enable 'Hyper-V' and 'Virtual Machine Platform' via OptionalFeatures.exe and reboot Windows",
            "Ensure virtualization support is enabled in firmware before re-launching Docker Desktop",
        ),
        metadata={
            "docker_worker_last_error_guidance_hyperv_disabled": (
                "Enable Hyper-V and the Virtual Machine Platform Windows features, then reboot"
            ),
        },
    ),
    "WSL_VM_PAUSED": _WorkerErrorCodeDirective(
        reason="Docker Desktop observed the WSL virtualization environment is paused",
        detail="Resume the Windows virtualization stack so Docker Desktop workers can recover",
        remediation=(
            "Run 'wsl --shutdown' and relaunch Docker Desktop to rebuild the virtualization VM",
            "Disable Windows Fast Startup or hibernation features that pause WSL between reboots",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_vm_paused": (
                "Resume WSL by running 'wsl --shutdown' and restarting Docker Desktop"
            ),
        },
    ),
    "WSL_VM_SUSPENDED": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop detected the WSL virtualization environment was suspended"
        ),
        detail=(
            "Windows sleep or hibernation left the docker-desktop WSL VM suspended, preventing worker recovery"
        ),
        remediation=(
            "Run 'wsl --shutdown' from an elevated PowerShell session and relaunch Docker Desktop",
            "Disable Windows Fast Startup or configure the machine to perform a full shutdown instead of hibernating",
            "Ensure security or endpoint management agents are not suspending the docker-desktop WSL VM",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_vm_suspended": (
                "Shut down WSL using 'wsl --shutdown', restart Docker Desktop, and avoid suspending the docker-desktop VM"
            ),
        },
    ),
    "WSL_VM_HIBERNATED": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop reported that the WSL virtualization environment resumed from hibernation"
        ),
        detail=(
            "Windows hibernation or Fast Startup left the docker-desktop WSL virtual machine in a suspended state"
        ),
        remediation=(
            "Run 'wsl --shutdown' from an elevated PowerShell session and restart Docker Desktop",
            "Disable Windows Fast Startup to prevent WSL from entering a hibernated state before launching Docker Desktop",
            "If hibernation is required, ensure Docker Desktop is fully shut down prior to putting Windows to sleep",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_vm_hibernated": (
                "Shut down WSL using 'wsl --shutdown', restart Docker Desktop, and disable Windows Fast Startup to avoid hibernating the docker-desktop VM"
            ),
        },
    ),
    "VIRTUALIZATION_UNAVAILABLE": _WorkerErrorCodeDirective(
        reason="Docker Desktop cannot access hardware virtualization",
        detail="Enable hardware virtualization support in firmware or the host OS",
        remediation=(
            "Enable virtualization extensions (Intel VT-x/AMD-V) in the system BIOS/UEFI settings",
            "Close other hypervisors (Hyper-V, VirtualBox, VMware) that might be monopolising virtualization",
        ),
        metadata={
            "docker_worker_last_error_guidance_virtualization_unavailable": (
                "Enable CPU virtualization in firmware and ensure no other hypervisor is monopolising it"
            ),
        },
    ),
    "VPNKIT_HEALTHCHECK_FAILED": _WorkerErrorCodeDirective(
        reason="Docker Desktop detected that its vpnkit networking service failed health checks",
        detail="The embedded vpnkit process is restarting after repeated health-check failures, leaving containers without networking",
        remediation=(
            "Restart Docker Desktop to rebuild the vpnkit networking service",
            "Allow 'com.docker.backend' and 'vpnkit' through local firewalls, antivirus, or VPN filters",
            "Reset Docker Desktop networking from Settings > Troubleshooting if restarts do not stabilise vpnkit",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_healthcheck_failed": (
                "Restart Docker Desktop, permit vpnkit through security software, or reset Docker Desktop networking"
            ),
        },
    ),
    "VPNKIT_UNRESPONSIVE": _WorkerErrorCodeDirective(
        reason="Docker Desktop reported that its vpnkit networking service stopped responding",
        detail="vpnkit is the network proxy that bridges Windows and Linux networking; when it stalls containers lose connectivity",
        remediation=(
            "Restart Docker Desktop to relaunch the vpnkit networking backend",
            "Temporarily disable or reconfigure VPN clients that intercept network adapters used by Docker Desktop",
            "Run 'wsl --shutdown' and relaunch Docker Desktop to rebuild the networking integration",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_unresponsive": (
                "Restart Docker Desktop, adjust VPN/firewall tooling, or rebuild networking with 'wsl --shutdown'"
            ),
        },
    ),
    "VPNKIT_BACKGROUND_SYNC_STALLED": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop reported that the vpnkit background sync worker stopped making progress"
        ),
        detail=(
            "The vpnkit background sync process keeps port forwarding and DNS configuration aligned between Windows and the Docker VM. "
            "When the worker stalls, container networking gradually degrades until vpnkit is restarted."
        ),
        remediation=(
            "Restart Docker Desktop to relaunch the vpnkit background sync worker",
            "Temporarily disable VPN, firewall, or endpoint protection agents that intercept Hyper-V loopback traffic",
            "If stalls persist, reset Docker Desktop networking from Settings > Troubleshooting and reboot Windows",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_background_sync_stalled": (
                "Restart Docker Desktop and relax VPN/firewall inspection so the vpnkit background sync worker can recover"
            ),
        },
    ),
    "VPNKIT_SYNC_TIMEOUT": _WorkerErrorCodeDirective(
        reason="Docker Desktop detected that the vpnkit background sync worker timed out",
        detail=(
            "vpnkit background sync timed out while replicating networking state between Windows and Linux. Containers may miss port or DNS updates until the worker is restarted."
        ),
        remediation=(
            "Restart Docker Desktop to restart the vpnkit background sync worker",
            "Pause VPN clients or local security software that inspects Docker's virtual adapters",
            "Run 'wsl --shutdown' and start Docker Desktop again to rebuild the networking integration",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_sync_timeout": (
                "Restart Docker Desktop and reduce VPN/firewall interference so vpnkit background sync completes"
            ),
        },
    ),
    "VPNKIT_BACKGROUND_SYNC_IO_PRESSURE": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop detected that the vpnkit background sync worker is throttled by host I/O pressure"
        ),
        detail=(
            "Heavy disk or network activity on Windows can starve vpnkit's background sync loop, forcing repeated restarts while the worker waits for bandwidth."
        ),
        remediation=(
            "Pause or reschedule antivirus scans, backup jobs, and file synchronisation tools that saturate disk or network throughput",
            "Ensure the Docker Desktop virtual disk resides on fast local storage with several gigabytes of free space",
            "Restart Docker Desktop after the host I/O pressure subsides so vpnkit can rebuild its synchronization state",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_background_sync_io_pressure": (
                "Reduce host I/O pressure so Docker Desktop's vpnkit background sync worker can stabilise"
            ),
        },
    ),
    "VPNKIT_BACKGROUND_SYNC_IO_THROTTLED": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop detected that host I/O throttling is delaying the vpnkit background sync worker"
        ),
        detail=(
            "Windows storage stacks occasionally throttle heavy read/write bursts when background maintenance runs. "
            "During those throttling windows vpnkit cannot checkpoint port forwarding and DNS state, so Docker "
            "restarts the worker repeatedly until throughput recovers."
        ),
        remediation=(
            "Temporarily pause disk-heavy backup or synchronisation utilities that saturate the Docker Desktop data disk",
            "Allow Windows maintenance tasks (such as Storage Sense or Defender scans) to complete before retrying Docker commands",
            "Restart Docker Desktop once I/O throttling subsides so the vpnkit background sync worker can resynchronise",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_background_sync_io_throttled": (
                "Reduce host I/O throttling and restart Docker Desktop so the vpnkit background sync worker can resynchronise"
            ),
        },
    ),
    "VPNKIT_BACKGROUND_SYNC_DISK_PRESSURE": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop reported that the vpnkit background sync worker is blocked by disk pressure on the Windows host"
        ),
        detail=(
            "When the Windows volume hosting Docker Desktop's data is nearly full or slow to respond, vpnkit fails to persist synchronization checkpoints and continually restarts."
        ),
        remediation=(
            "Free disk space on the drive that stores Docker Desktop data (typically %LOCALAPPDATA% or ProgramData)",
            "Move large container images or volumes off the affected disk and rerun 'docker system prune' if appropriate",
            "Restart Docker Desktop after reclaiming space so the vpnkit background sync worker can resume",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_background_sync_disk_pressure": (
                "Free disk space so Docker Desktop's vpnkit background sync worker is no longer blocked by storage pressure"
            ),
        },
    ),
    "VPNKIT_BACKGROUND_SYNC_NETWORK_JITTER": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop detected that network jitter is disrupting the vpnkit background sync worker"
        ),
        detail=(
            "vpnkit mirrors container networking configuration between Windows and the Linux VM. "
            "Erratic latency spikes or packet loss on the host can cause vpnkit's background sync loop to miss deadlines, "
            "triggering repeated restarts until connectivity stabilises."
        ),
        remediation=(
            "Disable or reconfigure VPN, firewall, and traffic-shaping tools that introduce jitter on the docker-desktop network adapters",
            "Avoid saturating Wi-Fi or metered links while Docker Desktop is synchronising networking state",
            "Restart Docker Desktop after stabilising the host network so vpnkit can rebuild its routing tables",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_background_sync_network_jitter": (
                "Stabilise host networking (VPN, Wi-Fi, firewall rules) and restart Docker Desktop so vpnkit can resynchronise"
            ),
        },
    ),
    "HNS_SERVICE_UNAVAILABLE": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop detected that the Windows Host Network Service (HNS) is unavailable"
        ),
        detail=(
            "HNS brokers virtual network switches for containers; when it is unreachable Docker Desktop cannot wire container networking"
        ),
        remediation=(
            "Restart the 'Host Network Service' Windows service via an elevated PowerShell session (Restart-Service hns)",
            "Run 'netsh winsock reset' and reboot Windows if HNS fails to initialise after a service restart",
            "After HNS is healthy, restart Docker Desktop so its networking workers can reconnect",
        ),
        metadata={
            "docker_worker_last_error_guidance_hns_service_unavailable": (
                "Restart the Host Network Service (HNS), reset Winsock if needed, then relaunch Docker Desktop"
            ),
        },
    ),
    "VPNKIT_HNS_UNAVAILABLE": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop reported that vpnkit cannot reach the Windows Host Network Service (HNS)"
        ),
        detail=(
            "HNS provisions the virtual switches vpnkit relies on. When it is unavailable the vpnkit workers restart continuously and container networking breaks."
        ),
        remediation=(
            "Restart the 'Host Network Service' (HNS) from an elevated PowerShell session (Restart-Service hns)",
            "Run 'netsh winsock reset' and reboot Windows if HNS refuses to start",
            "Restart Docker Desktop after HNS connectivity is restored",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_hns_unavailable": (
                "Restart HNS, reset Winsock if necessary, then relaunch Docker Desktop so vpnkit can reconnect"
            ),
        },
    ),
    "VPNKIT_HNS_UNREACHABLE": _WorkerErrorCodeDirective(
        reason="Docker Desktop detected that vpnkit lost connectivity with the Windows Host Network Service",
        detail=(
            "vpnkit needs a stable channel to HNS to attach container network adapters. Connectivity failures often stem from HNS crashes or aggressive security software."
        ),
        remediation=(
            "Restart the 'Host Network Service' (HNS) and verify it remains running",
            "Temporarily disable VPN or firewall tooling that could block Hyper-V networking",
            "After restoring connectivity, restart Docker Desktop to rebuild the vpnkit networking stack",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_hns_unreachable": (
                "Restore HNS connectivity and then restart Docker Desktop so vpnkit can reattach container networking"
            ),
        },
    ),
    "VPNKIT_VSOCK_UNRESPONSIVE": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop detected the vsock channel between Windows and the Docker VM became unresponsive"
        ),
        detail=(
            "vpnkit relies on the Hyper-V/WSL virtio-socket (vsock) transport to communicate with the Linux virtual machine. "
            "When that channel hangs, background workers stall and Docker restarts them in a loop."
        ),
        remediation=(
            "Restart Docker Desktop to rebuild the vsock connection to the docker-desktop WSL/Hyper-V VM",
            "Temporarily disable third-party VPN or firewall software that may intercept Hyper-V socket traffic",
            "If the issue recurs, reset Docker Desktop's networking stack or reinstall the WSL integration",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_vsock_unresponsive": (
                "Restart Docker Desktop and stabilise the Hyper-V vsock channel before retrying"
            ),
            "docker_worker_last_error_category": "vpnkit_vsock",
        },
    ),
    "VPNKIT_VSOCK_TIMEOUT": _WorkerErrorCodeDirective(
        reason=(
            "Docker Desktop reported timeouts while establishing the vsock channel to the Docker VM"
        ),
        detail=(
            "Repeated vsock handshakes timing out indicate the virtualization stack is overloaded or blocked, causing worker stalls."
        ),
        remediation=(
            "Restart Docker Desktop and allow the docker-desktop VM to boot cleanly",
            "Ensure Hyper-V or WSL 2 virtualization has sufficient CPU and memory allocations",
            "Review Windows event logs for Hyper-V socket errors and reinstall Docker Desktop if corruption is detected",
        ),
        metadata={
            "docker_worker_last_error_guidance_vpnkit_vsock_timeout": (
                "Restart Docker Desktop, verify virtualization resources, and repair the vsock channel if timeouts persist"
            ),
            "docker_worker_last_error_category": "vpnkit_vsock",
        },
    ),
    "WSL_VM_STOPPED": _WorkerErrorCodeDirective(
        reason="Docker Desktop observed that the docker-desktop WSL virtual machine stopped",
        detail=(
            "Docker Desktop runs inside the docker-desktop WSL distribution. When the VM stops, background workers restart in a loop until WSL is brought back online."
        ),
        remediation=(
            "Run 'wsl --status' to confirm the docker-desktop distributions are registered and running",
            "Execute 'wsl --update' followed by 'wsl --shutdown' to refresh the WSL kernel",
            "Restart Docker Desktop once WSL reports a healthy state",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_vm_stopped": (
                "Restart the docker-desktop WSL distributions and relaunch Docker Desktop after updating WSL"
            ),
        },
    ),
    "WSL_VM_CRASHED": _WorkerErrorCodeDirective(
        reason="Docker Desktop reported that the docker-desktop WSL virtual machine crashed",
        detail=(
            "Crashes in the docker-desktop WSL VM interrupt virtualization and force Docker Desktop to restart its workers until the VM boots successfully."
        ),
        remediation=(
            "Inspect 'wsl --status' and Windows Event Viewer for virtualization or driver failures",
            "Apply the latest Windows and WSL updates, then run 'wsl --shutdown' to rebuild the VM",
            "Restart Docker Desktop after confirming the WSL VM starts cleanly",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_vm_crashed": (
                "Update WSL, review virtualization crash logs, and restart Docker Desktop once the WSL VM is stable"
            ),
        },
    ),
    "WSL_KERNEL_MISSING": _WorkerErrorCodeDirective(
        reason="Docker Desktop detected that the Windows Subsystem for Linux kernel is missing",
        detail=(
            "Without the Microsoft-provided WSL kernel Docker Desktop cannot start its Linux VM, leaving workers stuck in a restart loop."
        ),
        remediation=(
            "Install the latest WSL kernel by running 'wsl --update' from an elevated PowerShell session",
            "Reboot Windows to finalise the kernel installation",
            "Launch Docker Desktop after the reboot so it can rebuild its WSL integration",
        ),
        metadata={
            "docker_worker_last_error_guidance_wsl_kernel_missing": (
                "Install the WSL kernel with 'wsl --update', reboot, then restart Docker Desktop"
            ),
        },
    ),
    "HCS_E_ACCESS_DENIED": _WorkerErrorCodeDirective(
        reason="Docker Desktop was blocked by the Windows Host Compute Service (HCS) because virtualization access was denied",
        detail=(
            "The Host Compute Service provisions the Hyper-V virtual machine that powers Docker Desktop. Access denied responses usually indicate virtualization features are disabled or group policy is preventing Hyper-V from launching."
        ),
        remediation=(
            "Enable 'Hyper-V' and 'Virtual Machine Platform' Windows features and reboot",
            "Ensure no group policy or third-party hypervisor is preventing the Host Compute Service from creating virtual machines",
            "Relaunch Docker Desktop after confirming the Host Compute Service starts without errors",
        ),
        metadata={
            "docker_worker_last_error_guidance_hcs_e_access_denied": (
                "Re-enable Hyper-V/Virtual Machine Platform and remove restrictions blocking the Host Compute Service"
            ),
        },
    ),
    "HCS_E_HYPERV_NOT_RUNNING": _WorkerErrorCodeDirective(
        reason="Docker Desktop detected that the Hyper-V hypervisor is not running",
        detail=(
            "Hyper-V must be running to host the docker-desktop virtual machine. When the hypervisor is disabled or stopped, Docker Desktop workers continually restart while waiting for the virtualization stack."
        ),
        remediation=(
            "Open an elevated PowerShell session and run 'bcdedit /set hypervisorlaunchtype auto' to ensure the Hyper-V hypervisor starts automatically",
            "Restart the 'Hyper-V Virtual Machine Management' and 'Hyper-V Host Compute Service' Windows services",
            "Reboot Windows after re-enabling Hyper-V so Docker Desktop can relaunch its virtualization backend",
        ),
        metadata={
            "docker_worker_last_error_guidance_hcs_e_hyperv_not_running": (
                "Re-enable the Hyper-V hypervisor (hypervisorlaunchtype auto) and restart the Hyper-V services"
            ),
        },
    ),
    "HCS_E_HYPERV_NOT_PRESENT": _WorkerErrorCodeDirective(
        reason="Docker Desktop detected that required Hyper-V features are not installed",
        detail=(
            "The Windows Hyper-V feature stack (including Virtual Machine Platform) is required to host the docker-desktop virtual machine. When those features are missing Docker Desktop cannot start its workers."
        ),
        remediation=(
            "Enable 'Hyper-V', 'Virtual Machine Platform', and 'Windows Hypervisor Platform' via OptionalFeatures.exe or the 'Enable-WindowsOptionalFeature' PowerShell cmdlet",
            "Ensure hardware virtualization support is enabled in firmware (Intel VT-x/AMD-V)",
            "Reboot Windows after installing Hyper-V features and relaunch Docker Desktop",
        ),
        metadata={
            "docker_worker_last_error_guidance_hcs_e_hyperv_not_present": (
                "Install the Hyper-V and Virtual Machine Platform Windows features and reboot"
            ),
        },
    ),
}

_WORKER_ERROR_CODE_GUIDANCE.setdefault(
    "WSL2_VM_SUSPENDED", _WORKER_ERROR_CODE_GUIDANCE["WSL_VM_SUSPENDED"]
)
_WORKER_ERROR_CODE_GUIDANCE.setdefault(
    "HYPERV_NOT_RUNNING",
    _WORKER_ERROR_CODE_GUIDANCE["HCS_E_HYPERV_NOT_RUNNING"],
)
_WORKER_ERROR_CODE_GUIDANCE.setdefault(
    "HYPERV_NOT_PRESENT",
    _WORKER_ERROR_CODE_GUIDANCE["HCS_E_HYPERV_NOT_PRESENT"],
)

_WORKER_ERROR_NARRATIVES: tuple[str, ...] = tuple(
    normaliser[2] for normaliser in _WORKER_ERROR_NORMALISERS
)


def _sanitize_error_code_suffix(code: str) -> str:
    """Return a filesystem-safe suffix derived from ``code``."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", code.upper()).strip("_")
    return normalized.lower() or "unknown"


def _build_vpnkit_background_sync_guidance(
    candidate: str,
    token_set: Collection[str],
    metadata_builder: Callable[[str], Mapping[str, str]],
) -> _WorkerErrorCodeDirective | None:
    """Return guidance for ``VPNKIT_BACKGROUND_SYNC`` error codes.

    Docker Desktop 4.33+ introduces additional ``VPNKIT_BACKGROUND_SYNC`` error
    codes that highlight the specific host resource starving the background
    synchronisation worker.  Earlier bootstrap revisions collapsed all
    ``vpnkit`` failures into a generic remediation block, which left Windows
    developers guessing whether CPU, memory, or network pressure triggered the
    ``worker stalled; restarting`` banner.  Providing tailored guidance here
    keeps the diagnostics actionable as Docker expands its telemetry surface.
    """

    prefix = "VPNKIT_BACKGROUND_SYNC_"
    if not candidate.startswith(prefix):
        return None

    # Normalise token variants so substring-based error codes (for example
    # ``IOPRESSURE``) are captured alongside delimited counterparts such as
    # ``IO_PRESSURE``.  Docker frequently tweaks error-code casing between
    # releases which makes relying on a single canonical spelling brittle.
    enriched_tokens = set(token_set)
    if "I" in enriched_tokens and "O" in enriched_tokens:
        enriched_tokens.add("IO")

    for token in tuple(token_set):
        upper = token.upper()
        if upper and upper not in {"BACKGROUND", "SYNC", "VPNKIT"}:
            if "IO" in upper:
                enriched_tokens.add("IO")
            if "DISK" in upper:
                enriched_tokens.add("DISK")
            if "STORAGE" in upper:
                enriched_tokens.add("STORAGE")
            if "LATENCY" in upper:
                enriched_tokens.add("LATENCY")
            if "THROTTLE" in upper:
                enriched_tokens.add("THROTTLE")

    token_set = enriched_tokens

    cpu_tokens = {"CPU"}
    memory_tokens = {"MEMORY", "RAM"}
    network_tokens = {
        "NETWORK",
        "NET",
        "BANDWIDTH",
        "CONGESTION",
        "LATENCY",
        "CONNECTIVITY",
        "THROUGHPUT",
        "SATURATION",
        "SATURATED",
        "SATURATING",
        "STARVATION",
        "STARVED",
        "STARVING",
        "UTILIZATION",
        "UTILISATION",
        "THROTTLE",
        "THROTTLED",
        "THROTTLING",
        "OVERLOAD",
        "OVERLOADED",
    }
    io_tokens = {
        "IO",
        "I/O",
        "IOPS",
        "LATENCY",
        "THROTTLE",
        "THROTTLED",
        "THROTTLING",
    }
    disk_tokens = {
        "DISK",
        "STORAGE",
        "VHD",
        "VHDX",
        "SSD",
        "HDD",
    }

    if token_set & cpu_tokens:
        summary = (
            "Reduce host CPU pressure so the vpnkit background sync worker can keep up"
        )
        reason = (
            "Docker Desktop detected that host CPU pressure is throttling the "
            "vpnkit background sync worker"
        )
        detail = (
            "Sustained CPU saturation on Windows or inside the docker-desktop VM "
            "prevents vpnkit from processing networking synchronisation events, "
            "forcing repeated restarts."
        )
        remediation = (
            "Close CPU-intensive workloads or increase Docker Desktop's CPU allocation in Settings > Resources",
            "Ensure Windows is using a high-performance power profile so the docker-desktop VM is not frequency limited",
            "Restart Docker Desktop after reducing CPU contention so the vpnkit background sync worker can stabilise",
        )
        return _WorkerErrorCodeDirective(
            reason=reason,
            detail=detail,
            remediation=remediation,
            metadata=metadata_builder(summary),
        )

    if token_set & memory_tokens:
        summary = (
            "Free host memory or expand Docker Desktop's allocation to stabilise the vpnkit background sync worker"
        )
        reason = (
            "Docker Desktop detected that memory pressure is starving the vpnkit "
            "background sync worker"
        )
        detail = (
            "When Windows exhausts available memory the docker-desktop VM is paged "
            "out, leaving vpnkit unable to reconcile port and DNS state until "
            "resources are freed."
        )
        remediation = (
            "Close memory-heavy applications or increase Docker Desktop's memory allocation in Settings > Resources",
            "Ensure antivirus, endpoint protection, or memory compression tools are not starving the docker-desktop WSL VM",
            "Restart Docker Desktop after freeing memory so vpnkit can rebuild its caches",
        )
        return _WorkerErrorCodeDirective(
            reason=reason,
            detail=detail,
            remediation=remediation,
            metadata=metadata_builder(summary),
        )

    if token_set & network_tokens:
        summary = (
            "Reduce network filtering, congestion, or saturation impacting the vpnkit background sync worker"
        )
        reason = (
            "Docker Desktop observed that host network congestion is delaying the "
            "vpnkit background sync worker"
        )
        detail = (
            "Aggressive VPN inspection, firewall filtering, or saturated network "
            "interfaces slow vpnkit's ability to propagate port forwarding and DNS "
            "updates between Windows and the docker-desktop VM."
        )
        remediation = (
            "Temporarily disable or relax VPN and firewall agents that intercept Hyper-V virtual switches",
            "Allow com.docker.backend and vpnkit.exe through endpoint protection tooling so networking updates are not throttled",
            "Restart Docker Desktop after reducing network congestion so the vpnkit background sync worker can resynchronise state",
        )
        return _WorkerErrorCodeDirective(
            reason=reason,
            detail=detail,
            remediation=remediation,
            metadata=metadata_builder(summary),
        )

    if (token_set & io_tokens) or (token_set & disk_tokens):
        disk_related = bool(token_set & disk_tokens)
        summary = (
            "Reduce disk pressure affecting the vpnkit background sync worker"
            if disk_related
            else "Reduce host I/O throttling impacting the vpnkit background sync worker"
        )
        reason = (
            "Docker Desktop detected that disk throughput limitations are restarting the vpnkit background sync worker"
            if disk_related
            else "Docker Desktop detected host I/O throttling is starving the vpnkit background sync worker"
        )
        detail = (
            "When Windows storage subsystems are saturated the docker-desktop virtual disk cannot service vpnkit's background"
            " synchronisation requests, forcing the worker to restart until disk pressure is relieved."
            if disk_related
            else "Host-level storage throttling or sustained I/O contention prevents vpnkit from flushing synchronisation"
            " events, causing repeated restarts until the contention abates."
        )
        remediation = (
            "Pause antivirus, backup, indexing, or encryption tools saturating the docker-desktop.vhdx virtual disk",
            "Verify the drive hosting Docker Desktop data has free space and, if necessary, move it to faster storage or an SSD",
            "Restart Docker Desktop after reducing disk and I/O contention so the vpnkit background sync worker can stabilise",
        )
        if not disk_related:
            remediation = (
                remediation[0],
                "Review Windows storage QoS policies, virtualization guard rails, or third-party tools that throttle the docker-desktop virtual disk",
                remediation[2],
            )
        return _WorkerErrorCodeDirective(
            reason=reason,
            detail=detail,
            remediation=remediation,
            metadata=metadata_builder(summary),
        )

    return None


def _build_vpnkit_vsock_guidance(
    candidate: str,
    token_set: Collection[str],
    metadata_builder: Callable[[str], Mapping[str, str]],
) -> _WorkerErrorCodeDirective | None:
    """Return guidance for ``VPNKIT_VSOCK`` style error codes."""

    if not candidate.startswith("VPNKIT_VSOCK_"):
        return None

    summary = (
        "Repair the Hyper-V vsock bridge between Windows and the docker-desktop VM"
    )
    reason = (
        "Docker Desktop detected instability in the Hyper-V vsock channel that vpnkit uses for networking"
    )
    detail = (
        "Docker Desktop surfaced vsock error code %s indicating the socket tunnel "
        "linking Windows and the docker-desktop virtual machine is unstable. "
        "vpnkit relies on this bridge to proxy networking; when the tunnel drops "
        "the worker stalls and restarts."
    ) % candidate
    remediation = (
        "Run 'wsl --shutdown' and restart Docker Desktop to rebuild the vsock transport",
        "Ensure virtualization-based security, antivirus, or third-party hypervisors are not blocking Hyper-V socket communication",
        "If the channel keeps failing, reset Docker Desktop from Settings > Troubleshoot or reinstall it to repair the vsock drivers",
    )

    if "TIMEOUT" in token_set:
        remediation = (
            *remediation[:-1],
            "Capture Windows Event Viewer > Applications and Services Logs > Microsoft > Windows > Hyper-V-Compute-Admin entries for persistent vsock timeouts and provide them to Docker Desktop support",
        )

    return _WorkerErrorCodeDirective(
        reason=reason,
        detail=detail,
        remediation=remediation,
        metadata=metadata_builder(summary),
    )


def _derive_generic_error_code_guidance(
    code: str,
) -> _WorkerErrorCodeDirective | None:
    """Heuristically derive remediation guidance for unrecognised *code*.

    Docker Desktop regularly introduces additional worker error codes without
    updating public documentation.  When the bootstrapper encounters an
    unfamiliar ``errCode`` value we still want to surface actionable guidance
    instead of emitting a bland "unknown code" notice.  This helper maps broad
    classes of error codes to production-ready remediation guidance by
    inspecting key substrings.
    """

    candidate = code.strip().upper()
    if not candidate:
        return None

    tokens = [token for token in re.split(r"[^A-Z0-9]+", candidate) if token]
    token_set = {token for token in tokens}

    def _metadata(summary: str) -> Mapping[str, str]:
        suffix = _sanitize_error_code_suffix(candidate)
        key = f"docker_worker_last_error_guidance_{suffix}"
        return {key: summary}

    wsl_tokens = [token for token in token_set if token.startswith("WSL")]
    if wsl_tokens:
        is_wsl2 = any(token.startswith("WSL2") for token in wsl_tokens)
        variant_label = "WSL 2" if is_wsl2 else "WSL"
        platform_label = (
            "Windows Subsystem for Linux 2" if is_wsl2 else "Windows Subsystem for Linux"
        )
        summary = (
            "Investigate the Docker Desktop %s distributions and restart the "
            "virtualisation backend"
        ) % variant_label
        remediation = (
            "Run 'wsl --status' to verify the docker-desktop distributions are registered and healthy",
            "Execute 'wsl --update' to install the latest WSL kernel and apply any pending fixes",
            "Run 'wsl --shutdown' followed by restarting Docker Desktop to relaunch the WSL backend",
        )
        detail = (
            "Docker Desktop surfaced %s worker error code %s which typically "
            "indicates the docker-desktop WSL distributions are stopped, "
            "outdated, or unreachable."
        ) % (platform_label, candidate)
        return _WorkerErrorCodeDirective(
            reason=(
                "Docker Desktop emitted a %s worker error (%s)"
                % (platform_label, candidate)
            ),
            detail=detail,
            remediation=remediation,
            metadata=_metadata(summary),
        )

    specialized = _build_vpnkit_background_sync_guidance(
        candidate, token_set, _metadata
    )
    if specialized is not None:
        return specialized

    specialized = _build_vpnkit_vsock_guidance(candidate, token_set, _metadata)
    if specialized is not None:
        return specialized

    if any(token.startswith("VPNKIT") for token in token_set):
        summary = (
            "Stabilise the vpnkit networking service used by Docker Desktop"
        )
        remediation = (
            "Restart Docker Desktop to relaunch the vpnkit networking proxy",
            "Allow 'com.docker.backend' and 'vpnkit.exe' through local firewalls, antivirus suites, and VPN clients",
            "If corporate VPNs are required, enable the Docker Desktop > Settings > Resources > Network > 'Enable integration with my VPN' option or equivalent",
        )
        detail = (
            "Docker Desktop reported vpnkit-specific worker error code %s, "
            "indicating its networking proxy is restarting repeatedly."
        ) % candidate
        return _WorkerErrorCodeDirective(
            reason=(
                "Docker Desktop flagged instability in its vpnkit networking component"
            ),
            detail=detail,
            remediation=remediation,
            metadata=_metadata(summary),
        )

    if "HNS" in token_set:
        summary = (
            "Repair the Windows Host Network Service (HNS) required for Docker Desktop networking"
        )
        remediation = (
            "Restart the 'Host Network Service' (HNS) via 'Restart-Service hns' from an elevated PowerShell session",
            "If HNS will not start, run 'netsh winsock reset' and reboot to rebuild the Windows networking stack",
            "Relaunch Docker Desktop after HNS is healthy so vpnkit and other networking workers can reconnect",
        )
        detail = (
            "Docker Desktop surfaced Host Network Service error code %s which typically indicates the Windows networking stack is offline."
        ) % candidate
        return _WorkerErrorCodeDirective(
            reason=(
                "Docker Desktop reported a Host Network Service (HNS) failure preventing networking workers from staying online"
            ),
            detail=detail,
            remediation=remediation,
            metadata=_metadata(summary),
        )

    virtualization_tokens = {"HYPERV", "VIRTUAL", "VIRT", "VMM", "HCS"}
    if any(any(token.startswith(prefix) for prefix in virtualization_tokens) for token in token_set):
        summary = (
            "Re-enable Windows virtualization features required by Docker Desktop"
        )
        remediation = (
            "Enable 'Hyper-V', 'Virtual Machine Platform', and virtualization support in firmware if they are disabled",
            "Close conflicting hypervisors such as VMware, VirtualBox, or WSL distributions consuming the same resources",
            "Reboot Windows after changing virtualization settings and relaunch Docker Desktop",
        )
        detail = (
            "Docker Desktop emitted virtualization-oriented worker error code %s, "
            "suggesting the Hyper-V/Host Compute Service stack cannot manage the virtual machine backing Docker Desktop."
        ) % candidate
        return _WorkerErrorCodeDirective(
            reason=(
                "Docker Desktop detected that required Windows virtualization services are misconfigured"
            ),
            detail=detail,
            remediation=remediation,
            metadata=_metadata(summary),
        )

    if "PRESSURE" in token_set:
        io_related = {token for token in token_set if token in {"IO", "DISK", "STORAGE", "I"}}
        summary = "Reduce host resource pressure impacting Docker Desktop background workers"
        base_detail = (
            "Host resource pressure (disk or I/O saturation) can starve Docker Desktop services and trigger worker restarts."
        )
        remediation = (
            "Close or reschedule disk-intensive tools (antivirus, backup, file sync) competing with Docker Desktop",
            "Ensure the drive hosting Docker Desktop data has adequate free space and performance headroom",
            "Restart Docker Desktop after reducing host resource pressure",
        )

        if io_related and "DISK" in io_related:
            detail = (
                "Windows reported disk pressure on the drive backing Docker Desktop. Clearing space and reducing heavy disk usage stabilises vpnkit and other workers."
            )
        elif io_related:
            detail = (
                "Windows reported I/O pressure affecting Docker Desktop. Relieving sustained bandwidth contention allows background workers to recover."
            )
        else:
            detail = base_detail

        return _WorkerErrorCodeDirective(
            reason="Docker Desktop observed host resource pressure throttling worker health",
            detail=detail,
            remediation=remediation,
            metadata=_metadata(summary),
        )

    return None


_WORKER_INLINE_CONTEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "been",
    "became",
    "become",
    "becomes",
    "becoming",
    "could",
    "had",
    "has",
    "have",
    "got",
    "gets",
    "getting",
    "gotten",
    "is",
    "may",
    "might",
    "nearly",
    "possibly",
    "probably",
    "reportedly",
    "remain",
    "remained",
    "remaining",
    "remains",
    "seems",
    "seemed",
    "seeming",
    "should",
    "stay",
    "stayed",
    "staying",
    "stays",
    "still",
    "that",
    "the",
    "this",
    "to",
    "keep",
    "keeps",
    "kept",
    "keeping",
    "continue",
    "continued",
    "continuing",
    "continues",
    "virtually",
    "was",
    "were",
    "would",
}

_WORKER_INLINE_CONTEXT_PATTERN = re.compile(
    r"""
    \bworker
    (?P<context_block>
        (?:(?:\s+
            (?:
                \[[^\]]{0,80}\]
                |
                \([^\)\n]{0,80}\)
                |
                \{[^\}\n]{0,80}\}
                |
                [\"'`][^\"'`\n]{0,80}[\"'`]
                |
                [A-Za-z0-9_.:/\\-]{1,64}
                (?:\s+[A-Za-z0-9_.:/\\-]{1,64}){0,3}
            )
        )){1,6}
    )
    \s+stalled\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _rewrite_inline_worker_contexts(message: str) -> str:
    """Normalise ``worker <context> stalled`` phrasing while retaining context."""

    if not message:
        return ""

    lowered = message.casefold()
    if "worker" not in lowered or "stall" not in lowered:
        return message

    segments: list[str] = []
    last_end = 0
    search_from = 0

    while True:
        position = lowered.find("worker", search_from)
        if position == -1:
            break

        match = _WORKER_INLINE_CONTEXT_PATTERN.match(message, position)
        if not match:
            search_from = position + len("worker")
            continue

        block = match.group("context_block")
        replacement = match.group(0)

        if block:
            candidate = _clean_worker_metadata_value(block)
            if candidate:
                tokens = [
                    token
                    for token in re.split(r"\s+", candidate)
                    if token
                    and token.strip().lower() not in _WORKER_INLINE_CONTEXT_STOPWORDS
                ]
                if tokens:
                    context = " ".join(tokens)
                    replacement = f"worker stalled (context={context})"

        segments.append(message[last_end : match.start()])
        segments.append(replacement)
        last_end = match.end()
        search_from = match.end()

    if not segments:
        return message

    segments.append(message[last_end:])
    return "".join(segments)


_WORKER_STALLED_VARIATIONS_PATTERN = re.compile(
    rf"""
    worker
{_WORKER_STALLED_VARIATIONS_BODY}
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORKER_GUIDANCE_SENTINELS: tuple[str, ...] = (
    "docker desktop worker processes are repeatedly restarting",
    "docker desktop reported worker restarts but indicated they are recovering automatically",
    "docker desktop reported it recovered from transient worker stalls and the background worker is stable",
    "docker desktop reported it briefly restarted a background worker and it is healthy",
)


def _contains_worker_stall_signal(message: str) -> bool:
    """Return ``True`` when *message* resembles a Docker worker stall banner."""

    if not message:
        return False

    lowered = message.casefold()
    if any(sentinel in lowered for sentinel in _WORKER_GUIDANCE_SENTINELS):
        return False

    if "worker" not in lowered:
        return False

    if not any(keyword in lowered for keyword in _WORKER_STALL_KEYWORD_TOKENS):
        return False

    normalized = _normalise_worker_stalled_phrase(message)
    collapsed = re.sub(r"[\s_-]+", " ", normalized).strip()
    collapsed_lower = collapsed.casefold()
    if "worker stalled" in collapsed_lower:
        return True

    condensed = re.sub(r"[\s_-]+", "", normalized).casefold()

    if _has_worker_recovery_marker(message, normalized_hint=normalized) and _WORKER_STALL_FUZZY_RESTART_PATTERN.search(
        normalized
    ):
        return True

    if _WORKER_STALL_CONTEXT_PATTERN.search(normalized):
        context_match = any(hint in collapsed_lower for hint in _WORKER_STALL_CONTEXT_HINTS)
        if not context_match:
            context_match = any(hint in condensed for hint in _WORKER_STALL_CONTEXT_HINTS)
        if context_match:
            return True

    return False


def _coalesce_iterable(values: Iterable[str]) -> list[str]:
    """Return *values* with duplicates removed while preserving ordering."""

    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


def _split_metadata_values(value: str | None) -> list[str]:
    """Return a list of normalised entries parsed from a metadata field."""

    if not value:
        return []

    if not isinstance(value, str):  # pragma: no cover - defensive guardrail
        value = str(value)

    tokens: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote: str | None = None

    pairing = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in pairing.items()}

    for char in value:
        if quote:
            buffer.append(char)
            if char == quote and (len(buffer) < 2 or buffer[-2] != "\\"):
                quote = None
            continue

        if char in {'"', "'"}:
            quote = char
            buffer.append(char)
            continue

        if char in pairing:
            depth += 1
            buffer.append(char)
            continue

        if char in closing and depth > 0:
            depth = max(0, depth - 1)
            buffer.append(char)
            continue

        if depth == 0 and char in {",", ";", "\n"}:
            token = "".join(buffer).strip()
            if token:
                tokens.append(token)
            buffer.clear()
            continue

        buffer.append(char)

    tail = "".join(buffer).strip()
    if tail:
        tokens.append(tail)

    return tokens


def _parse_int_sequence(value: str | None) -> tuple[int, ...]:
    """Parse a metadata field containing integer samples."""

    samples: list[int] = []
    seen: set[int] = set()
    for token in _split_metadata_values(value):
        try:
            number = int(token)
        except ValueError:
            continue
        if number in seen:
            continue
        seen.add(number)
        samples.append(number)
    return tuple(samples)


def _decode_docker_log_value(value: str) -> str:
    """Best-effort decoding of escaped Docker log field values."""

    if not value:
        return ""

    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:  # pragma: no cover - extremely defensive
        return value


def _stringify_envelope_value(value: Any) -> str | None:
    """Convert structured log payload values into displayable strings."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = value.decode("utf-8", errors="replace").strip()
        except Exception:  # pragma: no cover - defensive guardrail
            decoded = bytes(value).decode("utf-8", errors="ignore").strip()
        return decoded or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _ingest_structured_envelope(
    envelope: dict[str, str], payload: Any, prefix: str | None = None
) -> None:
    """Populate ``envelope`` with data extracted from *payload* recursively."""

    if isinstance(payload, MappingABC):
        _ingest_structured_mapping(envelope, payload, prefix)
        return

    if isinstance(payload, SequenceABC) and not isinstance(payload, (str, bytes, bytearray)):
        for index, item in enumerate(payload):
            child_prefix = f"{prefix}_{index}" if prefix else str(index)
            _ingest_structured_envelope(envelope, item, child_prefix)
        return

    if prefix and prefix not in envelope:
        text = _stringify_envelope_value(payload)
        if text:
            envelope[prefix] = text


def _ingest_structured_mapping(
    envelope: dict[str, str], mapping: Mapping[Any, Any], prefix: str | None = None
) -> None:
    """Flatten mapping-like Docker payloads into the ``envelope`` dictionary."""

    for raw_key, value in mapping.items():
        if not isinstance(raw_key, str):
            continue
        normalized_key = raw_key.strip()
        if not normalized_key:
            continue

        composite_key = (
            normalized_key if prefix is None else f"{prefix}_{normalized_key}"
        )

        if isinstance(value, MappingABC):
            _ingest_structured_mapping(envelope, value, composite_key)
            continue

        if isinstance(value, SequenceABC) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            inline_parts: list[str] = []
            for index, item in enumerate(value):
                if isinstance(item, MappingABC):
                    _ingest_structured_envelope(
                        envelope, item, f"{composite_key}_{index}"
                    )
                    continue
                text = _stringify_envelope_value(item)
                if not text:
                    continue
                inline_parts.append(text)
                if prefix is not None:
                    indexed_key = f"{composite_key}_{index}"
                    if indexed_key not in envelope:
                        envelope[indexed_key] = text
            if inline_parts:
                joined = ", ".join(inline_parts)
                if normalized_key not in envelope:
                    envelope[normalized_key] = joined
                if (
                    composite_key not in envelope
                    and composite_key != normalized_key
                ):
                    envelope[composite_key] = joined
            continue

        text = _stringify_envelope_value(value)
        if not text:
            continue

        if normalized_key not in envelope:
            envelope[normalized_key] = text
        if composite_key not in envelope and composite_key != normalized_key:
            envelope[composite_key] = text


def _ingest_json_fragments(message: str, envelope: dict[str, str]) -> None:
    """Extract JSON fragments embedded within Docker diagnostic output."""

    decoder = json.JSONDecoder()
    index = 0
    length = len(message)

    while index < length:
        char = message[index]
        if char not in "[{":
            index += 1
            continue
        try:
            payload, end = decoder.raw_decode(message, index)
        except ValueError:
            index += 1
            continue
        _ingest_structured_envelope(envelope, payload)
        index = end


def _parse_docker_log_envelope(message: str) -> dict[str, str]:
    """Extract key/value pairs embedded in structured Docker log lines."""

    if not message:
        return {}

    envelope: dict[str, str] = {}
    _ingest_json_fragments(message, envelope)

    for match in _DOCKER_LOG_FIELD_PATTERN.finditer(message):
        key = match.group("key")
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        decoded = _decode_docker_log_value(value)
        if key not in envelope:
            envelope[key] = decoded
    return envelope


def _resolve_windows_env_fallback(name: str) -> str | None:
    """Provide cross-platform fallbacks for common Windows placeholders."""

    normalized = name.upper()
    try:
        home = Path.home()
    except OSError:
        home = None

    if home is None:
        return None

    home_str = os.fspath(home)

    if normalized == "USERPROFILE":
        return home_str

    if normalized == "LOCALAPPDATA":
        return os.fspath(home / "AppData" / "Local")

    if normalized == "APPDATA":
        return os.fspath(home / "AppData" / "Roaming")

    if normalized in {"TEMP", "TMP"}:
        return os.fspath(home / "AppData" / "Local" / "Temp")

    if normalized == "HOMEPATH":
        drive = home.drive
        if drive:
            suffix = home_str[len(drive) :]
            return suffix or os.sep
        return home_str

    if normalized == "HOMEDRIVE":
        drive = home.drive
        return drive or None

    return None


def _collect_unresolved_env_tokens(value: str) -> set[str]:
    """Return unresolved environment variable placeholders found in *value*."""

    unresolved: set[str] = set()
    for match in _WINDOWS_ENV_VAR_PATTERN.finditer(value):
        unresolved.add(match.group(0))
    for match in _POSIX_ENV_VAR_PATTERN.finditer(value):
        token = match.group(0)
        if token == "$$":
            continue
        unresolved.add(token)
    return unresolved


def _expand_environment_path(value: str) -> str:
    """Expand environment variables in *value* across platforms."""

    expanded = os.path.expandvars(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        candidates = [name]
        if not _is_windows():
            for alias in (name.upper(), name.lower()):
                if alias not in candidates:
                    candidates.append(alias)
        for candidate in candidates:
            if candidate in os.environ:
                return os.environ[candidate]
        fallback = _resolve_windows_env_fallback(name)
        if fallback:
            return fallback
        return match.group(0)

    expanded = _WINDOWS_ENV_VAR_PATTERN.sub(replace, expanded)

    unresolved_tokens = _collect_unresolved_env_tokens(expanded)
    if unresolved_tokens:
        tokens = ", ".join(sorted(unresolved_tokens))
        raise BootstrapError(
            "Unable to expand environment variables in path "
            f"{value!r}; unresolved placeholder(s): {tokens}. "
            "Define the missing variables or escape literal percent/dollar "
            "symbols by doubling them."
        )

    return expanded


def _coerce_log_level(value: str | int | None) -> int:
    """Translate user provided log level into the numeric representation."""

    if value is None:
        return logging.INFO
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip().upper()
        if not candidate:
            return logging.INFO
        if candidate.isdigit():
            return int(candidate)
        level = logging.getLevelName(candidate)
        if isinstance(level, int):
            return level
    raise BootstrapError(f"Unsupported log level: {value!r}")


@dataclass(frozen=True)
class BootstrapConfig:
    """Normalized configuration derived from command-line flags."""

    skip_stripe_router: bool = False
    env_file: Path | None = None
    log_level: int = logging.INFO

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "BootstrapConfig":
        env_path_raw = namespace.env_file
        env_path: Path | None
        if env_path_raw is None:
            env_path = None
        else:
            path_string = os.fspath(env_path_raw)
            expanded = _expand_environment_path(path_string)
            # ``expandvars`` leaves values untouched when an environment
            # variable cannot be resolved; avoid introducing empty segments.
            sanitized = expanded.strip()
            if sanitized:
                env_path = Path(sanitized).expanduser()
            else:
                env_path = None
        log_level = _coerce_log_level(namespace.log_level)
        return cls(
            skip_stripe_router=namespace.skip_stripe_router,
            env_file=env_path,
            log_level=log_level,
        )

    def resolved_env_file(self) -> Path | None:
        if self.env_file is None:
            return None
        try:
            resolved = self.env_file.resolve(strict=False)
        except OSError as exc:  # pragma: no cover - environment specific
            raise BootstrapError(
                f"Unable to resolve environment file '{self.env_file}'"
            ) from exc

        if resolved.exists() and resolved.is_dir():
            raise BootstrapError(
                f"Environment file path '{resolved}' refers to a directory"
            )

        return resolved


def _ensure_parent_directory(path: Path | None) -> None:
    """Create the parent directory for *path* when it does not yet exist."""

    if path is None:
        return

    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - environment specific
        raise BootstrapError(
            f"Unable to create parent directory '{parent}' for '{path}'"
        ) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-stripe-router",
        action="store_true",
        help=(
            "Bypass the Stripe router startup verification. Useful when Stripe "
            "credentials are unavailable during local bootstraps."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Optional path to the environment file that should receive generated "
            "defaults.  When omitted the bootstrap process falls back to the "
            "standard discovery rules in bootstrap_defaults."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help=(
            "Logging level for bootstrap diagnostics. Accepts either a standard "
            "logging name (DEBUG, INFO, WARNING, ERROR, CRITICAL) or the "
            "corresponding numeric value."
        ),
    )
    return parser.parse_args(argv)


def _configure_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _apply_environment(overrides: Mapping[str, str]) -> None:
    for key, value in overrides.items():
        os.environ[key] = value


def _should_wait_for_windows_console() -> bool:
    """Return ``True`` when we should pause so Windows users can see output."""

    if os.name != "nt":  # pragma: no cover - Windows only behaviour
        return False

    if os.environ.get(_WINDOWS_VISIBILITY_SKIP_ENV):
        return False

    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    if stdin is None or stdout is None:
        return False

    if not stdin.isatty() or not stdout.isatty():
        return False

    return True


def _wait_for_windows_console_visibility() -> None:
    """Pause the process so Windows console users can read the log output."""

    if not _should_wait_for_windows_console():
        return

    message = (
        "\nAll output above has been written. Press Enter to exit and close this window."
    )
    try:
        print(message)
        sys.stdout.flush()
        input()
    except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive only
        pass


def _iter_windows_script_candidates(executable: Path) -> Iterable[Path]:
    """Yield plausible ``Scripts`` directories for the active interpreter."""

    scripts_dir = executable.with_name("Scripts")
    yield scripts_dir
    yield executable.parent / "Scripts"

    # ``sysconfig`` provides a reliable view into the interpreter layout even
    # when ``sys.executable`` points at a shim such as ``py.exe``.
    try:
        scripts_path = Path(sysconfig.get_path("scripts"))
    except (KeyError, TypeError, ValueError):
        scripts_path = None
    if scripts_path:
        yield scripts_path

    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        if prefix:
            yield Path(prefix) / "Scripts"

    venv_root = os.environ.get("VIRTUAL_ENV")
    if venv_root:
        yield Path(venv_root) / "Scripts"


def _translate_windows_path_with_root(path: Path | str, mount_root: Path) -> Path | None:
    """Translate *path* into a WSL path using the supplied mount root."""

    raw = os.fspath(path).strip()
    if not raw:
        return None

    normalized = raw.replace("/", "\\")
    try:
        windows_path = PureWindowsPath(normalized)
    except Exception:
        return None

    drive = windows_path.drive
    if drive:
        drive_letter = drive.rstrip(":").lower()
        if not drive_letter:
            return None

        relative_parts = [
            part
            for part in windows_path.parts
            if part
            and part not in {drive, windows_path.root, windows_path.anchor}
        ]

        target = mount_root / drive_letter
        if relative_parts:
            target = target.joinpath(*relative_parts)
        return target

    anchor = windows_path.anchor
    if anchor.startswith("\\\\"):
        host_share = anchor.strip("\\")
        if not host_share or "\\" not in host_share:
            return None
        server, share = host_share.split("\\", 1)
        relative = [part for part in windows_path.parts[1:] if part]
        base = mount_root / "unc" / server / share
        if relative:
            base = base.joinpath(*relative)
        return base

    return None


def _iter_wsl_configuration_paths() -> Iterable[Path]:
    """Yield known WSL configuration files that may define automount roots."""

    primary = Path("/etc/wsl.conf")
    yield primary

    conf_dir = primary.parent / "wsl.conf.d"
    try:
        if conf_dir.is_dir():
            for candidate in sorted(conf_dir.glob("*.conf")):
                yield candidate
    except OSError:
        return


def _normalise_wsl_root_candidate(raw: str) -> Path | None:
    """Return a normalised :class:`Path` for a WSL automount root value."""

    cleaned = raw.strip().strip('"').strip("'")
    if not cleaned:
        return None

    if cleaned.startswith("\\\\") or ":" in cleaned:
        translated = _translate_windows_path_with_root(cleaned, Path("/mnt"))
        if translated is not None:
            return translated
        return None

    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned.lstrip('/')}"

    return Path(cleaned)


def _discover_wsl_automount_root() -> Path | None:
    """Inspect WSL configuration files for an explicit automount root."""

    parser = configparser.ConfigParser()
    for config_path in _iter_wsl_configuration_paths():
        try:
            content = config_path.read_text(encoding="utf-8")
        except OSError:
            continue

        if not content.strip():
            continue

        parser.clear()
        try:
            parser.read_string(content)
        except (configparser.Error, UnicodeDecodeError):
            continue

        if parser.has_option("automount", "root"):
            candidate = parser.get("automount", "root")
            normalized = _normalise_wsl_root_candidate(candidate)
            if normalized is not None:
                return normalized

    return None


def _iter_wsl_mount_roots_from_proc() -> tuple[Path, ...]:
    """Return mount roots discovered from ``/proc`` mount information."""

    candidates: set[Path] = set()
    for proc_path in (Path("/proc/self/mountinfo"), Path("/proc/mounts")):
        try:
            with open(proc_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    if proc_path.name == "mountinfo":
                        mount_point = parts[4]
                        try:
                            dash_index = parts.index("-")
                        except ValueError:
                            continue
                        if dash_index + 1 >= len(parts):
                            continue
                        fs_type = parts[dash_index + 1]
                    else:
                        mount_point = parts[1]
                        fs_type = parts[2] if len(parts) > 2 else ""

                    if fs_type.lower() not in {"9p", "drvfs"}:
                        continue

                    if not mount_point.startswith("/"):
                        continue

                    mount_path = Path(mount_point)
                    name = mount_path.name.lower()
                    parent = mount_path.parent
                    if not parent:
                        continue

                    if len(name) == 1 and name.isalpha():
                        candidates.add(parent)
                    elif name == "unc":
                        candidates.add(parent)
        except OSError:
            continue

    return tuple(sorted(candidates, key=str))


def _is_viable_wsl_mount_root(candidate: Path) -> bool:
    """Return ``True`` when *candidate* resembles a WSL drive mount root."""

    try:
        entries = list(candidate.iterdir())
    except OSError:
        return False

    for entry in entries:
        name = entry.name.lower()
        if len(name) == 1 and name.isalpha():
            return True
        if name == "unc":
            return True

    return False


@lru_cache(maxsize=None)
def _get_wsl_host_mount_root() -> Path:
    """Return the root directory where Windows drives are mounted inside WSL."""

    override = os.environ.get(_WSL_HOST_MOUNT_ROOT_ENV)
    if override:
        return Path(override)

    if not _is_wsl():
        return Path("/mnt")

    candidates: list[Path] = []

    configured_root = _discover_wsl_automount_root()
    if configured_root is not None:
        candidates.append(configured_root)

    candidates.extend(_iter_wsl_mount_roots_from_proc())
    candidates.extend((Path("/mnt"), Path("/mnt/wsl"), Path("/mnt/host")))

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normpath(str(candidate))
        key = os.path.normcase(normalized)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(Path(normalized))

    for candidate in unique_candidates:
        if _is_viable_wsl_mount_root(candidate):
            return candidate

    for candidate in unique_candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue

    return Path("/mnt")


def _translate_windows_host_path(path: Path) -> Path | None:
    """Translate a Windows host path into its WSL-accessible counterpart."""

    if _is_windows():
        return None

    raw = str(path).strip()
    if not raw:
        return None

    if raw.startswith("/") or raw.startswith("\\\\wsl$"):
        return None

    if ":" not in raw and not raw.startswith("\\\\"):
        return None

    mount_root = _get_wsl_host_mount_root()
    return _translate_windows_path_with_root(raw, mount_root)


def _iter_dockerdesktop_app_directories(root: Path) -> Iterable[Path]:
    """Yield versioned Docker Desktop application directories under ``root``.

    Docker Desktop's Electron-based bundles install into per-version
    ``app-<semver>`` directories when deployed via user-mode installers (for
    example the Microsoft Store or winget).  Those directories expose
    ``resources\\bin`` folders that contain ``docker.exe``/``com.docker.cli.exe``
    and, starting with 4.30, optional ``resources\\cli-*`` overlays used by the
    Windows/WSL interoperability layer.

    Historically :func:`_iter_windows_docker_directories` only emitted a static
    list of potential directories.  That approach broke on hosts where Docker
    Desktop had been installed exclusively through the versioned app bundle
    because none of the static candidates existed, leading bootstrap to claim
    Docker was unavailable.  Enumerating the versioned directories when present
    keeps discovery resilient without hard-coding every possible version
    component.  The helper defensively catches filesystem errors so bootstrap
    remains robust on locked-down workstations that deny listing access.
    """

    app_root = root / "DockerDesktop"
    try:
        entries = list(app_root.iterdir())
    except FileNotFoundError:
        return
    except NotADirectoryError:
        return
    except PermissionError:
        LOGGER.debug("Unable to enumerate DockerDesktop app bundles under %s", app_root)
        return
    except OSError as exc:  # pragma: no cover - defensive safeguard
        LOGGER.debug("Failed to enumerate DockerDesktop app bundles: %s", exc)
        return

    pattern = re.compile(r"app-[0-9][0-9A-Za-z_.-]*", re.IGNORECASE)

    for entry in entries:
        if not entry.is_dir():
            continue
        if not pattern.fullmatch(entry.name):
            continue
        yield entry


def _iter_windows_docker_directories() -> Iterable[Path]:
    """Yield directories that commonly contain Docker Desktop CLIs on Windows."""

    def _unique(paths: Iterable[Path]) -> Iterable[Path]:
        seen: set[str] = set()
        for candidate in paths:
            normalized = os.path.normcase(str(candidate))
            if normalized in seen:
                continue
            seen.add(normalized)
            yield candidate

    program_roots: list[Path] = []
    for env_var in (
        "ProgramFiles",
        "ProgramW6432",
        "ProgramFiles(x86)",
        "ProgramFiles(Arm)",
    ):
        value = os.environ.get(env_var)
        if value:
            program_roots.append(Path(_strip_windows_quotes(value)))

    if not program_roots:
        program_roots.extend(
            Path(path)
            for path in (r"C:\\Program Files", r"C:\\Program Files (x86)")
        )
        arm_default = Path(r"C:\\Program Files (Arm)")
        if arm_default.exists():
            program_roots.append(arm_default)

    resource_suffixes: tuple[tuple[str, ...], ...] = (
        ("Docker", "Docker", "resources", "bin"),
        ("Docker", "Docker", "resources", "cli"),
        ("Docker", "Docker", "resources", "cli-wsl"),
        ("Docker", "Docker", "resources", "cli-linux"),
        ("Docker", "Docker", "resources", "cli-bin"),
        ("Docker", "Docker", "resources", "docker-cli"),
        ("Docker", "Docker", "resources", "cli-arm"),
        ("Docker", "Docker", "resources", "cli-arm64"),
    )

    app_resource_suffixes: tuple[tuple[str, ...], ...] = (
        ("resources", "bin"),
        ("resources", "cli"),
        ("resources", "cli-wsl"),
        ("resources", "cli-linux"),
        ("resources", "cli-bin"),
        ("resources", "docker-cli"),
        ("resources", "cli-arm"),
        ("resources", "cli-arm64"),
    )

    app_overlay_suffixes: tuple[tuple[str, ...], ...] = (
        ("cli",),
        ("cli-bin",),
        ("cli-tools",),
    )

    default_targets: list[Path] = []
    for root in program_roots:
        for suffix in resource_suffixes:
            default_targets.append(root.joinpath(*suffix))
        # Docker Desktop began shipping side-by-side CLI bundles that live next
        # to the legacy ``resources\\bin`` directory starting with 4.29.  Those
        # bundles mirror the Windows installer layout under ``Program Files``
        # and surface either ``docker.exe`` or ``com.docker.cli.exe`` directly.
        # Surfacing the base directory keeps discovery resilient as Docker
        # iterates on the exact folder name (``cli``, ``cli-wsl`` or
        # ``cli-linux``) without requiring bootstrap changes for each rename.
        default_targets.append(root / "Docker" / "Docker" / "cli")

    program_data = os.environ.get("ProgramData")
    if program_data:
        program_data_root = Path(_strip_windows_quotes(program_data)) / "DockerDesktop"
    else:
        program_data_root = Path(r"C:\\ProgramData") / "DockerDesktop"

    default_targets.extend(
        program_data_root / variant
        for variant in (
            Path("version-bin"),
            Path("cli"),
            Path("cli-bin"),
            Path("cli-tools"),
        )
    )

    def _extend_user_targets(root: Path) -> None:
        """Add Docker Desktop user-level installation directories."""

        candidate_bases = (root,)  # LOCALAPPDATA (e.g. C:\Users\name\AppData\Local)

        # Modern Docker Desktop installers prefer ``%LOCALAPPDATA%\Programs``
        # to avoid collisions with legacy resources placed directly under
        # ``%LOCALAPPDATA%``.  Older builds (and some manual installs) still use
        # the original layout.  Probe both aggressively so the bootstrap logic
        # continues to work when users upgrade from older releases or move
        # between insider/beta channels that experiment with the directory
        # layout.
        programs_dir = root / "Programs"
        candidate_bases += (programs_dir,)

        for base in candidate_bases:
            for suffix in resource_suffixes:
                default_targets.append(base.joinpath(*suffix))
            default_targets.append(base / "Docker" / "resources" / "cli")
            default_targets.extend(
                base / "DockerDesktop" / variant
                for variant in (
                    "version-bin",
                    "cli",
                    "cli-bin",
                    "cli-tools",
                    "cli-arm",
                    "cli-arm64",
                )
            )

            for app_dir in _iter_dockerdesktop_app_directories(base):
                for suffix in app_resource_suffixes:
                    default_targets.append(app_dir.joinpath(*suffix))
                for suffix in app_overlay_suffixes:
                    default_targets.append(app_dir.joinpath(*suffix))

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        user_root = Path(_strip_windows_quotes(local_appdata))
        _extend_user_targets(user_root)
    else:
        home_root = Path.home() / "AppData" / "Local"
        _extend_user_targets(home_root)

    for target in _unique(default_targets):
        yield target


def _convert_windows_path_to_wsl(path: Path) -> Path | None:
    """Return a WSL-accessible representation of *path* when possible."""

    translated = _translate_windows_host_path(path)
    if translated is not None:
        return translated

    candidate = Path(path)
    if not candidate.is_absolute():
        return None

    mount_root = _get_wsl_host_mount_root()
    try:
        candidate.relative_to(mount_root)
    except ValueError:
        return None

    return candidate


def _iter_wsl_docker_directories() -> Iterable[Path]:
    """Yield Docker CLI directories exposed via Windows when running inside WSL."""

    seen: set[str] = set()
    for candidate in _iter_windows_docker_directories():
        converted = _convert_windows_path_to_wsl(candidate)
        if converted is None:
            continue
        normalized = os.path.normcase(str(converted))
        if normalized in seen:
            continue
        seen.add(normalized)
        yield converted


def _iter_windows_system_roots() -> Iterable[Path]:
    """Yield Windows system roots accessible from the current execution host."""

    raw_roots: list[Path] = []

    for env_var in ("SystemRoot", "windir", "WINDIR"):
        raw = os.environ.get(env_var)
        if raw:
            raw_roots.append(Path(_strip_windows_quotes(raw)))

    if not raw_roots:
        raw_roots.append(Path(r"C:\\Windows"))

    if _is_wsl():
        mount_root = _get_wsl_host_mount_root()
        for drive_letter in ("c", "C"):
            raw_roots.append(mount_root / drive_letter / "Windows")

    seen: set[str] = set()
    for root in raw_roots:
        variants = [root]
        if _is_wsl():
            translated = _convert_windows_path_to_wsl(root)
            if translated is not None:
                variants.append(translated)

        for variant in variants:
            if variant is None:
                continue
            key = os.path.normcase(str(variant))
            if key in seen:
                continue
            seen.add(key)
            try:
                exists = variant.exists()
            except OSError:
                exists = False
            if exists:
                yield variant


def _iter_windows_system_directories() -> Iterable[Path]:
    """Yield directories that typically contain core Windows executables."""

    seen: set[str] = set()
    directories: list[Path] = []

    def _register(path: Path | None) -> None:
        if path is None:
            return
        key = os.path.normcase(str(path))
        if key in seen:
            return
        try:
            exists = path.exists()
        except OSError:
            exists = False
        if not exists:
            return
        seen.add(key)
        directories.append(path)

    for root in _iter_windows_system_roots():
        _register(root)
        for suffix in _WINDOWS_SYSTEM_DIRECTORY_SUFFIXES:
            _register(root.joinpath(*suffix))

    path_value = os.environ.get("PATH")
    if path_value:
        separators = {os.pathsep}
        if ";" in path_value:
            separators.add(";")
        for separator in separators:
            for entry in path_value.split(separator):
                entry = entry.strip()
                if not entry:
                    continue
                raw_entry = _strip_windows_quotes(entry)
                candidate = Path(raw_entry)
                _register(candidate)
                if _is_wsl():
                    _register(_convert_windows_path_to_wsl(candidate))

    for directory in directories:
        yield directory


@lru_cache(maxsize=None)
def _resolve_command_path(executable: str) -> str | None:
    """Resolve *executable* into an absolute path when possible."""

    if not executable:
        return None

    if os.path.isabs(executable):
        if os.path.exists(executable):
            return executable
        if _is_wsl():
            translated = _convert_windows_path_to_wsl(Path(executable))
            if translated is not None and translated.exists():
                return os.fspath(translated)

    if _is_wsl() and (":" in executable or executable.startswith("\\")):
        translated = _convert_windows_path_to_wsl(Path(executable))
        if translated is not None and translated.exists():
            return os.fspath(translated)

    discovered = shutil.which(executable)
    if discovered:
        return discovered

    if not (_is_windows() or _is_wsl()):
        return None

    name = Path(executable).name
    if name != executable:
        return None

    candidate_names: list[str] = []
    if not name.lower().endswith(".exe"):
        candidate_names.append(f"{name}.exe")
    candidate_names.append(name)

    inspected: set[str] = set()
    for directory in _iter_windows_system_directories():
        for candidate_name in candidate_names:
            candidate_path = directory / candidate_name
            key = os.path.normcase(str(candidate_path))
            if key in inspected:
                continue
            inspected.add(key)
            try:
                exists = candidate_path.exists()
            except OSError:
                exists = False
            if not exists:
                continue
            if os.access(candidate_path, os.X_OK):
                return os.fspath(candidate_path)

    return None


def _is_windows() -> bool:
    return os.name == "nt"


@lru_cache(maxsize=None)
def _is_wsl() -> bool:
    """Return ``True`` when executing inside the Windows Subsystem for Linux."""

    if _is_windows():
        return False

    indicators = []
    for probe in ("/proc/sys/kernel/osrelease", "/proc/version"):
        try:
            with open(probe, "r", encoding="utf-8", errors="ignore") as fh:
                indicators.append(fh.read())
        except OSError:
            continue

    signature = "\n".join(indicators)
    return "Microsoft" in signature or "WSL" in signature


def _detect_container_indicators() -> tuple[bool, str | None, tuple[str, ...]]:
    """Return containerisation hints discovered on the current host."""

    indicators: list[str] = []
    runtime: str | None = None

    for path, label in (
        (Path("/.dockerenv"), "dockerenv"),
        (Path("/run/.containerenv"), "containerenv"),
    ):
        try:
            exists = path.exists()
        except OSError:
            exists = False
        if exists:
            indicator = f"path:{label}"
            indicators.append(indicator)
            if runtime is None:
                runtime = "docker" if label == "dockerenv" else None

    for env_var in ("container", "CONTAINER", "OCI_CONTAINERS", "KUBERNETES_SERVICE_HOST"):
        value = os.getenv(env_var)
        if not value:
            continue
        indicator = f"env:{env_var.lower()}"
        indicators.append(indicator)
        if runtime is None and env_var.lower().startswith("oci"):
            runtime = "oci"
        elif runtime is None and env_var.lower().startswith("kubernetes"):
            runtime = "kubernetes"

    token_runtime_map = {
        "docker": "docker",
        "kubepods": "kubernetes",
        "containerd": "containerd",
        "crio": "cri-o",
        "podman": "podman",
        "libpod": "podman",
        "lxc": "lxc",
        "garden": "garden",
    }

    for probe in ("/proc/1/cgroup", "/proc/self/cgroup"):
        try:
            with open(probe, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    for token, token_runtime in token_runtime_map.items():
                        if token in stripped:
                            indicators.append(f"cgroup:{token}")
                            if runtime is None:
                                runtime = token_runtime
        except OSError:
            continue

    deduped_indicators = tuple(dict.fromkeys(indicators))
    return bool(deduped_indicators), runtime, deduped_indicators


def _detect_ci_indicators() -> tuple[bool, tuple[str, ...]]:
    """Detect whether execution appears to be running under CI orchestration."""

    hints: list[str] = []
    ci_markers = {
        "CI": "ci",
        "GITHUB_ACTIONS": "github-actions",
        "GITLAB_CI": "gitlab-ci",
        "BUILDKITE": "buildkite",
        "CIRCLECI": "circleci",
        "TRAVIS": "travis-ci",
        "APPVEYOR": "appveyor",
        "TF_BUILD": "azure-pipelines",
        "TEAMCITY_VERSION": "teamcity",
        "BITBUCKET_BUILD_NUMBER": "bitbucket-pipelines",
        "JENKINS_URL": "jenkins",
        "CODEBUILD_BUILD_ID": "aws-codebuild",
        "CODESPACES": "github-codespaces",
    }

    for env_var, label in ci_markers.items():
        raw = os.getenv(env_var)
        if not raw:
            continue
        normalized = raw.strip().lower()
        if env_var == "CI" and normalized in {"0", "false", "no", "off"}:
            continue
        hints.append(label)

    deduped_hints = tuple(dict.fromkeys(hints))
    return bool(deduped_hints), deduped_hints


def _detect_runtime_context() -> RuntimeContext:
    """Aggregate runtime heuristics for Docker diagnostics."""

    inside_container, container_runtime, container_indicators = _detect_container_indicators()
    is_ci, ci_indicators = _detect_ci_indicators()
    return RuntimeContext(
        platform=sys.platform,
        is_windows=_is_windows(),
        is_wsl=_is_wsl(),
        inside_container=inside_container,
        container_runtime=container_runtime,
        container_indicators=container_indicators,
        is_ci=is_ci,
        ci_indicators=ci_indicators,
    )


@lru_cache(maxsize=None)
def _windows_path_normalizer() -> Callable[[str], str]:
    """Return a callable that normalizes Windows paths for comparison."""

    def _normalize(value: str) -> str:
        collapsed = ntpath.normcase(ntpath.normpath(value))
        collapsed = collapsed.rstrip("\\/")
        return collapsed

    return _normalize


def _strip_windows_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _is_quoted_windows_value(value: str) -> bool:
    value = value.strip()
    return len(value) >= 2 and value[0] == value[-1] == '"'


def _needs_windows_path_quotes(value: str) -> bool:
    return any(symbol in value for symbol in (" ", ";", "(", ")", "&"))


def _format_windows_path_entry(value: str) -> str:
    trimmed = _strip_windows_quotes(value)
    if not trimmed:
        return trimmed
    if _needs_windows_path_quotes(trimmed):
        return f'"{trimmed}"'
    return trimmed


def _score_windows_entry(entry: str) -> tuple[int, int, int, int]:
    stripped = _strip_windows_quotes(entry)
    try:
        exists = Path(stripped).exists()
    except OSError:
        exists = False
    quotes_mismatch = int(_needs_windows_path_quotes(stripped) != _is_quoted_windows_value(entry))
    trailing_sep = int(stripped.endswith(("\\", "/")))
    return (
        0 if exists else 1,
        quotes_mismatch,
        trailing_sep,
        len(entry),
    )


def _choose_preferred_path_entry(
    existing: str,
    candidate: str,
    normalizer: Callable[[str], str],
) -> str:
    if existing == candidate:
        return existing

    existing_core = _strip_windows_quotes(existing)
    candidate_core = _strip_windows_quotes(candidate)
    existing_normalized = normalizer(existing_core)
    candidate_normalized = normalizer(candidate_core)

    if existing_normalized == candidate_normalized:
        normalized_candidate = _format_windows_path_entry(candidate)
        if normalized_candidate != candidate:
            candidate = normalized_candidate
            candidate_core = _strip_windows_quotes(candidate)
        existing_requires_quotes = _needs_windows_path_quotes(existing_core)
        candidate_requires_quotes = _needs_windows_path_quotes(candidate_core)
        existing_is_quoted = _is_quoted_windows_value(existing)
        candidate_is_quoted = _is_quoted_windows_value(candidate)
        if existing_requires_quotes and not existing_is_quoted and candidate_requires_quotes:
            return candidate
        if candidate_requires_quotes and not candidate_is_quoted and existing_requires_quotes:
            candidate = _format_windows_path_entry(candidate)
            candidate_core = _strip_windows_quotes(candidate)
        if candidate_requires_quotes and not existing_requires_quotes:
            return candidate
        if existing_requires_quotes and not candidate_requires_quotes:
            return _format_windows_path_entry(existing)
        if existing_core != candidate_core:
            return candidate
        return existing

    existing_score = _score_windows_entry(existing)
    candidate_score = _score_windows_entry(candidate)
    if existing_score == candidate_score and existing_core != candidate_core:
        return candidate
    return existing if existing_score <= candidate_score else candidate


def _gather_existing_path_entries() -> tuple[list[str], dict[str, str], bool]:
    """Collect the current PATH entries de-duplicated by Windows semantics."""

    raw_path = os.environ.get("PATH") or os.environ.get("Path") or ""
    separator = os.pathsep
    if ";" in raw_path and separator != ";":
        parts: Iterable[str] = raw_path.split(";")
    else:
        parts = raw_path.split(separator)
    entries = [entry.strip() for entry in parts if entry and entry.strip()]
    normalizer = _windows_path_normalizer()
    seen: dict[str, str] = {}
    ordered: list[str] = []
    deduplicated = False
    for entry in entries:
        try:
            normalized = normalizer(_strip_windows_quotes(entry))
        except (TypeError, ValueError):
            continue
        existing = seen.get(normalized)
        if existing is not None:
            preferred = _choose_preferred_path_entry(existing, entry, normalizer)
            if preferred != existing:
                try:
                    index = ordered.index(existing)
                except ValueError:
                    ordered.append(preferred)
                else:
                    ordered[index] = preferred
                seen[normalized] = preferred
            deduplicated = True
            continue
        seen[normalized] = entry
        ordered.append(entry)
    return ordered, seen, deduplicated


def _set_windows_path(value: str) -> None:
    os.environ["PATH"] = value
    os.environ["Path"] = value


def _ensure_windows_compatibility() -> None:
    """Augment environment defaults with Windows specific safeguards."""

    if not _is_windows():  # pragma: no cover - exercised via integration tests
        return

    scripts_dirs: list[Path] = []
    normalized_updates: list[tuple[str, str]] = []
    docker_dirs: list[Path] = []
    docker_updates: list[tuple[str, str]] = []
    executable = Path(sys.executable)
    candidates = list(_iter_windows_script_candidates(executable))

    ordered_entries, seen, deduplicated = _gather_existing_path_entries()
    normalizer = _windows_path_normalizer()

    updated = False
    for candidate in candidates:
        if not candidate:
            continue
        try:
            candidate_resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if not candidate_resolved.exists():
            continue
        key = _format_windows_path_entry(str(candidate_resolved))
        normalized_key = normalizer(_strip_windows_quotes(key))
        existing_entry = seen.get(normalized_key)
        if existing_entry is None:
            seen[normalized_key] = key
            ordered_entries.insert(0, key)
            scripts_dirs.append(candidate_resolved)
            updated = True
            continue

        preferred = _choose_preferred_path_entry(existing_entry, key, normalizer)
        if preferred == existing_entry:
            continue
        try:
            index = ordered_entries.index(existing_entry)
        except ValueError:
            ordered_entries.insert(0, preferred)
        else:
            ordered_entries[index] = preferred
        seen[normalized_key] = preferred
        normalized_updates.append((existing_entry, preferred))
        updated = True

    docker_insertion_index = len(scripts_dirs)
    docker_candidates = list(_iter_windows_docker_directories())
    for candidate in docker_candidates:
        try:
            candidate_resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if not candidate_resolved.exists():
            continue
        key = _format_windows_path_entry(str(candidate_resolved))
        normalized_key = normalizer(_strip_windows_quotes(key))
        existing_entry = seen.get(normalized_key)
        if existing_entry is None:
            seen[normalized_key] = key
            ordered_entries.insert(docker_insertion_index, key)
            docker_dirs.append(candidate_resolved)
            docker_insertion_index += 1
            updated = True
            continue

        preferred = _choose_preferred_path_entry(existing_entry, key, normalizer)
        if preferred == existing_entry:
            continue
        try:
            index = ordered_entries.index(existing_entry)
        except ValueError:
            ordered_entries.insert(docker_insertion_index, preferred)
            docker_insertion_index += 1
        else:
            ordered_entries[index] = preferred
        seen[normalized_key] = preferred
        docker_updates.append((existing_entry, preferred))
        updated = True

    if updated or deduplicated:
        new_path = os.pathsep.join(ordered_entries)
        _set_windows_path(new_path)
        if updated:
            if scripts_dirs:
                LOGGER.info(
                    "Ensured Windows PATH contains Scripts directories: %s",
                    ", ".join(str(path) for path in scripts_dirs),
                )
            if docker_dirs:
                LOGGER.info(
                    "Ensured Windows PATH contains Docker CLI directories: %s",
                    ", ".join(str(path) for path in docker_dirs),
                )
            if normalized_updates:
                LOGGER.info(
                    "Normalized existing Windows PATH entries: %s",
                    ", ".join(
                        f"{original!r} -> {updated_entry!r}"
                        for original, updated_entry in normalized_updates
                    ),
                )
            if docker_updates:
                LOGGER.info(
                    "Normalized existing Docker PATH entries: %s",
                    ", ".join(
                        f"{original!r} -> {updated_entry!r}"
                        for original, updated_entry in docker_updates
                    ),
                )
        elif deduplicated:
            LOGGER.info("Normalized Windows PATH by removing duplicate entries")

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    pathext = os.environ.get("PATHEXT")
    required_exts = (".COM", ".EXE", ".BAT", ".CMD", ".PY", ".PYW")
    if pathext:
        current = [ext.strip() for ext in pathext.split(os.pathsep) if ext]
        normalized = {ext.upper() for ext in current}
        if not set(required_exts).issubset(normalized):
            updated_exts = list(dict.fromkeys(ext.upper() for ext in current))
            for ext in required_exts:
                if ext.upper() not in normalized:
                    updated_exts.append(ext)
            os.environ["PATHEXT"] = os.pathsep.join(updated_exts)
            LOGGER.info(
                "Augmented PATHEXT with Python aware extensions: %s",
                ", ".join(required_exts),
            )
    else:
        os.environ["PATHEXT"] = os.pathsep.join(required_exts)
        LOGGER.info("Initialized PATHEXT to include Python executables")


def _prepare_environment(config: BootstrapConfig) -> Path | None:
    resolved_env_file = config.resolved_env_file()
    defaults = {
        "MENACE_ALLOW_MISSING_HF_TOKEN": "1",
        "MENACE_NON_INTERACTIVE": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    overrides: dict[str, str] = {
        "MENACE_SAFE": "0",
        "MENACE_SUPPRESS_PROMETHEUS_FALLBACK_NOTICE": "1",
        "SANDBOX_DISABLE_CLEANUP": "1",
    }
    if config.skip_stripe_router:
        overrides["MENACE_SKIP_STRIPE_ROUTER"] = "1"
    if resolved_env_file is not None:
        overrides["MENACE_ENV_FILE"] = str(resolved_env_file)
    _apply_environment(overrides)
    _ensure_parent_directory(resolved_env_file)
    _ensure_windows_compatibility()
    return resolved_env_file


@dataclass(frozen=True)
class RuntimeContext:
    """Represents key characteristics of the current execution environment."""

    platform: str
    is_windows: bool
    is_wsl: bool
    inside_container: bool
    container_runtime: str | None
    container_indicators: tuple[str, ...]
    is_ci: bool
    ci_indicators: tuple[str, ...]

    def to_metadata(self) -> dict[str, str]:
        """Return a serialisable representation suitable for diagnostics."""

        metadata: dict[str, str] = {
            "platform": self.platform,
            "is_windows": str(self.is_windows).lower(),
            "is_wsl": str(self.is_wsl).lower(),
            "inside_container": str(self.inside_container).lower(),
        }
        if self.container_runtime or self.container_indicators:
            if self.container_runtime:
                metadata["container_runtime"] = self.container_runtime
            if self.container_indicators:
                metadata["container_indicators"] = ",".join(self.container_indicators)
        if self.is_ci and self.ci_indicators:
            metadata["ci_indicators"] = ",".join(self.ci_indicators)
        return metadata


@dataclass(frozen=True)
class DockerDiagnosticResult:
    """Outcome of Docker environment verification."""

    cli_path: Path | None
    available: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    infos: tuple[str, ...]
    metadata: Mapping[str, str]
    skipped: bool = False
    skip_reason: str | None = None


def _discover_docker_cli() -> tuple[Path | None, list[str]]:
    """Locate the Docker CLI executable if available."""

    warnings: list[str] = []
    for executable in ("docker", "docker.exe", "com.docker.cli", "com.docker.cli.exe"):
        discovered = shutil.which(executable)
        if not discovered:
            continue
        path = Path(discovered)
        if _is_windows() and path.name.lower() == "com.docker.cli.exe":
            warnings.append(
                "Resolved Docker CLI via com.docker.cli.exe shim; docker.exe alias was not present on PATH"
            )
        return path, warnings

    if _is_windows():
        for directory in _iter_windows_docker_directories():
            for candidate in ("docker.exe", "com.docker.cli.exe"):
                target = directory / candidate
                if target.exists():
                    return target, warnings
        warnings.append(
            "Docker Desktop installation was not discovered in standard locations. "
            "Install Docker Desktop or ensure docker.exe is on PATH."
        )
    elif _is_wsl():
        for directory in _iter_wsl_docker_directories():
            for candidate in ("docker.exe", "com.docker.cli.exe"):
                target = directory / candidate
                if target.exists():
                    warnings.append(
                        "Using Windows Docker CLI through WSL interop. Consider enabling the "
                        "Docker Desktop WSL integration for improved reliability."
                    )
                    return target, warnings

    return None, warnings


def _run_docker_command(
    cli_path: Path,
    args: Sequence[str],
    *,
    timeout: float,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Execute a Docker CLI command and capture failures as textual diagnostics."""

    command = [str(cli_path), *args]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed, None
    except FileNotFoundError:
        return None, f"Docker executable '{cli_path}' is not accessible"
    except subprocess.TimeoutExpired:
        rendered = " ".join(args)
        return None, (
            f"Docker command '{rendered}' timed out after {timeout:.1f}s; "
            "ensure Docker Desktop is running and responsive"
        )
    except OSError as exc:  # pragma: no cover - environment specific
        rendered = " ".join(args)
        return None, f"Failed to execute docker {rendered!s}: {exc}"


def _run_command(command: Sequence[str], *, timeout: float) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Execute an arbitrary command and capture failures as diagnostics."""

    if not command:
        return None, "No command specified"

    vector = [os.fspath(part) for part in command]
    original_executable = vector[0]
    resolved_executable = _resolve_command_path(original_executable)
    if resolved_executable:
        vector[0] = resolved_executable

    try:
        completed = subprocess.run(
            vector,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed, None
    except FileNotFoundError:
        if resolved_executable and resolved_executable != original_executable:
            detail = (
                f"Executable '{original_executable}' resolved to '{resolved_executable}' "
                "but is not accessible"
            )
        else:
            detail = f"Executable '{original_executable}' is not available on PATH"
        return None, detail
    except subprocess.TimeoutExpired:
        rendered_args = " ".join(vector[1:])
        if rendered_args:
            command_preview = f"{vector[0]} {rendered_args}"
        else:
            command_preview = vector[0]
        return None, f"Command '{command_preview}' timed out after {timeout:.1f}s"
    except OSError as exc:  # pragma: no cover - environment specific
        return None, f"Failed to execute {vector[0]!s}: {exc}"


def _iter_docker_warning_messages(
    value: object, *, context: tuple[str, ...] = ()
) -> Iterable[str]:
    """Yield normalized warning strings from Docker diagnostic payloads."""

    if value is None:
        return

    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8", "ignore")
        except Exception:  # pragma: no cover - defensive fallback
            return
        yield from _iter_docker_warning_messages(decoded, context=context)
        return

    if isinstance(value, str):
        text = _strip_control_sequences(value).replace("\r", "\n")
        yield from _coalesce_warning_lines(text)
        return

    if isinstance(value, MappingABC):
        if _mapping_contains_payload_fields(value):
            rendered = _stringify_structured_warning(value, context)
            if rendered:
                yield rendered
                return
        for key, child in value.items():
            child_context = context + (str(key),)
            canonical = _canonicalize_warning_key(str(key)) if isinstance(key, str) else ""
            if canonical in _WARNING_STRUCTURED_CONTEXT_KEYS or canonical in _WARNING_STRUCTURED_MESSAGE_KEYS:
                if isinstance(child, (MappingABC, IterableABC)) and not isinstance(
                    child, (str, bytes, bytearray)
                ):
                    yield from _iter_docker_warning_messages(child, context=child_context)
                continue
            yield from _iter_docker_warning_messages(child, context=child_context)
        return

    if isinstance(value, IterableABC):
        for index, item in enumerate(value):
            child_context = context + (str(index),)
            yield from _iter_docker_warning_messages(item, context=child_context)
        return


_ANSI_ESCAPE_PATTERN = re.compile(
    r"""
    (?:
        \x1B[@-Z\\-_]
        |
        \x1B\[[0-?]*[ -/]*[@-~]
        |
        \x1B\][^\x1B]*\x1B\\
        |
        \x9B[0-?]*[ -/]*[@-~]
    )
    """,
    re.VERBOSE,
)


def _strip_control_sequences(text: str) -> str:
    """Remove ANSI escapes and non-printable control characters from *text*."""

    if not text:
        return ""

    cleaned = _ANSI_ESCAPE_PATTERN.sub("", text)
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = cleaned.translate({
        0x00: None,
        0x01: None,
        0x02: None,
        0x03: None,
        0x04: None,
        0x05: None,
        0x06: None,
        0x07: None,
        0x08: None,
        0x0B: None,
        0x0C: None,
        0x0E: None,
        0x0F: None,
        0x10: None,
        0x11: None,
        0x12: None,
        0x13: None,
        0x14: None,
        0x15: None,
        0x16: None,
        0x17: None,
        0x18: None,
        0x19: None,
        0x1A: None,
        0x1B: None,
        0x1C: None,
        0x1D: None,
        0x1E: None,
        0x1F: None,
        0x7F: None,
        0x80: None,
        0x81: None,
        0x82: None,
        0x83: None,
        0x84: None,
        0x85: None,
        0x86: None,
        0x87: None,
        0x88: None,
        0x89: None,
        0x8A: None,
        0x8B: None,
        0x8C: None,
        0x8D: None,
        0x8E: None,
        0x8F: None,
        0x90: None,
        0x91: None,
        0x92: None,
        0x93: None,
        0x94: None,
        0x95: None,
        0x96: None,
        0x97: None,
        0x98: None,
        0x99: None,
        0x9A: None,
        0x9B: None,
        0x9C: None,
        0x9D: None,
        0x9E: None,
        0x9F: None,
        0x200B: None,
        0x200C: None,
        0x200D: None,
        0x2060: None,
    })
    return cleaned


def _decode_worker_payload_bytes(payload: bytes) -> str:
    """Return a best-effort textual decoding for Docker telemetry payloads."""

    if not payload:
        return ""

    candidates: list[str] = []

    def _register(encoding: str) -> None:
        if encoding not in candidates:
            candidates.append(encoding)

    if payload.startswith(codecs.BOM_UTF16_LE):
        _register("utf-16")
        _register("utf-16-le")
    elif payload.startswith(codecs.BOM_UTF16_BE):
        _register("utf-16")
        _register("utf-16-be")
    else:
        even_slice = payload[0::2]
        odd_slice = payload[1::2]
        even_null_fraction = (even_slice.count(0) / len(even_slice)) if even_slice else 0.0
        odd_null_fraction = (odd_slice.count(0) / len(odd_slice)) if odd_slice else 0.0

        utf16_bias = 0.30
        if odd_null_fraction >= utf16_bias and even_null_fraction < utf16_bias / 2:
            _register("utf-16-le")
            _register("utf-16")
        elif even_null_fraction >= utf16_bias and odd_null_fraction < utf16_bias / 2:
            _register("utf-16-be")
            _register("utf-16")
        elif odd_null_fraction >= utf16_bias or even_null_fraction >= utf16_bias:
            _register("utf-16")

    if payload.startswith(codecs.BOM_UTF8):
        _register("utf-8-sig")

    _register("utf-8")
    _register("latin-1")

    for encoding in candidates:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue

    return payload.decode("utf-8", "replace")


def _coerce_textual_value(value: object) -> str | None:
    """Return a textual representation of ``value`` when possible."""

    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return _decode_worker_payload_bytes(payload)

    return None


def _fingerprint_worker_banner(raw_value: object) -> str | None:
    """Return a deterministic fingerprint for raw worker stall banners."""

    text = _coerce_textual_value(raw_value)
    if not text:
        return None

    cleaned = _strip_control_sequences(str(text))
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    if not collapsed:
        return None

    digest = hashlib.sha256(collapsed.encode("utf-8")).hexdigest()
    return f"{_WORKER_STALLED_SIGNATURE_PREFIX}{digest}"


def _normalize_worker_banner_characters(message: str) -> str:
    """Normalise banner punctuation emitted by Windows-localised Docker builds."""

    if not message:
        return ""

    normalized = unicodedata.normalize("NFKC", message)
    # ``docker.exe`` on Windows may emit full-width or compatibility punctuation
    # characters when the host locale is configured for East Asian languages.
    # The worker stall detectors rely on ASCII separators, so normalise these
    # variants to their half-width counterparts before applying the heuristics.
    normalized = html.unescape(normalized)
    normalized = normalized.translate(_WORKER_BANNER_CHARACTER_TRANSLATION)
    return normalized


def _contains_literal_worker_restart_banner(
    message: str, *, normalized: str | None = None
) -> bool:
    """Return ``True`` when ``message`` still carries the raw stall banner."""

    if not message and not normalized:
        return False

    candidate = normalized if normalized is not None else _normalize_worker_banner_characters(message)
    if not candidate:
        return False

    # ``docker.exe`` occasionally interleaves carriage returns when repainting
    # progress bars, leaving fragments such as ``worker stalled;\r restarting``
    # in the diagnostic stream.  Treat these control characters as hard spaces
    # so substring probes remain stable irrespective of the host terminal
    # behaviour.
    candidate = candidate.replace("\r\n", "\n").replace("\r", " ")
    if "\n" in candidate:
        candidate = re.sub(r"\s*\n\s*", " ", candidate)

    # Normalise whitespace around the separator to ensure variants like
    # ``worker stalled ; restarting`` collapse into the canonical token while
    # preserving the expected spacing used in downstream messaging.
    candidate = re.sub(r"\s*;\s*", "; ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    return "worker stalled; restarting" in candidate.casefold()


def _normalise_worker_stalled_phrase(message: str) -> str:
    """Collapse phrasing variants of ``worker has stalled`` into ``worker stalled``."""

    if not message:
        return ""

    normalized = _normalize_worker_banner_characters(message)
    normalized = _rewrite_inline_worker_contexts(normalized)
    normalized = _canonicalize_worker_stall_tokens(normalized)
    return _WORKER_STALLED_VARIATIONS_PATTERN.sub("worker stalled", normalized)


_WORKER_STALL_CAMELCASE_PATTERN = re.compile(
    r"(?i)\b(workers?)(?=(?:st(?:all|uck)|hang|hung|freez|froz))",
)


_WORKER_RECOVERY_MARKERS: tuple[str, ...] = (
    "restart",
    "restarting",
    "restart-loop",
    "restart loop",
    "reset",
    "resetting",
    "reset-loop",
    "reset loop",
    "relaunch",
    "relaunching",
    "re-launch",
    "re-launching",
    "relaunch-loop",
    "relaunch loop",
    "reinitialize",
    "reinitializing",
    "reinitialise",
    "reinitialising",
    "re-initialize",
    "re-initializing",
    "re-initialise",
    "re-initialising",
    "reinitialization",
    "reinitialisation",
    "reinit",
    "reiniting",
)

_WORKER_RECOVERY_MARKERS_COMPACT: tuple[str, ...] = tuple(
    marker.replace(" ", "").replace("-", "").replace("_", "")
    for marker in _WORKER_RECOVERY_MARKERS
)

_WORKER_RECOVERY_LOCALISED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\breinici[a-záéíóúãõçñ]*\b", re.IGNORECASE),
    re.compile(r"\breinicializ[a-záéíóúãõçñ]*\b", re.IGNORECASE),
    re.compile(r"\br[ée]initialis[a-zàâçéèêëîïôûùüÿœæ]*\b", re.IGNORECASE),
    re.compile(r"\bred[ée]marr[a-zàâçéèêëîïôûùüÿœæ]*\b", re.IGNORECASE),
    re.compile(r"\bneu[-\s]?start[a-zäöüß]*\b", re.IGNORECASE),
    re.compile(r"\briavvi[a-zàèéìòù]*\b", re.IGNORECASE),
)

_WORKER_RECOVERY_CJK_MARKERS: tuple[str, ...] = (
    "重新启动",
    "重新啟動",
    "重新開機",
    "重新开机",
    "重启",
    "重啟",
    "重启中",
    "重啟中",
    "再启动",
    "再啟動",
    "再起動",
    "再啓動",
    "再起动",
    "再起動中",
    "再啟動中",
    "再起動します",
    "再起動しています",
    "再啟動します",
    "再啟動しています",
    "재시작",
    "재시작 중",
    "재부팅",
    "재부팅 중",
    "다시 시작",
    "다시시작",
)


def _has_worker_recovery_marker(
    message: str,
    *,
    normalized_hint: str | None = None,
) -> bool:
    """Return ``True`` when *message* references a worker restart event."""

    if not message:
        return False

    lowered = message.casefold()

    if any(marker in lowered for marker in _WORKER_RECOVERY_MARKERS):
        return True

    condensed_lowered = re.sub(r"[\s_-]+", "", lowered)
    if any(marker and marker in condensed_lowered for marker in _WORKER_RECOVERY_MARKERS_COMPACT):
        return True

    normalized = normalized_hint if normalized_hint is not None else _normalize_worker_banner_characters(message)
    normalized_casefold = normalized.casefold()

    if normalized_casefold != lowered:
        if any(marker in normalized_casefold for marker in _WORKER_RECOVERY_MARKERS):
            return True
        condensed_normalized = re.sub(r"[\s_-]+", "", normalized_casefold)
        if any(marker and marker in condensed_normalized for marker in _WORKER_RECOVERY_MARKERS_COMPACT):
            return True
    else:
        condensed_normalized = condensed_lowered

    for pattern in _WORKER_RECOVERY_LOCALISED_PATTERNS:
        if pattern.search(normalized_casefold):
            return True

    for token in _WORKER_RECOVERY_CJK_MARKERS:
        if token and (token in normalized or token in normalized_casefold):
            return True

    return False


def _normalize_worker_token_case(token: str) -> str:
    """Return a ``worker`` token that preserves the source capitalisation."""

    if token.endswith(("s", "S")):
        base = token[:-1]
    else:
        base = token

    if token.isupper():
        return base.upper() + " "

    if token[0].isupper():
        return base[0].upper() + base[1:].lower() + " "

    return base.lower() + " "


_WORKER_STALL_CANONICALISERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bworker[\s_-]+stall(?!ed)\b", re.IGNORECASE), "worker stalled"),
    (re.compile(r"\bworker[\s_-]+stalling\b", re.IGNORECASE), "worker stalled"),
    (re.compile(r"\bworker[\s_-]+stalls\b", re.IGNORECASE), "worker stalled"),
    (re.compile(r"\bworker[\s_-]+stuck\b", re.IGNORECASE), "worker stalled"),
    (
        re.compile(
            r"\bworker\s+(?:has|have|had|is|are|was|were)\s+(?:been\s+)?stall(?:ed|ing)?\b",
            re.IGNORECASE,
        ),
        "worker stalled",
    ),
    (
        re.compile(
            r"\bworker\s+(?:has|have|had|is|are|was|were)\s+(?:been\s+)?stuck\b",
            re.IGNORECASE,
        ),
        "worker stalled",
    ),
    (
        re.compile(
            rf"\bworker{_WORKER_STALLED_VARIATIONS_BODY}\b",
            re.IGNORECASE | re.VERBOSE,
        ),
        "worker stalled",
    ),
    (
        re.compile(
            r"""
            \bworkers?\b
            (?:
                \s+(?:has|have|had|is|are|was|were|become|became|becoming|remains?|stays?|stayed|got|getting|gotten)
            )?
            (?:\s+(?:been|still|yet|again))?
            (?:\s+(?:marked|reported|detected))?
            (?:\s+as)?
            (?:\s+(?:chronically|persistently|repeatedly|intermittently))?
            \s*
            (?:
                non[-_\s]?responsive
                |unrespons(?:ive|iveness)?
                |not[-_\s]?respond(?:ing|ed)?
                |no[-_\s]?response
                |timed[-_\s]?out
                |timeout
                |unreach(?:able|ability)?
                |no[-_\s]?heartbeat
                |heartbeat[-_\s]?lost
                |lost[-_\s]?heartbeat
                |off[-_\s]?line
                |offline
                |disconnect(?:ed|ing|s)?
                |no[-_\s]?reply
                |non[-_\s]?reply
            )
            \b
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
        "worker stalled",
    ),
)


_WORKER_STALL_FUZZY_RESTART_PATTERN = re.compile(
    rf"\bworkers?\b.{{0,200}}?\b{_WORKER_STALL_ROOT_PATTERN}\b",
    re.IGNORECASE | re.DOTALL,
)


_WORKER_STALL_CONTEXT_PATTERN = re.compile(
    rf"\bworker\b.{{0,240}}?\b{_WORKER_STALL_ROOT_PATTERN}\b",
    re.IGNORECASE | re.DOTALL,
)


_WORKER_STALL_CONTEXT_HINTS: tuple[str, ...] = (
    "docker",
    "desktop",
    "vpnkit",
    "moby",
    "buildkit",
    "hyper-v",
    "hyperv",
    "wsl",
    "errcode",
    "component",
    "restartcount",
    "status",
    "background",
    "context=",
    "unresponsive",
    "nonresponsive",
    "respond",
    "timeout",
    "heartbeat",
    "unreachable",
    "offline",
    "disconnect",
)


_WORKER_STALLED_BANNER_PATTERN = re.compile(
    rf"""
    workers?                        # worker or workers tokens
    (?:\s+|[-_]\s*)?               # whitespace or common separators
    (?:
        (?:has|have|had|is|are|was|were)\s+ # optional auxiliary verbs
        (?:been\s+)?                # optional ``been`` filler token
    )?
    {_WORKER_STALL_ROOT_PATTERN}    # stall/stuck variations
    (?:\s+|[^\w\s])*              # permissive punctuation/spacing between clauses
    re[-\s]*(?:start|set)(?:ed|ing)? # restart/restarting/re-starting and reset/resetting variants
    """,
    re.IGNORECASE | re.VERBOSE,
)


_WORKER_STALLED_DETECTION_PATTERN = re.compile(
    rf"(\bworker\s+{_WORKER_STALL_ROOT_PATTERN})(?:\s+(?:has|have|had|is|are|was|were)\s+(?:been\s+)?)?\s+(?:detected|detection)\b",
    re.IGNORECASE,
)


def _canonicalize_worker_stall_tokens(message: str) -> str:
    """Rewrite ``worker stall`` and ``worker stalling`` phrases to ``worker stalled``.

    Docker Desktop on Windows occasionally emits diagnostics such as
    ``worker stall detected; restarting``.  Earlier normalisation only matched
    the literal ``worker stalled`` token which meant these variants slipped
    through and the raw banner leaked into user facing logs.  To keep the
    warning handling resilient we proactively harmonise these tokens so later
    processing can treat every variation consistently.
    """

    if not message:
        return ""

    lowered = message.casefold()
    if "worker" not in lowered:
        return message

    if not any(keyword in lowered for keyword in _WORKER_STALL_KEYWORD_TOKENS):
        return message

    # Restrict rewrites to situations that look like restart diagnostics so we
    # avoid clobbering unrelated log lines that mention stalls in other
    # contexts (for example, a "queue worker stall threshold" configuration).

    message = _WORKER_STALL_CAMELCASE_PATTERN.sub(
        lambda match: _normalize_worker_token_case(match.group(1)),
        message,
    )
    if not _has_worker_recovery_marker(message, normalized_hint=message):
        return message

    canonical = message

    # Canonicalise plural ``workers`` (and ``worker(s)``) into the singular form so
    # that downstream detection logic can treat "workers stalled" banners emitted
    # by newer Docker Desktop builds the same way as the long-standing singular
    # variant.  The plural phrases always appear alongside stall/restart
    # diagnostics which means the substitution remains specific to Docker worker
    # telemetry rather than generic log output mentioning unrelated worker pools.
    canonical = re.sub(
        r"worker\s*\(\s*s\s*\)",
        "workers",
        canonical,
        flags=re.IGNORECASE,
    )
    canonical = re.sub(
        r"\bworkers\b",
        "worker",
        canonical,
        flags=re.IGNORECASE,
    )

    for pattern, replacement in _WORKER_STALL_CANONICALISERS:
        canonical = pattern.sub(replacement, canonical)

    canonical = _WORKER_STALLED_DETECTION_PATTERN.sub(r"\1", canonical)

    return canonical


_DOCKER_WARNING_PREFIX_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        warn(?:ing)?
        (?:\[[^\]]+\])?
        (?:[:\-]|::)?
        \s*
        |
        (?:warn|warning)\[[^\]]+\]\s+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORKER_VALUE_PATTERN = (
    r"(?:\"[^\"]+\"|'[^']+'|[A-Za-z0-9_.:/\\-]+(?:\s+(?![A-Za-z0-9_.:/\\-]+\s*(?:=|:))[A-Za-z0-9_.:/\\-]+)*)"
)

_WORKER_CONTEXT_PREFIX_PATTERN = re.compile(
    r"""
    (?P<context>[A-Za-z0-9_.:/\\-]+(?:\s+[A-Za-z0-9_.:/\\-]+)*)
    \s*
    (?:
        [:\-…]|::|->|=>|—|–|→|⇒
    )
    \s*
    worker\s+stalled
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORKER_CONTEXT_BASE_KEYS = (
    "context",
    "component",
    "module",
    "id",
    "name",
    "worker",
    "scope",
    "subsystem",
    "service",
    "pipeline",
    "task",
    "unit",
    "process",
    "engine",
    "backend",
    "runner",
    "channel",
    "queue",
    "thread",
    "target",
    "namespace",
    "project",
    "group",
    "agent",
    "executor",
    "handler",
)

_WORKER_CONTEXT_BASE_KEY_TOKENS: frozenset[str] = frozenset(
    key.lower() for key in _WORKER_CONTEXT_BASE_KEYS
)

_WORKER_CONTEXT_SEVERITY_TOKENS: frozenset[str] = frozenset(
    {
        "warn",
        "warning",
        "error",
        "err",
        "info",
        "notice",
        "debug",
        "fatal",
        "crit",
        "critical",
        "trace",
    }
)

_WORKER_CONTEXT_LOG_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "level",
        "severity",
        "time",
        "timestamp",
        "ts",
        "date",
        "datetime",
        "time_local",
        "time_utc",
        "logger",
        "category",
        "msg",
        "message",
    }
)

_WORKER_CONTEXT_SEVERITY_PATTERN = re.compile(
    r"^(?:warn|warning|error|err|info|notice|debug|fatal|crit|critical|trace)(?:\[[0-9]+\])?$",
    re.IGNORECASE,
)

_WORKER_CONTEXT_NOISE_TOKENS = {
    "backoff",
    "backoffs",
    "delay",
    "delays",
    "cooldown",
    "cooldowns",
    "interval",
    "intervals",
    "retry",
    "retries",
    "restart",
    "restarts",
    "restarting",
    "restartcount",
    "restartcounts",
    "attempt",
    "attempts",
    "attemptcount",
    "attemptcounts",
    "wait",
    "waiting",
    "code",
    "codes",
    "errcode",
    "errorcode",
    "stall",
    "stalled",
    "stalling",
    "stalls",
    "stuck",
}

_WORKER_CONTEXT_NOISE_LEADING = {"next", "pending", "upcoming", "future"}
_WORKER_CONTEXT_CAUSAL_LEADING = {
    "after",
    "as",
    "because",
    "caused",
    "causing",
    "due",
    "from",
    "owing",
    "since",
    "thanks",
}

_WORKER_CONTEXT_DURATION_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)?)(?:ms|msec|milliseconds|s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)?$",
    flags=re.IGNORECASE,
)

_WORKER_CONTEXT_KEY_PATTERN = rf"(?P<key>(?:{'|'.join(_WORKER_CONTEXT_BASE_KEYS)})(?:(?:[._-][A-Za-z0-9]+)|(?:[A-Z][a-z0-9]+)|(?:\d+))*)"

_WORKER_CONTEXT_KV_PATTERN = re.compile(
    rf"{_WORKER_CONTEXT_KEY_PATTERN}\s*(?:=|:)\s*(?P<value>{_WORKER_VALUE_PATTERN})",
    re.IGNORECASE,
)

_WORKER_CONTEXT_RESTART_PATTERN = re.compile(
    rf"restarting(?:\s+worker)?\s+(?P<context>{_WORKER_VALUE_PATTERN})",
    re.IGNORECASE,
)

_WORKER_CONTEXT_STALLED_PATTERN = re.compile(
    rf"worker\s+(?P<context>{_WORKER_VALUE_PATTERN})\s+stalled",
    re.IGNORECASE,
)

_WORKER_CONTEXT_LEADING_PATTERN = re.compile(
    rf"(?P<context>{_WORKER_VALUE_PATTERN})\s+worker\s+stalled",
    re.IGNORECASE,
)

_WORKER_CONTEXT_BRACKET_PATTERN = re.compile(
    r"\[(?P<context>[^\]\s][^\]]*?)\]\s*(?:worker\s+)?stalled",
    re.IGNORECASE,
)

_WORKER_METADATA_TOKEN_PATTERN = re.compile(
    rf"(?P<key>[A-Za-z0-9_.-]+)\s*(?:=|:)\s*(?P<value>{_WORKER_VALUE_PATTERN})",
    re.IGNORECASE,
)

_WORKER_METADATA_HEURISTIC_KEYWORDS = {
    "component",
    "context",
    "module",
    "subsystem",
    "worker",
    "namespace",
    "service",
    "scope",
    "target",
    "restarts",
    "restart",
    "backoff",
    "last_restart",
    "last-restart",
    "last error",
    "last_error",
    "error",
    "reason",
    "err",
    "retry",
    "attempt",
}


def _iter_structured_json_tokens(message: str) -> Iterable[tuple[str, str]]:
    """Yield ``(key, json_text)`` pairs for inline JSON diagnostic fields."""

    if not message:
        return

    length = len(message)
    index = 0

    while index < length:
        match = re.search(r"([A-Za-z0-9_.-]+)\s*(=|:)\s*([\[{])", message[index:])
        if not match:
            break
        key = match.group(1)
        opener = match.group(3)
        closer = "}" if opener == "{" else "]"
        start = index + match.start(3)
        cursor = start
        depth = 0
        in_string: str | None = None
        escaped = False

        while cursor < length:
            char = message[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
            else:
                if char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        cursor += 1
                        break
                elif char in {'"', "'"}:
                    in_string = char
            cursor += 1

        if depth != 0:
            break

        value = message[start:cursor]
        yield key, value
        index = cursor


def _unwrap_structured_payload(value: str) -> str:
    """Return *value* without redundant quoting around JSON payloads."""

    candidate = value.strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
        inner = candidate[1:-1]
        if inner and inner[0] in "[{":
            candidate = inner
    return candidate


def _coerce_json_like_scalar(token: str) -> str:
    """Normalise JSON-like scalars so ``ast.literal_eval`` can parse them."""

    replacements = {
        "true": "True",
        "false": "False",
        "null": "None",
    }

    def _replace(match: re.Match[str]) -> str:
        value = match.group(0)
        return replacements.get(value.lower(), value)

    return re.sub(r"\b(?:true|false|null)\b", _replace, token, flags=re.IGNORECASE)


def _maybe_parse_structured_value(value: object) -> Any | None:
    """Best-effort decoding of JSON-like payloads embedded in diagnostics."""

    if isinstance(value, MappingABC):
        return value
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if not isinstance(value, str):
        return None

    candidate = _unwrap_structured_payload(value)
    if not candidate or candidate[0] not in "[{":
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        python_candidate = _coerce_json_like_scalar(candidate)
        return ast.literal_eval(python_candidate)
    except (SyntaxError, ValueError):
        return None


def _merge_structured_error_metadata(
    target: dict[str, str], incoming: Mapping[str, str]
) -> None:
    """Merge structured worker error metadata into ``target`` without clobbering."""

    for key, value in incoming.items():
        if not value:
            continue

        if key == "docker_worker_last_error_code":
            existing = target.get(key)
            if existing and existing != value:
                combined = [existing]
                existing_multi = target.get("docker_worker_last_error_codes")
                if existing_multi:
                    for token in _split_metadata_values(existing_multi):
                        if token not in combined:
                            combined.append(token)
                if value not in combined:
                    combined.append(value)
                target["docker_worker_last_error_codes"] = ", ".join(combined)
            else:
                target.setdefault(key, value)
            continue

        if key == "docker_worker_last_error_codes":
            existing = target.get(key)
            if existing:
                merged = list(dict.fromkeys(_split_metadata_values(existing)))
                for token in _split_metadata_values(value):
                    if token not in merged:
                        merged.append(token)
                target[key] = ", ".join(merged)
            else:
                target[key] = value
            continue

        target.setdefault(key, value)


def _extract_structured_error_details(payload: Any) -> tuple[str | None, dict[str, str]]:
    """Return an informative error message and metadata from *payload*."""

    prioritized_messages: list[str] = []
    fallback_messages: list[str] = []
    codes: list[str] = []

    def _register_message(bucket: list[str], candidate: Any) -> None:
        if candidate is None:
            return
        if isinstance(candidate, (MappingABC, SequenceABC)) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            for item in candidate:
                _register_message(bucket, item)
            return
        text = _stringify_envelope_value(candidate)
        if not text:
            return
        cleaned = _clean_worker_metadata_value(text)
        if cleaned:
            bucket.append(cleaned)

    def _register_code(candidate: Any) -> None:
        if candidate is None:
            return
        text = _stringify_envelope_value(candidate)
        if not text:
            return
        cleaned = _clean_worker_metadata_value(text)
        if cleaned:
            codes.append(cleaned)

    def _walk(node: Any) -> None:
        if isinstance(node, MappingABC):
            for raw_key, value in node.items():
                if not isinstance(raw_key, str):
                    continue
                canonical = _canonicalize_warning_key(raw_key)
                if canonical in {"detail", "description", "reason", "cause"}:
                    _register_message(prioritized_messages, value)
                elif canonical in {"message", "msg", "summary", "status"}:
                    _register_message(fallback_messages, value)
                elif canonical in {"code", "err_code", "error_code"}:
                    _register_code(value)

                if canonical in {"error", "last_error", "lasterror", "failure"}:
                    _walk(value)
                    continue

                if isinstance(value, (MappingABC, SequenceABC)) and not isinstance(
                    value, (str, bytes, bytearray)
                ):
                    _walk(value)
        elif isinstance(node, SequenceABC) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                _walk(item)

    _walk(payload)

    metadata: dict[str, str] = {}

    if codes:
        unique_codes: list[str] = []
        for code in codes:
            if code not in unique_codes:
                unique_codes.append(code)
        metadata["docker_worker_last_error_code"] = unique_codes[0]
        if len(unique_codes) > 1:
            metadata["docker_worker_last_error_codes"] = ", ".join(unique_codes)

    def _select_message(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if not _contains_worker_stall_signal(candidate):
                return candidate
        return candidates[0] if candidates else None

    message = _select_message(prioritized_messages) or _select_message(fallback_messages)
    if not message:
        return None, metadata

    compact_raw = re.sub(r"\s+", " ", message).strip()
    if not compact_raw:
        return None, metadata

    sanitised = _sanitize_worker_banner_text(compact_raw)
    metadata.setdefault("docker_worker_last_error_structured_message", sanitised)

    if sanitised != compact_raw:
        metadata.setdefault(
            "docker_worker_last_error_structured_message_raw",
            compact_raw,
        )
        fingerprint = _fingerprint_worker_banner(compact_raw)
        if fingerprint:
            metadata.setdefault(
                "docker_worker_last_error_structured_message_fingerprint",
                fingerprint,
            )

    return sanitised, metadata


def _looks_like_worker_metadata_line(line: str) -> bool:
    """Heuristically determine whether ``line`` contains worker metadata."""

    if not line:
        return False

    token_iter = _WORKER_METADATA_TOKEN_PATTERN.finditer(line)
    for _ in token_iter:
        return True

    lowered = line.casefold()
    return any(keyword in lowered for keyword in _WORKER_METADATA_HEURISTIC_KEYWORDS)


_WORKER_CONTINUATION_PREFIXES: tuple[str, ...] = (
    "restarting",
    "restart",
    "backoff",
    "due",
    "because",
    "component",
    "context",
    "errcode",
    "error",
    "last",
    "status",
    "telemetry",
    "metadata",
)


def _should_merge_worker_continuation(previous: str, current: str) -> bool:
    """Return ``True`` when ``current`` continues a worker stall banner."""

    if not previous or not current:
        return False

    prev_lower = previous.casefold()
    curr_lower = current.casefold()

    if curr_lower.startswith(_WORKER_CONTINUATION_PREFIXES):
        return True

    if prev_lower.endswith((";", ":", "-", "—", "…")):
        return True

    if prev_lower.endswith("worker stalled") and "restarting" in curr_lower:
        return True

    if "restarting" in curr_lower and "worker" not in curr_lower:
        return True

    if "errcode" in curr_lower or "restart" in curr_lower:
        return True

    return False


def _coalesce_warning_lines(payload: str) -> Iterable[str]:
    """Combine continuation lines emitted as part of structured warnings."""

    pending: list[str] = []
    pending_is_worker_warning = False

    for raw_line in payload.split("\n"):
        if not raw_line:
            continue
        stripped = raw_line.strip()
        if not stripped:
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        line_reports_worker = _contains_worker_stall_signal(stripped)
        looks_like_metadata = _looks_like_worker_metadata_line(stripped)

        if pending:
            if indent > 0:
                pending.append(stripped)
                pending_is_worker_warning = pending_is_worker_warning or line_reports_worker
                continue
            if (
                pending_is_worker_warning
                and looks_like_metadata
                and not line_reports_worker
            ):
                pending.append(stripped)
                continue
            if pending_is_worker_warning and _should_merge_worker_continuation(
                pending[-1], stripped
            ):
                pending.append(stripped)
                pending_is_worker_warning = pending_is_worker_warning or line_reports_worker
                continue

            yield " ".join(pending)
            pending = []
            pending_is_worker_warning = False

        pending.append(stripped)
        pending_is_worker_warning = pending_is_worker_warning or line_reports_worker

    if pending:
        yield " ".join(pending)


_WARNING_CONTEXT_PATH_IGNORED_TOKENS = {
    "",
    "warning",
    "warnings",
    "detail",
    "details",
    "diagnostic",
    "diagnostics",
    "telemetry",
    "data",
    "payload",
    "entries",
    "items",
    "list",
    "values",
    "status",
    "message",
    "messages",
    "worker",
    "workers",
    "component",
    "components",
    "context",
    "contexts",
    "info",
    "information",
    "metadata",
    "value",
}

_WARNING_STRUCTURED_CONTEXT_KEYS = (
    "context",
    "component_display_name",
    "componentdisplayname",
    "component_friendly_name",
    "componentfriendlyname",
    "friendly_name",
    "friendlyname",
    "component_title",
    "componenttitle",
    "component_label",
    "componentlabel",
    "display_name",
    "displayname",
    "component",
    "component_name",
    "componentname",
    "component_id",
    "componentid",
    "component_identifier",
    "componentidentifier",
    "component_uid",
    "componentuid",
    "component_guid",
    "componentguid",
    "component_slug",
    "componentslug",
    "name",
    "worker",
    "module",
    "service",
    "scope",
    "subsystem",
    "target",
    "unit",
    "process",
    "pipeline",
    "channel",
    "namespace",
    "source",
    "origin",
)

def _augment_with_localized_variants(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Extend ``keys`` with ``localized``/``localised`` prefixed variants."""

    variants: list[str] = list(dict.fromkeys(keys))
    seen: set[str] = set(variants)

    for key in keys:
        lowered = key.lower()
        if lowered.startswith("localized") or lowered.startswith("localised"):
            continue

        normalized = key.strip()
        if not normalized:
            continue

        compact = normalized.replace("_", "")
        has_separator = "_" in normalized

        for prefix in ("localized", "localised"):
            with_separator = (
                f"{prefix}_{normalized}" if has_separator else f"{prefix}_{normalized}"
            )
            if with_separator not in seen:
                variants.append(with_separator)
                seen.add(with_separator)

            compact_value = compact if has_separator else normalized
            without_separator = f"{prefix}{compact_value}"
            if without_separator not in seen:
                variants.append(without_separator)
                seen.add(without_separator)

    return tuple(variants)


_WARNING_STRUCTURED_MESSAGE_KEYS_BASE = (
    "status",
    "msg",
    "message",
    "warning",
    "detail",
    "description",
    "summary",
    "status_long_message",
    "statuslongmessage",
    "status_long_text",
    "statuslongtext",
    "status_text",
    "status_message",
    "statusmessage",
    "status_short_message",
    "statusshortmessage",
    "status_detail_text",
    "statusdetailtext",
    "status_body",
    "statusbody",
    "short_message",
    "shortmessage",
    "short_text",
    "shorttext",
    "short_error_message",
    "shorterrormessage",
    "long_message",
    "longmessage",
    "long_text",
    "longtext",
    "text",
    "status_detail",
    "statusdetail",
    "title",
    "headline",
    "body",
)

_WARNING_STRUCTURED_MESSAGE_KEYS = _augment_with_localized_variants(
    _WARNING_STRUCTURED_MESSAGE_KEYS_BASE
)

_WARNING_PAYLOAD_FIELD_MARKERS_BASE = {
    "status",
    "msg",
    "message",
    "warning",
    "detail",
    "description",
    "summary",
    "status_long_message",
    "status_long_text",
    "status_message",
    "status_short_message",
    "status_detail_text",
    "status_body",
    "long_message",
    "long_text",
    "text",
    "status_text",
    "status_detail",
    "statusdetail",
    "title",
    "headline",
    "body",
    "short_message",
    "short_text",
    "short_error_message",
    "last_error",
    "last_error_message",
    "last_error_code",
    "error_code",
    "error",
    "err",
    "err_code",
    "errcode",
    "restart",
    "restart_count",
    "restartcounts",
    "restartattempt",
    "restart_attempts_total",
    "restart_total",
    "total_restart_attempts",
    "total_restart",
    "backoff",
    "backoff_interval",
    "backoffseconds",
    "backoff_interval_seconds",
    "backoff_ms",
    "backoff_millis",
    "backoff_milliseconds",
    "backoff_duration",
}

_WARNING_PAYLOAD_FIELD_MARKERS = set(
    _augment_with_localized_variants(tuple(_WARNING_PAYLOAD_FIELD_MARKERS_BASE))
)

_WARNING_METADATA_TOKEN_ALIASES: Mapping[str, str] = {
    "restart": "restartCount",
    "restarts": "restartCount",
    "restart_count": "restartCount",
    "restartcount": "restartCount",
    "restartcounts": "restartCount",
    "restart_attempt": "restartCount",
    "restartattempt": "restartCount",
    "restartattempts": "restartCount",
    "restart_attempts": "restartCount",
    "restart_attempts_total": "restartCount",
    "restartattemptstotal": "restartCount",
    "restart_total": "restartCount",
    "restarttotal": "restartCount",
    "total_restart_attempts": "restartCount",
    "total_restart": "restartCount",
    "totalrestartattempts": "restartCount",
    "totalrestart": "restartCount",
    "attempt": "restartCount",
    "attempts": "restartCount",
    "retry": "restartCount",
    "retry_count": "restartCount",
    "retrycount": "restartCount",
    "tries": "restartCount",
    "trycount": "restartCount",
    "bouncecount": "restartCount",
    "backoff": "backoff",
    "backoff_interval": "backoff",
    "backoffinterval": "backoff",
    "backoff_interval_ms": "backoff",
    "backoffintervalms": "backoff",
    "backoff_interval_seconds": "backoff",
    "backoffintervalseconds": "backoff",
    "backoff_ms": "backoff",
    "backoff_millis": "backoff",
    "backoff_milliseconds": "backoff",
    "backoff_seconds": "backoff",
    "backoffseconds": "backoff",
    "backoff_duration": "backoff",
    "delay": "backoff",
    "wait": "backoff",
    "cooldown": "backoff",
    "next_restart": "backoff",
    "nextrestart": "backoff",
    "next_restart_in": "backoff",
    "nextrestartin": "backoff",
    "next_restart_seconds": "backoff",
    "nextrestartseconds": "backoff",
    "next_restart_sec": "backoff",
    "nextrestartsec": "backoff",
    "next_restart_secs": "backoff",
    "nextrestartsecs": "backoff",
    "next_restart_s": "backoff",
    "nextrestarts": "backoff",
    "next_restart_ms": "backoff",
    "nextrestartms": "backoff",
    "next_restart_millis": "backoff",
    "nextrestartmillis": "backoff",
    "next_restart_milliseconds": "backoff",
    "nextrestartmilliseconds": "backoff",
    "next_start": "backoff",
    "nextstart": "backoff",
    "next_start_in": "backoff",
    "nextstartin": "backoff",
    "next_start_ms": "backoff",
    "nextstartms": "backoff",
    "next_start_seconds": "backoff",
    "nextstartseconds": "backoff",
    "next_retry": "backoff",
    "nextretry": "backoff",
    "next_retry_in": "backoff",
    "nextretryin": "backoff",
    "next_retry_ms": "backoff",
    "nextretryms": "backoff",
    "next_retry_seconds": "backoff",
    "nextretryseconds": "backoff",
    "retry_after": "backoff",
    "retryafter": "backoff",
    "retry_after_ms": "backoff",
    "retryafterms": "backoff",
    "retry_after_seconds": "backoff",
    "retryafterseconds": "backoff",
    "retry_delay": "backoff",
    "retrydelay": "backoff",
    "retry_delay_ms": "backoff",
    "retrydelayms": "backoff",
    "retry_delay_seconds": "backoff",
    "retrydelayseconds": "backoff",
    "restart_delay": "backoff",
    "restartdelay": "backoff",
    "restart_delay_ms": "backoff",
    "restartdelayms": "backoff",
    "restart_delay_seconds": "backoff",
    "restartdelayseconds": "backoff",
    "lasterror": "lastError",
    "last_error": "lastError",
    "last_error_message": "lastError",
    "lasterrormessage": "lastError",
    "error": "lastError",
    "err": "lastError",
    "reason": "lastError",
    "failure": "lastError",
    "failreason": "lastError",
    "cause": "lastError",
    "description": "lastError",
    "lasterrorcode": "errCode",
    "last_error_code": "errCode",
    "err_code": "errCode",
    "errcode": "errCode",
    "error_code": "errCode",
    "errorcode": "errCode",
    "last_restart": "lastRestart",
    "lastrestart": "lastRestart",
    "last_seen": "lastRestart",
    "lastseen": "lastRestart",
    "last_start": "lastRestart",
    "laststart": "lastRestart",
    "last_success": "lastRestart",
    "lastsuccess": "lastRestart",
}

_STRUCTURED_METADATA_PREFIXES = (
    "metadata_",
    "meta_",
    "details_",
    "detail_",
    "diagnostic_",
    "diagnostics_",
    "info_",
    "information_",
    "telemetry_",
    "payload_",
    "data_",
)


def _canonicalize_warning_key(key: str) -> str:
    """Return a lowercase snake_case token suitable for structured analysis."""

    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", key.strip())
    if not sanitized:
        return ""
    sanitized = re.sub(r"(?<!^)(?=[A-Z])", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_").lower()


def _mapping_contains_payload_fields(mapping: Mapping[Any, Any]) -> bool:
    """Return True when *mapping* resembles a worker warning payload."""

    for raw_key in mapping.keys():
        if not isinstance(raw_key, str):
            continue
        canonical = _canonicalize_warning_key(raw_key)
        if canonical in _WARNING_PAYLOAD_FIELD_MARKERS:
            return True
    return False


def _derive_warning_context_from_path(context: tuple[str, ...]) -> str | None:
    """Return a human friendly context extracted from the traversal path."""

    candidates: list[str] = []
    for token in context:
        normalized = str(token).strip()
        if not normalized:
            continue
        canonical = _canonicalize_warning_key(normalized)
        if not canonical or canonical.isdigit():
            continue
        if canonical in _WARNING_CONTEXT_PATH_IGNORED_TOKENS:
            continue
        candidates.append(normalized)

    if not candidates:
        return None

    return candidates[-1] if len(candidates) == 1 else " ".join(candidates)


def _normalize_backoff_metadata_value(key: str, value: str) -> str:
    """Render a structured backoff value into a consistent textual form."""

    cleaned = _clean_worker_metadata_value(value)
    if not cleaned:
        return ""

    numeric: float | None = None
    try:
        numeric = float(cleaned)
    except ValueError:
        numeric = None

    canonical = key.lower()
    if numeric is not None:
        if "ms" in canonical or "millis" in canonical:
            seconds = numeric / 1000.0
            if seconds >= 1.0:
                if abs(seconds - round(seconds)) < 1e-9:
                    return f"{int(round(seconds))}s"
                return ("%g" % seconds).rstrip("0").rstrip(".") + "s"
            return f"{int(round(numeric))}ms"
        if "min" in canonical:
            seconds = numeric * 60.0
            if abs(seconds - round(seconds)) < 1e-9:
                return f"{int(round(seconds))}s"
            return ("%g" % seconds).rstrip("0").rstrip(".") + "s"
        if abs(numeric - round(numeric)) < 1e-9:
            return f"{int(round(numeric))}s"
        return ("%g" % numeric).rstrip("0").rstrip(".") + "s"

    normalized = _normalise_backoff_hint(cleaned)
    if not normalized:
        return cleaned

    prefix_render: str | None = None
    suffix_render: str | None = None
    body = normalized

    prefix_match = _APPROX_PREFIX_PATTERN.match(body)
    if prefix_match:
        prefix_render = (
            _normalise_approx_prefix(prefix_match.group("prefix"))
            or prefix_match.group("prefix").strip()
        )
        body = body[prefix_match.end() :].lstrip()

    suffix_match = _APPROX_SUFFIX_PATTERN.search(body)
    if suffix_match and suffix_match.end() == len(body):
        suffix_render = suffix_match.group(0).strip()
        body = body[: suffix_match.start()].rstrip()

    seconds = _estimate_backoff_seconds(body)
    if seconds is not None:
        rendered = _render_backoff_seconds(abs(seconds))
        if seconds < 0:
            rendered = f"-{rendered}"
        if prefix_render:
            if prefix_render == "~":
                rendered = f"~{rendered}"
            else:
                rendered = f"{prefix_render} {rendered}".strip()
        if suffix_render:
            rendered = f"{rendered} {suffix_render}".strip()
        return rendered

    if prefix_render or suffix_render:
        fragments = []
        if prefix_render:
            fragments.append(prefix_render)
        if body:
            fragments.append(body)
        if suffix_render:
            fragments.append(suffix_render)
        return " ".join(fragment for fragment in fragments if fragment)

    return body or normalized


def _format_warning_metadata_token(
    canonical_key: str, value: str
) -> tuple[str, str] | None:
    """Return a normalized ``(key, value)`` pair for structured warning metadata."""

    cleaned = _clean_worker_metadata_value(value)
    if not cleaned:
        return None

    alias = _WARNING_METADATA_TOKEN_ALIASES.get(canonical_key)
    effective_key = canonical_key

    if alias is None:
        for prefix in _STRUCTURED_METADATA_PREFIXES:
            if canonical_key.startswith(prefix):
                candidate = canonical_key[len(prefix) :]
                alias = _WARNING_METADATA_TOKEN_ALIASES.get(candidate)
                if alias is not None:
                    effective_key = candidate
                    break

    if alias is None and effective_key.endswith("code"):
        alias = "errCode"

    if alias is None:
        return None

    normalized_value = cleaned
    lowered_alias = alias.lower()

    if lowered_alias == "restartcount":
        match = re.search(r"-?\d+", cleaned)
        if not match:
            return None
        normalized_value = match.group(0)
    elif lowered_alias == "backoff":
        normalized_value = _normalize_backoff_metadata_value(effective_key, cleaned)
    return alias, normalized_value


def _render_warning_token(key: str, value: str) -> str:
    """Render ``key=value`` pairs while preserving readability."""

    if not value:
        return key
    if re.search(r"\s", value) or any(ch in value for ch in {'"', "'"}):
        return f"{key}={json.dumps(value, ensure_ascii=False)}"
    return f"{key}={value}"


def _stringify_structured_warning(
    mapping: Mapping[Any, Any], context: tuple[str, ...]
) -> str | None:
    """Convert structured Docker warning payloads into textual diagnostics."""

    if not _mapping_contains_payload_fields(mapping):
        return None

    envelope: dict[str, str] = {}
    _ingest_structured_mapping(envelope, mapping)
    if not envelope:
        return None

    canonical: dict[str, str] = {}
    for raw_key, raw_value in envelope.items():
        canonical_key = _canonicalize_warning_key(raw_key)
        if not canonical_key:
            continue
        canonical.setdefault(canonical_key, raw_value)

    context_hint: str | None = None
    for candidate in _WARNING_STRUCTURED_CONTEXT_KEYS:
        value = canonical.get(candidate)
        if not value:
            continue
        cleaned = _clean_worker_metadata_value(value)
        if cleaned:
            context_hint = cleaned
            break
    if not context_hint:
        path_hint = _derive_warning_context_from_path(context)
        if path_hint:
            context_hint = _clean_worker_metadata_value(path_hint) or path_hint

    def _select_preferred_warning_message() -> str | None:
        """Return the most informative textual summary from ``canonical``."""

        candidates: list[tuple[int, int, str]] = []

        for field in _WARNING_STRUCTURED_MESSAGE_KEYS:
            value = canonical.get(field)
            if not value:
                continue
            cleaned = _clean_worker_metadata_value(value)
            if not cleaned:
                continue

            normalized = cleaned.casefold()
            score = 0

            if _contains_worker_stall_signal(cleaned):
                score += 100
            elif "restart" in normalized or "error" in normalized:
                score += 35

            if field in {
                "status_long_message",
                "statuslongmessage",
                "status_long_text",
                "statuslongtext",
                "long_message",
                "longmessage",
                "long_text",
                "longtext",
            }:
                score += 40
            elif field in {
                "status_message",
                "statusmessage",
                "status_text",
                "status_detail_text",
                "statusdetailtext",
                "status_body",
                "statusbody",
            }:
                score += 25
            elif field in {
                "message",
                "warning",
                "detail",
                "description",
                "summary",
                "text",
            }:
                score += 20
            elif field == "status":
                score -= 10

            # Prefer richer narratives when scores tie so guidance includes the
            # most context available.
            candidates.append((score, len(cleaned), cleaned))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            best = candidates[-1]
            return best[2]

        def _gather_values(keys: tuple[str, ...]) -> list[str]:
            values: list[str] = []
            for field in keys:
                value = canonical.get(field)
                if not value:
                    continue
                cleaned = _clean_worker_metadata_value(value)
                if cleaned:
                    values.append(cleaned)
            return values

        status_fields = (
            "status",
            "status_text",
            "statusmessage",
            "status_message",
            "statusshortmessage",
            "short_message",
            "shortmessage",
        )
        detail_fields = (
            "status_detail_text",
            "statusdetailtext",
            "detail",
            "description",
            "summary",
            "status_long_text",
            "statuslongtext",
            "status_long_message",
            "statuslongmessage",
            "long_message",
            "longmessage",
            "long_text",
            "longtext",
        )

        status_values = _gather_values(status_fields)
        detail_values = _gather_values(detail_fields)

        composite_candidates: list[str] = []
        for status_value in status_values:
            composite_candidates.append(status_value)
            for detail_value in detail_values:
                combined = f"{status_value} {detail_value}".strip()
                if combined:
                    composite_candidates.append(combined)

        if not composite_candidates:
            composite_candidates = detail_values

        for candidate_text in composite_candidates:
            normalized = _normalise_worker_stalled_phrase(candidate_text)
            if _contains_worker_stall_signal(normalized):
                return candidate_text

        for fallback in ("last_error", "last_error_message", "error"):
            value = canonical.get(fallback)
            if not value:
                continue
            cleaned = _clean_worker_metadata_value(value)
            if cleaned:
                return cleaned

        return None

    message = _select_preferred_warning_message()

    token_map: dict[str, str] = {}
    for canonical_key, raw_value in canonical.items():
        if canonical_key in _WARNING_STRUCTURED_MESSAGE_KEYS or canonical_key in _WARNING_STRUCTURED_CONTEXT_KEYS:
            continue
        formatted = _format_warning_metadata_token(canonical_key, raw_value)
        if not formatted:
            continue
        normalized_key, normalized_value = formatted
        if normalized_key not in token_map:
            token_map[normalized_key] = normalized_value

    parts: list[str] = []
    if context_hint:
        parts.append(f"context={context_hint}")
    if message:
        parts.append(message)

    if token_map:
        tokens = [_render_warning_token(key, value) for key, value in sorted(token_map.items())]
        parts.append(" ".join(tokens))

    rendered = " ".join(segment.strip() for segment in parts if segment and segment.strip())
    if not rendered:
        return None

    normalized = _normalise_worker_stalled_phrase(rendered)
    lowered = normalized.lower()
    if not any(pattern.search(lowered) for pattern, _code, _ in _WORKER_ERROR_NORMALISERS):
        keywords = {"worker stalled", "restart", "backoff", "errcode", "error"}
        if not any(keyword in lowered for keyword in keywords):
            return None

    return rendered


_WORKER_RESTART_KEYS = {
    "restart",
    "restarts",
    "restart_count",
    "restartcounts",
    "restartattempt",
    "restartattempts",
    "attempt",
    "attempts",
    "retry",
    "retries",
    "tries",
    "trycount",
    "retrycount",
    "retry_count",
    "next_retry_attempt",
    "bouncecount",
}

_WORKER_RESTART_PREFIXES = {
    "restart",
    "retry",
    "attempt",
    "try",
    "bounce",
}

_WORKER_ERROR_KEYS = {
    "error",
    "err",
    "last_error",
    "lasterror",
    "error_message",
    "failure",
    "failreason",
    "reason",
}

_WORKER_ERROR_PREFIXES = {
    "error",
    "fail",
    "reason",
    "last_error",
}

_WORKER_BACKOFF_KEYS = {
    "backoff",
    "delay",
    "wait",
    "cooldown",
    "interval",
    "duration",
    "next_retry",
    "nextretry",
    "next_retry_in",
    "nextretryin",
    "retry_after",
    "retryafter",
    "retry_delay",
    "retrydelay",
    "next_restart",
    "nextrestart",
    "restart_delay",
    "restartdelay",
    "nextstart",
    "next_start",
}

_WORKER_BACKOFF_PREFIXES = {
    "backoff",
    "delay",
    "wait",
    "cooldown",
    "interval",
    "duration",
    "next_retry",
    "retry_after",
    "retry_delay",
    "next_restart",
    "restart_delay",
    "nextstart",
    "next_start",
}

_WORKER_LAST_SEEN_KEYS = {
    "since",
    "last_restart",
    "lastrestart",
    "last_start",
    "laststart",
    "last_seen",
    "lastseen",
    "last_success",
    "lastsuccess",
}

_WORKER_LAST_SEEN_PREFIXES = {
    "last",
    "since",
    "previous",
}

_WORKER_LAST_HEALTHY_KEYS = {
    "last_healthy",
    "lasthealthy",
    "last_healthy_at",
    "lasthealthyat",
    "last_health",
    "lasthealth",
    "last_healthy_timestamp",
    "lasthealthytimestamp",
    "last_healthy_time",
    "lasthealthytime",
    "last_healthy_check",
    "lasthealthcheck",
}

_WORKER_LAST_HEALTHY_PREFIXES = {
    "last_healthy",
    "lasthealthy",
    "last_health",
    "lasthealth",
    "last_good",
    "lastgood",
}


def _tokenize_metadata_key(key: str) -> tuple[str, ...]:
    """Return normalized token segments extracted from a metadata key."""

    normalized = key.strip("_")
    if not normalized:
        return ()

    segments: list[str] = []
    seen: set[str] = set()

    for candidate in (normalized, *normalized.split("_")):
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        segments.append(candidate)

    return tuple(segments)


def _classify_worker_metadata_key(key: str) -> str | None:
    """Return a semantic category for a worker telemetry key."""

    if not key:
        return None

    tokens = _tokenize_metadata_key(key)
    if not tokens:
        return None

    def _matches(
        category_keys: set[str],
        category_prefixes: set[str],
        *,
        allow_substring: bool,
    ) -> bool:
        if any(token in category_keys for token in tokens):
            return True
        for prefix in category_prefixes:
            if any(token.startswith(prefix) for token in tokens):
                return True
            if allow_substring and prefix in tokens[0]:
                return True
        return False

    last_healthy_match = _matches(
        _WORKER_LAST_HEALTHY_KEYS,
        _WORKER_LAST_HEALTHY_PREFIXES,
        allow_substring=True,
    )
    if last_healthy_match:
        return "last_healthy"

    last_seen_match = _matches(
        _WORKER_LAST_SEEN_KEYS, _WORKER_LAST_SEEN_PREFIXES, allow_substring=False
    )
    if last_seen_match and not any(
        "healthy" in token
        or token in {"error", "err", "failure", "reason"}
        or "error" in token
        or "fail" in token
        for token in tokens
    ):
        return "last_seen"

    if _matches(
        _WORKER_BACKOFF_KEYS, _WORKER_BACKOFF_PREFIXES, allow_substring=True
    ):
        return "backoff"

    if _matches(
        _WORKER_ERROR_KEYS, _WORKER_ERROR_PREFIXES, allow_substring=True
    ):
        return "error"

    # Docker Desktop on Windows frequently emits ``errCode`` or
    # ``lastErrorCode`` metadata alongside worker stall notifications.  These
    # identifiers do not contain underscores and therefore bypass the prefix
    # based detection above even though they are semantically error related.
    # Recognising them ensures that remediation guidance driven by
    # ``_WORKER_ERROR_CODE_GUIDANCE`` is applied to Windows specific error
    # codes such as ``WSL_KERNEL_OUTDATED``.
    for token in tokens:
        if re.search(r"(?:err|error|fail|reason)[a-z0-9]*code$", token):
            return "error"

    if _matches(
        _WORKER_RESTART_KEYS, _WORKER_RESTART_PREFIXES, allow_substring=True
    ):
        return "restart"

    return None


def _normalize_worker_context_candidate(candidate: str | None) -> str | None:
    """Return a cleaned worker context string if *candidate* is meaningful."""

    if not candidate:
        return None

    cleaned = candidate.strip().strip("\"'()[]{}<>")
    cleaned = re.sub(r"^(?:warn(?:ing)?|err(?:or)?|info|debug)\s*(?:[:\-]|::)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(".,;:")
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"worker", "workers", "after", "restart", "restarting", "in"}:
        return None
    if lowered.startswith("after") or lowered.startswith("workerafter"):
        return None
    if not any(char.isalpha() for char in cleaned):
        return None

    normalized = re.sub(r"\s+", " ", cleaned).strip()
    if not normalized:
        return None

    tokens = [token for token in re.split(r"\s+", normalized) if token]
    if not tokens:
        return None

    filtered_tokens = [
        token
        for token in tokens
        if token.strip().lower() not in _WORKER_INLINE_CONTEXT_STOPWORDS
    ]
    if filtered_tokens:
        normalized = " ".join(filtered_tokens)
        tokens = filtered_tokens

    if re.match(r"^[x×]\s*\d+", normalized):
        return None
    lowered_normalized = normalized.lower()
    if lowered_normalized.startswith("x") and any(char.isdigit() for char in normalized):
        return None
    if lowered_normalized.startswith("in ") and any(char.isdigit() for char in normalized):
        return None
    separators = {"=", ":"}
    if "code" in lowered_normalized and any(sep in normalized for sep in separators):
        return None
    if lowered_normalized.startswith("errcode"):
        return None
    if " because " in lowered_normalized:
        return None
    if not any(char.isalpha() for char in normalized):
        return None

    if all(token.lower() in _WORKER_INLINE_CONTEXT_STOPWORDS for token in tokens):
        return None

    return normalized or None


def _is_worker_context_noise(candidate: str) -> bool:
    """Return ``True`` when *candidate* resembles telemetry rather than context."""

    lowered = candidate.strip().lower()
    if not lowered:
        return True

    trimmed_lowered = lowered.strip("[]{}()<>:;,.\"'")

    if _WORKER_CONTEXT_SEVERITY_PATTERN.match(trimmed_lowered):
        return True

    severity_signature = re.sub(r"[^a-z]+", "", trimmed_lowered)
    if severity_signature and severity_signature in _WORKER_CONTEXT_SEVERITY_TOKENS:
        return True

    if _WORKER_CONTEXT_DURATION_PATTERN.fullmatch(lowered):
        return True

    normalized = re.sub(r"[\s\-_/]+", " ", lowered).strip()
    if not normalized:
        return True

    if "=" in normalized:
        key_part, _, value_part = normalized.partition("=")
        canonical_key = re.sub(r"[^a-z0-9]+", "", key_part)
        if not canonical_key:
            return True
        if canonical_key in _WORKER_CONTEXT_LOG_METADATA_KEYS:
            return True
        if canonical_key not in _WORKER_CONTEXT_BASE_KEY_TOKENS:
            return True
        candidate_value = value_part.strip().strip("\"'")
        if candidate_value:
            value_signature = re.sub(r"[^a-z]+", "", candidate_value.lower())
            if value_signature in _WORKER_CONTEXT_SEVERITY_TOKENS:
                return True

    colon_index = normalized.find(":")
    if colon_index != -1:
        key_part = normalized[:colon_index].strip()
        value_part = normalized[colon_index + 1 :].strip()
        if value_part:
            canonical_key = re.sub(r"[^a-z0-9]+", "", key_part)
            if canonical_key in _WORKER_CONTEXT_LOG_METADATA_KEYS:
                return True
            if canonical_key and canonical_key not in _WORKER_CONTEXT_BASE_KEY_TOKENS:
                return True
            value_signature = re.sub(r"[^a-z]+", "", value_part.lower())
            if value_signature in _WORKER_CONTEXT_SEVERITY_TOKENS:
                return True

    tokens = [token for token in normalized.split(" ") if token]
    if not tokens:
        return True

    if all(token in _WORKER_CONTEXT_NOISE_TOKENS for token in tokens):
        return True

    if tokens[0] in _WORKER_CONTEXT_CAUSAL_LEADING:
        return True

    if tokens[0] in _WORKER_CONTEXT_NOISE_LEADING and len(tokens) > 1:
        remaining = [token for token in tokens[1:] if token not in _WORKER_INLINE_CONTEXT_STOPWORDS]
        if remaining and all(token in _WORKER_CONTEXT_NOISE_TOKENS for token in remaining):
            return True

    return False


def _extract_worker_context(message: str, cleaned_message: str) -> str | None:
    """Extract the most meaningful worker context descriptor from *message*."""

    candidates: list[tuple[str, int]] = []
    candidate_positions: dict[str, int] = {}

    def _record_candidate(text: str, weight: int) -> None:
        key = text.lower()
        existing_index = candidate_positions.get(key)
        if existing_index is None:
            candidate_positions[key] = len(candidates)
            candidates.append((text, weight))
            return
        existing_text, existing_weight = candidates[existing_index]
        if weight > existing_weight or (
            weight == existing_weight and len(text) > len(existing_text)
        ):
            candidates[existing_index] = (text, weight)

    normalized_message = _normalise_worker_stalled_phrase(message)
    normalized_cleaned = _normalise_worker_stalled_phrase(cleaned_message)

    cleaned_candidates: list[str] = []
    for candidate in (cleaned_message, normalized_cleaned):
        if candidate and candidate not in cleaned_candidates:
            cleaned_candidates.append(candidate)

    for candidate_text in cleaned_candidates:
        context_match = re.search(
            r"""
            worker\s+stalled
            (?:(?:\s*(?:[;:,.\-–—…]|->|=>|→|⇒)\s*)*)
            restart(?:ing)?
            (?:\s+(?:in|after)\s+[^()]+)?
            (?:\s*(?:[:\-–—]\s*|\(\s*)(?P<context>[^)]+?)(?:\s*\)|$))?
            """,
            candidate_text,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        if context_match:
            candidate = _normalize_worker_context_candidate(context_match.group("context"))
            if candidate:
                _record_candidate(candidate, 90)
            break

    message_sources: list[str] = []
    for source in (message, normalized_message):
        if source and source not in message_sources:
            message_sources.append(source)

    for pattern in (
        _WORKER_CONTEXT_PREFIX_PATTERN,
        _WORKER_CONTEXT_KV_PATTERN,
        _WORKER_CONTEXT_RESTART_PATTERN,
        _WORKER_CONTEXT_STALLED_PATTERN,
        _WORKER_CONTEXT_LEADING_PATTERN,
        _WORKER_CONTEXT_BRACKET_PATTERN,
    ):
        for source in message_sources:
            for match in pattern.finditer(source):
                if "value" in match.groupdict():
                    raw_candidate = match.group("value")
                else:
                    raw_candidate = match.group("context")
                normalized = _normalize_worker_context_candidate(raw_candidate)
                if not normalized:
                    continue
                weight = 20
                key = match.groupdict().get("key", "")
                key_normalized = key.lower() if key else ""
                base_key = key_normalized
                if key_normalized:
                    if any(sep in key_normalized for sep in {".", "-", "_"}):
                        base_key = re.split(r"[._-]", key_normalized, 1)[0]
                    else:
                        for candidate in _WORKER_CONTEXT_BASE_KEYS:
                            if key_normalized.startswith(candidate):
                                base_key = candidate
                                break
                if base_key in {"worker", "id", "name"}:
                    weight = 80
                elif base_key in {"context", "component"}:
                    weight = 60
                elif base_key in {"module"}:
                    weight = 50
                elif base_key in {"subsystem"}:
                    weight = 58
                elif base_key in {"scope"}:
                    weight = 52
                elif base_key in {
                    "service",
                    "pipeline",
                    "task",
                    "unit",
                    "process",
                    "channel",
                    "queue",
                    "thread",
                    "engine",
                    "backend",
                    "runner",
                    "target",
                    "namespace",
                    "project",
                    "group",
                    "agent",
                    "executor",
                    "handler",
                }:
                    weight = 55
                elif pattern in {_WORKER_CONTEXT_RESTART_PATTERN, _WORKER_CONTEXT_PREFIX_PATTERN}:
                    weight = 70
                elif pattern is _WORKER_CONTEXT_BRACKET_PATTERN:
                    weight = 65
                _record_candidate(normalized, weight)
            # allow normalized source to contribute when original fails while avoiding duplicate matches

    if not candidates:
        return None

    best_candidate, _ = max(
        candidates,
        key=lambda item: (item[1], len(item[0])),
    )

    if best_candidate.lower().startswith("worker "):
        for option, _ in sorted(candidates, key=lambda item: (-item[1], -len(item[0]))):
            if not option.lower().startswith("worker "):
                return option
        return None

    if _is_worker_context_noise(best_candidate):
        for option, _ in sorted(candidates, key=lambda item: (-item[1], -len(item[0]))):
            if option == best_candidate:
                continue
            if option.lower().startswith("worker "):
                continue
            if _is_worker_context_noise(option):
                continue
            return option
        return None

    return best_candidate


def _strip_enclosing_pairs(value: str, pairs: SequenceABC[tuple[str, str]]) -> str:
    """Remove repeated wrapping pairs such as quotes or parentheses."""

    candidate = value
    changed = True
    while changed and candidate:
        changed = False
        for prefix, suffix in pairs:
            if candidate.startswith(prefix) and candidate.endswith(suffix):
                candidate = candidate[len(prefix) : -len(suffix)].strip()
                changed = True
                break
    return candidate


_WRAPPER_BALANCED_PAIRS: tuple[tuple[str, str], ...] = (
    ("\"", "\""),
    ("'", "'"),
    ("`", "`"),
)

_WRAPPER_STRUCTURAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
)

_TRAILING_CLOSERS = {
    ")": "(",
    "]": "[",
    "}": "{",
    ">": "<",
}

_LEADING_OPENERS = {value: key for key, value in _TRAILING_CLOSERS.items()}


def _clean_worker_metadata_value(raw_value: str) -> str:
    """Return a sanitised token extracted from worker diagnostic payloads."""

    cleaned = raw_value.strip()
    if not cleaned:
        return ""

    cleaned = _strip_enclosing_pairs(cleaned, _WRAPPER_BALANCED_PAIRS)
    cleaned = _strip_enclosing_pairs(cleaned, _WRAPPER_STRUCTURAL_PAIRS)
    cleaned = _strip_enclosing_pairs(cleaned, _WRAPPER_BALANCED_PAIRS)

    while cleaned and cleaned[0] in _LEADING_OPENERS:
        opener = cleaned[0]
        closer = _LEADING_OPENERS[opener]
        if cleaned.count(opener) > cleaned.count(closer):
            cleaned = cleaned[1:].lstrip()
            continue
        break

    while cleaned and cleaned[-1] in _TRAILING_CLOSERS:
        closer = cleaned[-1]
        opener = _TRAILING_CLOSERS[closer]
        if cleaned.count(opener) < cleaned.count(closer):
            cleaned = cleaned[:-1].rstrip()
            continue
        break

    cleaned = cleaned.strip(" \t\r\n;,:")
    return cleaned


def _normalise_worker_original_token(
    value: str,
) -> tuple[str, str | None]:
    """Return a sanitised original token and the preserved banner (if different)."""

    token = value.strip()
    if not token:
        return "", None

    # Strip Docker's warning prefix and redundant punctuation so that the raw
    # metadata reflects the actionable portion of the banner.
    token = _DOCKER_WARNING_PREFIX_PATTERN.sub("", token)
    token = re.sub(r"\s+", " ", token).strip(" .;:-")

    if not token:
        return "", None

    normalized = _normalise_worker_stalled_phrase(token)
    lowered = normalized.lower()

    for pattern, _code, narrative in _WORKER_ERROR_NORMALISERS:
        if pattern.search(lowered):
            # Preserve the exact banner so that advanced diagnostics can still
            # surface the original payload when needed.
            banner = token if token and token != narrative else None
            return narrative, banner

    if normalized != token:
        return normalized, token

    return token, None



_WORKER_ERRCODE_HINT_PATTERN = re.compile(
    r'\b(?:err(?:or)?code|code)\b\s*(?:[:=]\s*|\s+)(?:"(?P<double>[^"]+)"|\'(?P<single>[^\']+)\'|(?P<bare>[A-Za-z0-9_./-]+))',
    re.IGNORECASE,
)


def _extract_worker_error_code_hint(text: str | None) -> str | None:
    """Return a canonical worker error code extracted from free-form text."""

    if not text:
        return None

    for match in _WORKER_ERRCODE_HINT_PATTERN.finditer(text):
        candidate = match.group('double') or match.group('single') or match.group('bare')
        if not candidate:
            continue
        cleaned = _clean_worker_metadata_value(str(candidate))
        if not cleaned:
            continue
        normalized = re.sub(r"[^A-Za-z0-9_./-]+", "", cleaned).strip()
        if not normalized:
            continue
        return normalized.upper()

    return None


def _infer_worker_error_code_from_context(*values: str | None) -> str | None:
    """Infer a worker error code from contextual hints when ``errCode`` is absent."""

    tokens: list[str] = []
    for value in values:
        if not value:
            continue
        cleaned = _clean_worker_metadata_value(value)
        if not cleaned:
            continue
        tokens.append(cleaned)

    if not tokens:
        return None

    corpus = " ".join(tokens).casefold()

    virtualization_markers = ("wsl", "virtual machine", "vm", "docker-desktop")
    suspension_markers = ("suspend", "suspended", "sleep")
    hibernation_markers = (
        "hibernat",
        "fast startup",
        "fast-startup",
        "faststart",
        "hybrid sleep",
        "resume from sleep",
        "resumed from sleep",
        "resumed from hibernation",
        "hibernated",
        "hibernation",
    )
    pause_markers = ("pause", "paused", "standby")

    virtualization_context = any(marker in corpus for marker in virtualization_markers)
    suspension_context = any(marker in corpus for marker in suspension_markers)
    pause_context = any(marker in corpus for marker in pause_markers)
    hibernation_context = any(marker in corpus for marker in hibernation_markers)

    if virtualization_context or hibernation_context or suspension_context or pause_context:
        if suspension_context:
            return "WSL_VM_SUSPENDED"
        if pause_context:
            return "WSL_VM_PAUSED"
        if hibernation_context:
            return "WSL_VM_HIBERNATED"

    hyperv_markers = (
        "hyper-v",
        "hyperv",
        "hypervisor",
        "host compute service",
        "hcs",
        "vm compute service",
    )
    if any(marker in corpus for marker in hyperv_markers):
        missing_markers = (
            "not present",
            "missing",
            "not installed",
            "absent",
            "install hyper-v",
        )
        offline_markers = (
            "not running",
            "stopped",
            "stop",  # covers "cannot stop" / "stopped"
            "disabled",
            "turned off",
            "hypervisorlaunchtype off",
            "failed to start",
            "cannot start",
        )

        if any(marker in corpus for marker in missing_markers):
            return "HCS_E_HYPERV_NOT_PRESENT"
        if any(marker in corpus for marker in offline_markers):
            return "HCS_E_HYPERV_NOT_RUNNING"

    if "vsock" in corpus or "hyper-v socket" in corpus or "hvsock" in corpus:
        timeout_markers = (
            "timeout",
            "timed out",
            "deadline",
            "hang",
            "stuck",
        )
        if any(marker in corpus for marker in timeout_markers):
            return "VPNKIT_VSOCK_TIMEOUT"
        degraded_markers = ("refused", "unresponsive", "reset", "closed", "broken")
        if any(marker in corpus for marker in degraded_markers):
            return "VPNKIT_VSOCK_UNRESPONSIVE"
        return "VPNKIT_VSOCK_UNRESPONSIVE"

    return None


def _synthesise_worker_stall_error_detail(
    *,
    message: str | None,
    canonical_error: str,
    original_token: str,
    metadata: dict[str, str],
    codes: Sequence[str],
) -> tuple[str, str]:
    """Return a narrative and detail summary for worker stall diagnostics."""

    narrative = _WORKER_STALLED_PRIMARY_NARRATIVE
    metadata["docker_worker_last_error"] = narrative
    metadata["docker_worker_last_error_original"] = narrative
    metadata["docker_worker_last_error_raw"] = narrative
    metadata["docker_worker_last_error_banner"] = narrative
    metadata["docker_worker_last_error_samples"] = narrative
    metadata["docker_worker_last_error_original_samples"] = narrative
    metadata["docker_worker_last_error_raw_samples"] = narrative
    metadata["docker_worker_last_error_banner_samples"] = narrative
    metadata.setdefault("docker_worker_last_error_category", "worker_stalled")
    metadata.setdefault("docker_worker_last_error_interpreted", "worker_stalled")
    metadata.setdefault("docker_worker_last_error_narrative", narrative)

    if not metadata.get("docker_worker_last_error_banner_raw"):
        metadata["docker_worker_last_error_banner_raw"] = narrative
    metadata["docker_worker_last_error_banner_raw_samples"] = narrative

    signature_source = original_token or message or canonical_error
    signature = _fingerprint_worker_banner(signature_source)
    if signature:
        metadata.setdefault("docker_worker_last_error_banner_signature", signature)

    preserved_banner = metadata.get("docker_worker_last_error_banner")
    if preserved_banner and preserved_banner.strip():
        metadata.setdefault("docker_worker_last_error_banner_preserved", preserved_banner)
        metadata.setdefault(
            "docker_worker_last_error_banner_preserved_samples",
            preserved_banner,
        )

    unique_codes: list[str] = []
    seen: set[str] = set()
    for code in codes:
        if not code:
            continue
        normalized = code.strip()
        if not normalized:
            continue
        canonical = normalized.upper()
        if canonical in seen:
            continue
        seen.add(canonical)
        unique_codes.append(normalized)

    if not unique_codes:
        hint = _extract_worker_error_code_hint(original_token or message)
        if hint:
            unique_codes.append(hint)

    if unique_codes:
        metadata.setdefault("docker_worker_last_error_code", unique_codes[0])
        if len(unique_codes) > 1:
            metadata.setdefault("docker_worker_last_error_codes", ", ".join(unique_codes))

        primary_code = metadata.get("docker_worker_last_error_code", "")
        if primary_code.upper() == _WORKER_STALLED_PRIMARY_CODE.upper():
            inferred = _infer_worker_error_code_from_context(
                message,
                canonical_error,
                original_token,
                metadata.get("docker_worker_last_error_banner_preserved"),
            )
            if inferred and inferred.upper() != primary_code.upper():
                metadata["docker_worker_last_error_code"] = inferred
                unique_codes = [inferred]
    else:
        inferred = _infer_worker_error_code_from_context(
            message,
            canonical_error,
            original_token,
            metadata.get("docker_worker_last_error_banner_preserved"),
        )
        if inferred:
            metadata.setdefault("docker_worker_last_error_code", inferred)
            unique_codes = [inferred]
        else:
            metadata.setdefault("docker_worker_last_error_code", _WORKER_STALLED_PRIMARY_CODE)

    sentences: list[str] = []

    def _append(sentence: str) -> None:
        normalized_sentence = sentence.strip()
        if not normalized_sentence:
            return
        if not normalized_sentence.endswith("."):
            normalized_sentence += "."
        sentences.append(normalized_sentence)

    _append(narrative)

    context = metadata.get("docker_worker_context")
    context_source = original_token or canonical_error or (message or "")
    if not context and context_source:
        normalized_source = _normalise_worker_stalled_phrase(context_source)
        cleaned_source = re.sub(r"\s+", " ", normalized_source).strip()
        candidate = _extract_worker_context(context_source, cleaned_source)
        if candidate:
            metadata.setdefault("docker_worker_context", candidate)
            context = candidate

    if context:
        _append(f"Affected component: {context}")

    code_value = metadata.get("docker_worker_last_error_code")
    if code_value:
        label = _WORKER_ERROR_CODE_LABELS.get(code_value.upper())
        if label:
            _append(f"Diagnostic code {code_value} indicates {label}")
        else:
            _append(f"Docker reported diagnostic code {code_value}")

    restart_count = metadata.get("docker_worker_restart_count")
    if restart_count:
        try:
            count = int(restart_count)
        except (TypeError, ValueError):
            count = None
        if count is not None and count >= 0:
            plural = "restart" if count == 1 else "restarts"
            _append(f"The worker restarted {count} time{'s' if count != 1 else ''} during diagnostics")
        else:
            _append(f"Docker reported {restart_count} worker restarts during diagnostics")

    backoff = metadata.get("docker_worker_backoff")
    if backoff:
        _append(f"Docker is applying a restart backoff of {backoff}")

    detail = " ".join(sentences)
    if detail:
        metadata.setdefault("docker_worker_last_error_summary", detail)

    return narrative, detail or narrative


def _normalise_worker_error_message(
    raw_value: str,
    *,
    raw_original: str | None = None,
    existing_code: str | None = None,
) -> tuple[str | None, str | None, dict[str, str]]:
    """Return a user-facing worker error description and supplemental metadata."""

    cleaned = _clean_worker_metadata_value(raw_value)
    if not cleaned:
        return None, None, {}

    collapsed = re.sub(r"\s+", " ", cleaned).strip(" .;:,-")
    if not collapsed:
        return None, None, {}

    lowered = collapsed.lower()
    original_token = (raw_original or raw_value).strip()
    normalized_original, preserved_banner = _normalise_worker_original_token(
        original_token
    )

    canonical_error = normalized_original or collapsed
    metadata: dict[str, str] = {
        "docker_worker_last_error_original": collapsed,
        "docker_worker_last_error_raw": canonical_error,
        "docker_worker_last_error_banner": canonical_error,
        "docker_worker_last_error_banner_raw": canonical_error,
    }
    preserved_signature = _fingerprint_worker_banner(
        preserved_banner or original_token or raw_value
    )
    if preserved_signature:
        metadata["docker_worker_last_error_banner_signature"] = preserved_signature

    preserved_banner_text = preserved_banner or metadata.get(
        "docker_worker_last_error_banner"
    )
    sanitized_preserved_banner = (
        _sanitize_worker_banner_text(preserved_banner_text)
        if preserved_banner_text
        else None
    )

    if sanitized_preserved_banner:
        metadata["docker_worker_last_error_banner_preserved"] = (
            sanitized_preserved_banner
        )
        metadata["docker_worker_last_error_banner_preserved_samples"] = (
            sanitized_preserved_banner
        )

    if (
        preserved_banner_text
        and sanitized_preserved_banner
        and preserved_banner_text != sanitized_preserved_banner
    ):
        metadata["docker_worker_last_error_banner_preserved_raw"] = (
            preserved_banner_text
        )
        metadata["docker_worker_last_error_banner_preserved_raw_samples"] = (
            preserved_banner_text
        )

    codes: list[str] = []

    def _unique_codes(values: Iterable[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value:
                continue
            normalized = value.strip()
            if not normalized:
                continue
            canonical = normalized.upper()
            if canonical in seen:
                continue
            seen.add(canonical)
            unique.append(normalized)
        return unique

    if existing_code:
        normalized_existing = existing_code.strip()
        if normalized_existing:
            metadata["docker_worker_last_error_code"] = normalized_existing
            codes.append(normalized_existing)

    for pattern, code, narrative in _WORKER_ERROR_NORMALISERS:
        if pattern.search(lowered):
            metadata["docker_worker_last_error_original"] = narrative
            if not codes:
                metadata["docker_worker_last_error_code"] = code
                codes.append(code)
            elif code not in codes:
                codes.append(code)
                metadata.setdefault(
                    "docker_worker_last_error_code_inferred", code
                )
            else:
                metadata.setdefault(
                    "docker_worker_last_error_code_inferred", code
                )
            detail = f"{narrative}."
            unique_codes = _unique_codes(codes)
            if len(unique_codes) > 1:
                metadata["docker_worker_last_error_codes"] = ", ".join(unique_codes)
            if code.upper() == _WORKER_STALLED_PRIMARY_CODE.upper():
                inferred = _infer_worker_error_code_from_context(
                    raw_value,
                    raw_original,
                    metadata.get("docker_worker_last_error_banner_raw"),
                    metadata.get("docker_worker_last_error_banner_preserved"),
                    metadata.get("docker_worker_last_error_raw"),
                )
                if inferred and inferred.upper() != code.upper():
                    metadata["docker_worker_last_error_code"] = inferred
                    metadata.setdefault(
                        "docker_worker_last_error_code_inferred", inferred
                    )
                    inferred_codes = _unique_codes([*codes, inferred])
                    metadata["docker_worker_last_error_codes"] = ", ".join(
                        inferred_codes
                    )
            return narrative, detail, metadata

    
    codes = _unique_codes(codes)
    if codes:
        metadata.setdefault("docker_worker_last_error_code", codes[0])
        if len(codes) > 1:
            metadata.setdefault("docker_worker_last_error_codes", ", ".join(codes))

    detection_sources = [collapsed, canonical_error, original_token]
    has_worker_signal = any(
        _contains_worker_stall_signal(source) for source in detection_sources if source
    )
    if not has_worker_signal and raw_original:
        has_worker_signal = _contains_worker_stall_signal(raw_original)

    if has_worker_signal:
        narrative, detail = _synthesise_worker_stall_error_detail(
            message=raw_original or raw_value,
            canonical_error=canonical_error,
            original_token=original_token,
            metadata=metadata,
            codes=codes,
        )
        return narrative, detail, metadata

    inferred = _infer_worker_error_code_from_context(
        raw_value,
        raw_original,
        canonical_error,
        original_token,
        metadata.get("docker_worker_last_error_banner_preserved"),
    )
    if inferred:
        metadata.setdefault("docker_worker_last_error_code", inferred)
        existing_multi = metadata.get("docker_worker_last_error_codes")
        if existing_multi:
            tokens = _split_metadata_values(existing_multi)
            if inferred not in tokens:
                tokens.append(inferred)
                metadata["docker_worker_last_error_codes"] = ", ".join(tokens)
        else:
            metadata.setdefault("docker_worker_last_error_codes", inferred)
        metadata.setdefault("docker_worker_last_error_code_inferred", inferred)

    fallback_detail = (
        "Docker Desktop reported the worker error '%s'." % collapsed
    )

    fallback_sources = (
        raw_value,
        canonical_error,
        normalized_original,
        collapsed,
    )
    for source in fallback_sources:
        if not source:
            continue
        harmonised = _normalise_worker_stalled_phrase(str(source))
        if "worker stalled" in harmonised.casefold():
            narrative, detail = _synthesise_worker_stall_error_detail(
                message=raw_value,
                canonical_error=canonical_error,
                original_token=original_token,
                metadata=metadata,
                codes=codes,
            )
            metadata.setdefault("docker_worker_last_error", narrative)
            return narrative, detail, metadata

    return collapsed, fallback_detail, metadata



def _normalise_worker_metadata_key(raw_key: str) -> str:
    """Return a canonical lowercase key for worker diagnostic attributes."""

    normalised = raw_key.strip().lower()
    if not normalised:
        return normalised
    return re.sub(r"[^a-z0-9]+", "_", normalised)


def _strip_interval_clause_suffix(raw: str) -> str:
    """Trim descriptive tails from interval clauses such as ``"30s due to"``."""

    trimmed = raw.strip()
    if not trimmed:
        return ""

    trimmed = re.split(
        r"\b(?:due|because|caused|cause|owing|reason|while|when|with|after|before|as|pending)\b",
        trimmed,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    if "=" in trimmed:
        key, value = trimmed.split("=", 1)
        key_normalized = key.strip().lower()
        if key_normalized in {
            "backoff",
            "delay",
            "wait",
            "cooldown",
            "interval",
            "duration",
            "next",
            "nextretry",
            "next_retry",
            "nextretryin",
            "next_retry_in",
            "nextrestart",
            "next_restart",
            "nextrestartin",
            "next_restart_in",
            "nextstart",
            "next_start",
            "nextstartin",
            "next_start_in",
            "retry_after",
            "retryafter",
            "retry_delay",
            "retrydelay",
            "restart_delay",
            "restartdelay",
        }:
            trimmed = value

    return trimmed.strip(" \t\n\r,.);:")


def _normalise_approx_prefix(raw: str | None) -> str | None:
    """Return a user-friendly approximation qualifier if *raw* is meaningful."""

    if not raw:
        return None
    lowered = raw.strip().lower()
    if not lowered:
        return None
    if lowered in {"~", "≈"}:
        return "~"
    if lowered.startswith("approx"):
        return "approximately"
    if lowered in {"about", "around", "roughly"}:
        return "about"
    if lowered in {"near", "nearly"}:
        return "nearly"
    return lowered


def _format_go_duration(token: str) -> str | None:
    """Return a human readable representation of Go-style duration *token*."""

    sanitized = token.replace(" ", "")
    if not sanitized:
        return None

    components = list(_GO_DURATION_COMPONENT_PATTERN.finditer(sanitized))
    if not components:
        return None

    reconstructed = "".join(match.group(0) for match in components)
    if reconstructed.lower() != sanitized.lower():
        return None

    normalized_segments: list[str] = []
    for match in components:
        value = match.group("value").lstrip("0") or "0"
        unit = match.group("unit").lower()
        normalized_segments.append(f"{value}{unit}")
    return " ".join(normalized_segments) if normalized_segments else None


def _interpret_clock_duration(value: str) -> tuple[str, float] | None:
    """Return a normalized clock-style duration and its total seconds."""

    candidate = value.strip()
    if not candidate or not _CLOCK_DURATION_PATTERN.match(candidate):
        return None

    parts = candidate.split(":")
    layout = _CLOCK_DURATION_LAYOUTS.get(len(parts))
    if not layout:
        return None

    numeric_parts: list[float] = []
    for token, unit in zip(parts, layout):
        if unit == "seconds":
            try:
                numeric = float(token)
            except ValueError:
                return None
        else:
            try:
                numeric = int(token)
            except ValueError:
                return None
        numeric_parts.append(numeric)

    total_seconds = sum(
        numeric * _CLOCK_DURATION_FACTORS[unit]
        for numeric, unit in zip(numeric_parts, layout)
    )

    segments: list[str] = []
    for numeric, unit in zip(numeric_parts, layout):
        symbol = _CLOCK_DURATION_SYMBOLS[unit]
        if unit == "seconds":
            if numeric == 0 and segments:
                continue
            if abs(numeric - round(numeric)) < 1e-9:
                rendered = str(int(round(numeric)))
            else:
                rendered = ("%g" % numeric).rstrip("0").rstrip(".")
            segments.append(f"{rendered}{symbol}")
        else:
            if numeric == 0:
                continue
            segments.append(f"{int(numeric)}{symbol}")

    if not segments:
        segments.append("0s")

    return " ".join(segments), total_seconds


def _clean_iso_numeric(token: str | None) -> float:
    if not token:
        return 0.0
    normalized = token.replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return 0.0


def _parse_iso8601_duration_seconds(value: str) -> float | None:
    """Return total seconds represented by an ISO-8601 duration token."""

    if not value:
        return None

    match = _ISO_DURATION_PATTERN.fullmatch(value.strip())
    if not match:
        return None

    sign = -1.0 if match.group("sign") == "-" else 1.0

    years = _clean_iso_numeric(match.group("years"))
    months = _clean_iso_numeric(match.group("months"))
    weeks = _clean_iso_numeric(match.group("weeks"))
    days = _clean_iso_numeric(match.group("days"))
    hours = _clean_iso_numeric(match.group("hours"))
    minutes = _clean_iso_numeric(match.group("minutes"))
    seconds = _clean_iso_numeric(match.group("seconds"))

    total_seconds = 0.0
    total_seconds += years * 365.0 * _CLOCK_DURATION_FACTORS["days"]
    total_seconds += months * 30.0 * _CLOCK_DURATION_FACTORS["days"]
    total_seconds += weeks * 7.0 * _CLOCK_DURATION_FACTORS["days"]
    total_seconds += days * _CLOCK_DURATION_FACTORS["days"]
    total_seconds += hours * _CLOCK_DURATION_FACTORS["hours"]
    total_seconds += minutes * _CLOCK_DURATION_FACTORS["minutes"]
    total_seconds += seconds * _CLOCK_DURATION_FACTORS["seconds"]

    return sign * total_seconds


def _format_seconds_duration(value: float) -> str:
    """Render ``value`` seconds as a compact ``1h 2m 3s`` style duration."""

    seconds = float(value)
    if not math.isfinite(seconds):
        return "0s"

    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)

    hours = int(seconds // _CLOCK_DURATION_FACTORS["hours"])
    seconds -= hours * _CLOCK_DURATION_FACTORS["hours"]
    minutes = int(seconds // _CLOCK_DURATION_FACTORS["minutes"])
    seconds -= minutes * _CLOCK_DURATION_FACTORS["minutes"]

    def _format_seconds_component(component: float) -> str:
        if abs(component - round(component)) < 1e-9:
            return f"{int(round(component))}s"
        text = ("%.6f" % component).rstrip("0").rstrip(".")
        return f"{text}s"

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(_format_seconds_component(seconds))

    rendered = " ".join(parts)
    return f"{sign}{rendered}" if sign else rendered


def _normalise_backoff_hint(value: str) -> str | None:
    """Return a normalised representation of a worker backoff interval."""

    if not value:
        return None

    candidate = value.strip().strip(";.,:)")
    candidate = candidate.strip("()[]{}")
    if not candidate:
        return None

    def _combine(prefix_value: str | None, token: str) -> str:
        if not prefix_value:
            return token
        if prefix_value == "~":
            return f"~{token}".strip()
        return f"{prefix_value} {token}".strip()

    prefix: str | None = None
    prefix_match = _APPROX_PREFIX_PATTERN.match(candidate)
    if prefix_match:
        prefix = _normalise_approx_prefix(prefix_match.group("prefix"))
        candidate = candidate[prefix_match.end() :].lstrip()

    iso_seconds = _parse_iso8601_duration_seconds(candidate)
    if iso_seconds is not None:
        normalized = _format_seconds_duration(abs(iso_seconds))
        if iso_seconds < 0:
            normalized = f"-{normalized}"
        return _combine(prefix, normalized)

    go_candidate = _format_go_duration(candidate)
    if go_candidate:
        normalized = go_candidate
    else:
        clock_candidate = _interpret_clock_duration(candidate)
        if clock_candidate:
            normalized, _ = clock_candidate
        else:
            match = _BACKOFF_INTERVAL_PATTERN.match(candidate)
            if not match:
                combined = _combine(prefix, candidate)
                return combined or None
            number = match.group("number")
            unit = match.group("unit")
            if not number:
                combined = _combine(prefix, candidate)
                return combined or None
            normalized = number
            if unit:
                normalized_unit = _DURATION_UNIT_NORMALISATION.get(unit.lower())
                if normalized_unit:
                    normalized = f"{number}{normalized_unit}"
                else:
                    normalized = f"{number}{unit.strip()}"
    if prefix:
        return _combine(prefix, normalized)
    return normalized.strip() if normalized else None


def _scan_backoff_hint_from_message(message: str) -> str | None:
    """Extract a plausible backoff interval embedded within *message*."""

    if not message:
        return None

    lowered = message.lower()
    if not any(token in lowered for token in {"restart", "retry", "backoff", "stalled"}):
        return None

    for match in _GO_DURATION_PATTERN.finditer(message):
        prefix_fragment = message[: match.start()]
        suffix_match = _APPROX_SUFFIX_PATTERN.search(prefix_fragment)
        if suffix_match:
            candidate_start = suffix_match.start()
        else:
            candidate_start = match.start()
        candidate = message[candidate_start:match.end()]
        normalized = _normalise_backoff_hint(candidate)
        if normalized and any(char.isalpha() for char in normalized):
            return normalized

    for match in _CLOCK_DURATION_SEARCH_PATTERN.finditer(message):
        prefix_fragment = message[: match.start()]
        suffix_match = _APPROX_SUFFIX_PATTERN.search(prefix_fragment)
        if suffix_match:
            candidate_start = suffix_match.start()
        else:
            candidate_start = match.start()
        candidate = message[candidate_start:match.end()]
        normalized = _normalise_backoff_hint(candidate)
        if normalized and any(char.isalpha() for char in normalized):
            return normalized

    for match in _BACKOFF_INTERVAL_PATTERN.finditer(message):
        candidate = match.group(0)
        normalized = _normalise_backoff_hint(candidate)
        if normalized and any(char.isalpha() for char in normalized):
            return normalized

    return None


def _extract_worker_flapping_descriptors(
    message: str, *, normalized_message: str | None = None
) -> tuple[list[str], dict[str, str]]:
    """Derive human friendly descriptors for flapping Docker workers."""

    context_descriptor: str | None = None
    context_details: list[str] = []
    metadata: dict[str, str] = {}

    restart_count: int | None = None
    last_error: str | None = None
    last_error_raw_value: str | None = None
    backoff_hint: str | None = None
    last_seen: str | None = None
    last_healthy: str | None = None

    normalized_source = normalized_message or _normalise_worker_stalled_phrase(message)

    envelope = _parse_docker_log_envelope(message)

    def _score_context_key(key: str) -> int:
        lowered = key.lower()
        base = 0
        if lowered in {"context", "worker", "component", "module", "namespace", "service", "scope", "subsystem", "target", "name"}:
            base = 90
        elif lowered.endswith("context"):
            base = 85
        elif "context" in lowered:
            base = 80
        elif lowered.endswith("component"):
            base = 70
        elif "component" in lowered:
            base = 60
        elif any(token in lowered for token in {"worker", "module", "service", "scope", "target", "namespace", "name"}):
            base = 55
        if not base:
            return 0
        if "diagnostic" in lowered or "telemetry" in lowered:
            base += 5
        base += lowered.count("_")
        return base

    context_scores: dict[str, int] = {}
    for key, value in envelope.items():
        score = _score_context_key(key)
        if not score:
            continue
        normalized = _clean_worker_metadata_value(value)
        if not normalized:
            continue

        lowered_value = normalized.lower()

        # ``errCode`` fields often accompany worker stall banners on Windows.
        # They describe remediation categories rather than the actual worker
        # name and therefore should never be treated as the affected worker
        # context.  Earlier heuristics occasionally elevated values such as
        # ``Code=WSL_KERNEL_OUTDATED`` to the primary context which produced
        # awkward guidance like "Affected component: Code=WSL_KERNEL_OUTDATED".
        # Filter these fragments so that subsequent diagnostics focus on the
        # real worker (for example ``vpnkit``) when available.
        if "code" in lowered_value and "=" in normalized:
            continue
        if lowered_value.startswith("errcode"):
            continue

        penalty = 0
        if lowered_value in {"docker-desktop", "desktop-linux"}:
            penalty -= 6
        elif lowered_value in {"desktop-windows", "desktop"}:
            penalty -= 3
        current = context_scores.get(normalized)
        candidate_score = score + penalty
        if current is None or candidate_score > current:
            context_scores[normalized] = candidate_score

    if context_scores:
        best_context = max(
            context_scores.items(), key=lambda item: (item[1], len(item[0]))
        )[0]
        metadata["docker_worker_context"] = best_context

    def _set_backoff_hint(candidate: str | None) -> None:
        nonlocal backoff_hint
        if backoff_hint is not None or not candidate:
            return
        normalized = _normalise_backoff_hint(candidate)
        if normalized:
            backoff_hint = normalized

    def _ingest_metadata_candidate(key: str | None, value: str | None) -> None:
        nonlocal restart_count, last_error, backoff_hint, last_seen, last_error_raw_value, last_healthy
        if not key or value is None:
            return
        normalized_key = _normalise_worker_metadata_key(key)
        cleaned_value = _clean_worker_metadata_value(value)
        if not normalized_key or not cleaned_value:
            return
        category = _classify_worker_metadata_key(normalized_key)
        if category == "restart" and restart_count is None:
            number_match = re.search(r"(-?\d+)", cleaned_value)
            if number_match:
                try:
                    restart_count = int(number_match.group(1))
                except ValueError:
                    restart_count = None
            return
        if category == "error":
            def _should_accept_error(existing: str | None) -> bool:
                if existing is None:
                    return True
                if existing in _WORKER_ERROR_NARRATIVES:
                    return True
                if _looks_like_truncated_worker_error_token(existing):
                    return True
                return False

            structured_payload = _maybe_parse_structured_value(value)
            if structured_payload is not None:
                structured_message, structured_metadata = _extract_structured_error_details(
                    structured_payload
                )
                _merge_structured_error_metadata(metadata, structured_metadata)
                if structured_message and _should_accept_error(last_error):
                    if isinstance(value, str):
                        last_error_raw_value = value.strip()
                    else:
                        serialized = _stringify_envelope_value(structured_payload)
                        if not serialized:
                            serialized = json.dumps(
                                structured_payload, ensure_ascii=False, sort_keys=True
                            )
                        last_error_raw_value = serialized
                    last_error = structured_message
                    return
            key_suffix = normalized_key.rsplit("_", 1)[-1]
            if key_suffix.endswith("code") or normalized_key.endswith("code"):
                metadata.setdefault("docker_worker_last_error_code", cleaned_value)
                return

            message_suffixes = {
                "error",
                "err",
                "message",
                "summary",
                "detail",
                "reason",
                "failure",
                "failreason",
                "cause",
                "description",
            }

            if any(key_suffix.endswith(token) for token in message_suffixes):
                if _should_accept_error(last_error):
                    last_error = cleaned_value
                    if isinstance(value, str):
                        last_error_raw_value = value.strip()
                    else:
                        last_error_raw_value = str(value).strip()
                return

            if _should_accept_error(last_error):
                last_error = cleaned_value
                if isinstance(value, str):
                    last_error_raw_value = value.strip()
                else:
                    last_error_raw_value = str(value).strip()
            return
        if category == "backoff":
            numeric_hint = _derive_numeric_backoff_hint(normalized_key, cleaned_value)
            if numeric_hint:
                _set_backoff_hint(numeric_hint)
            else:
                _set_backoff_hint(cleaned_value)
            return
        if category == "last_seen":
            if last_seen is None:
                last_seen = cleaned_value
            return
        if category == "last_healthy":
            if last_healthy is None:
                last_healthy = cleaned_value
            return

    for key, value in envelope.items():
        _ingest_metadata_candidate(key, value)

    for match in _WORKER_METADATA_TOKEN_PATTERN.finditer(message):
        _ingest_metadata_candidate(match.group("key"), match.group("value"))

    for key, value in _iter_structured_json_tokens(message):
        _ingest_metadata_candidate(key, value)

    if restart_count is None:
        fallback_restart = re.search(
            r"(?:attempts?|retries?|restart(?:s|_count|count)?)(?![_-]?(?:delay|second|seconds|sec|secs|ms|millis|millisecond|minute|minutes|min))(?!ing)\D*(?P<count>\d+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_restart:
            try:
                restart_count = int(fallback_restart.group("count"))
            except ValueError:
                restart_count = None

    if restart_count is None:
        multiplier_match = re.search(r"(?<![0-9A-Za-z])[x×]\s*(?P<count>\d+)", message)
        if multiplier_match:
            try:
                restart_count = int(multiplier_match.group("count"))
            except ValueError:
                restart_count = None

    if last_error is None:
        fallback_error = re.search(
            r"error\s*[=:]\s*(?P<value>\"[^\"]+\"|'[^']+'|[^;\n]+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_error:
            raw_candidate = fallback_error.group("value")
            last_error_raw_value = raw_candidate.strip()
            last_error = _clean_worker_metadata_value(raw_candidate)

    if last_error is None:
        due_match = re.search(
            r"(?:due to|because(?: of)?)\s+(?P<reason>[^;.,()]+)",
            normalized_source,
            flags=re.IGNORECASE,
        )
        if due_match:
            raw_candidate = due_match.group("reason")
            candidate = _clean_worker_metadata_value(raw_candidate)
            if candidate:
                last_error = candidate
                last_error_raw_value = raw_candidate.strip()

    if backoff_hint is None:
        fallback_backoff = re.search(
            r"backoff\s*[=:]\s*(?P<value>\"[^\"]+\"|'[^']+'|[^;\n]+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_backoff:
            _set_backoff_hint(_clean_worker_metadata_value(fallback_backoff.group("value")))

    if backoff_hint is None:
        interval_match = re.search(
            r"""
            worker\s+stalled
            (?:(?:\s*(?:[;:,.\-–—…]|->|=>|→|⇒)\s*)*)
            restart(?:ing)?
            \s+(?:in|after)\s+
            (?P<interval>[^;.,()\n]+)
            """,
            normalized_source,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        if not interval_match:
            interval_match = re.search(
                r"re(?:start(?:ing)?|starting)\s+(?:in|after)\s+(?P<interval>[^;.,()\n]+)",
                normalized_source,
                flags=re.IGNORECASE,
            )
        if interval_match:
            interval = interval_match.group("interval")
            if interval:
                cleaned_interval = _strip_interval_clause_suffix(interval)
                if cleaned_interval:
                    _set_backoff_hint(cleaned_interval)

    if backoff_hint is None:
        derived_backoff = _scan_backoff_hint_from_message(normalized_source)
        if derived_backoff:
            backoff_hint = derived_backoff

    if last_healthy is None:
        healthy_match = re.search(
            r"""
            (?:last\s+(?:known\s+)?)
            healthy
            (?:\s+(?:time|at|timestamp|check|state)?)?
            \s*(?:=|:)?\s*
            (?P<value>[^;.,()\n]+)
            """,
            normalized_source,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        if healthy_match:
            candidate = _clean_worker_metadata_value(healthy_match.group("value"))
            if candidate:
                last_healthy = candidate

    if "docker_worker_context" not in metadata:
        cleaned_message = re.sub(
            r"\s+", " ", _strip_control_sequences(normalized_source)
        ).strip()
        context_candidate = _extract_worker_context(normalized_source, cleaned_message)
        if context_candidate:
            metadata["docker_worker_context"] = context_candidate

    context_value = metadata.get("docker_worker_context")
    if context_value:
        context_descriptor = f"Affected component: {context_value}."

    if restart_count is not None and restart_count >= 0:
        metadata["docker_worker_restart_count"] = str(restart_count)
        plural = "s" if restart_count != 1 else ""
        context_details.append(
            f"Docker reported {restart_count} restart{plural} during diagnostics."
        )

    if backoff_hint:
        metadata["docker_worker_backoff"] = backoff_hint
        context_details.append(
            f"Docker advertised a restart backoff interval of {backoff_hint}."
        )

    if last_seen:
        metadata["docker_worker_last_restart"] = last_seen
        context_details.append(
            f"Last restart marker emitted by Docker: {last_seen}."
        )

    if last_healthy:
        metadata["docker_worker_last_healthy"] = last_healthy
        context_details.append(
            f"Docker last reported the worker as healthy at {last_healthy}."
        )

    if last_error:
        existing_code = metadata.get("docker_worker_last_error_code")
        normalized_error, error_detail, error_metadata = _normalise_worker_error_message(
            last_error,
            raw_original=last_error_raw_value or last_error,
            existing_code=existing_code,
        )
        if normalized_error:
            metadata["docker_worker_last_error"] = normalized_error
            metadata.update(error_metadata)
            detail_override = error_detail or metadata.get(
                "docker_worker_last_error_summary"
            )
            context_details.append(
                detail_override
                or f"Most recent worker error: {normalized_error}."
            )
    else:
        fallback_source = normalized_source or message
        fallback_error, fallback_detail, fallback_metadata = (
            _normalise_worker_error_message(
                fallback_source,
                raw_original=fallback_source,
                existing_code=metadata.get("docker_worker_last_error_code"),
            )
            if fallback_source
            else (None, None, {})
        )
        if fallback_error:
            metadata.setdefault("docker_worker_last_error", fallback_error)
            for key, value in fallback_metadata.items():
                metadata.setdefault(key, value)
            if fallback_detail and fallback_detail not in context_details:
                context_details.append(fallback_detail)

    descriptors: list[str] = []
    if context_descriptor:
        descriptors.append(context_descriptor)
    if context_details:
        if context_descriptor:
            descriptors.append(
                "Additional context: " + " ".join(context_details)
            )
        else:
            descriptors.extend(context_details)

    return descriptors, metadata


def _compose_worker_flapping_guidance(metadata: MutableMapping[str, str]) -> str | None:
    """Render a narrative for worker stall diagnostics using structured metadata."""

    try:
        telemetry = WorkerRestartTelemetry.from_metadata(metadata)
        context = _detect_runtime_context()
        assessment = _classify_worker_flapping(telemetry, context)
    except Exception as exc:  # pragma: no cover - defensive safety net
        LOGGER.debug(
            "Failed to synthesise worker flapping guidance: %s", exc, exc_info=True
        )
        fallback_sources = (
            metadata.get("docker_worker_last_error"),
            metadata.get("docker_worker_last_error_original"),
            metadata.get("docker_worker_last_error_raw"),
            _WORKER_STALLED_PRIMARY_NARRATIVE,
        )
        for candidate in fallback_sources:
            if candidate:
                return candidate
        return None

    summary = assessment.render().strip()
    severity = assessment.severity

    if summary:
        metadata.setdefault("docker_worker_health_summary", summary)
    metadata.setdefault("docker_worker_health_severity", severity)

    if assessment.reasons and "docker_worker_health_reasons" not in metadata:
        metadata["docker_worker_health_reasons"] = "; ".join(assessment.reasons)
    if assessment.details and "docker_worker_health_details" not in metadata:
        metadata["docker_worker_health_details"] = "; ".join(assessment.details)
    if assessment.remediation and "docker_worker_health_remediation" not in metadata:
        metadata["docker_worker_health_remediation"] = "; ".join(
            assessment.remediation
        )

    if assessment.metadata:
        for key, value in assessment.metadata.items():
            metadata.setdefault(key, value)

    return summary or _WORKER_STALLED_PRIMARY_NARRATIVE


def _normalise_docker_warning(message: str) -> tuple[str | None, dict[str, str]]:
    """Return a cleaned warning and metadata extracted from Docker output."""

    cleaned = _DOCKER_WARNING_PREFIX_PATTERN.sub("", message)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None, {}

    metadata: dict[str, str] = {}
    normalized_cleaned = _normalise_worker_stalled_phrase(cleaned)
    if _contains_worker_stall_signal(normalized_cleaned):
        metadata["docker_worker_health"] = "flapping"

        normalized_original = _normalise_worker_stalled_phrase(message)
        descriptors, worker_metadata = _extract_worker_flapping_descriptors(
            message,
            normalized_message=normalized_original,
        )
        metadata.update(worker_metadata)

        cleaned = _compose_worker_flapping_guidance(metadata)

        if not cleaned:
            headline = (
                "Docker Desktop reported repeated restarts of a background worker."
            )
            remediation = (
                "Restart Docker Desktop, ensure Hyper-V or WSL 2 virtualization is enabled, and "
                "allocate additional CPU/RAM to the Docker VM before retrying."
            )

            segments: list[str] = [headline]
            if descriptors:
                segments.extend(descriptors)
            segments.append(remediation)
            cleaned = " ".join(
                segment.strip() for segment in segments if segment.strip()
            )

    _redact_worker_banner_artifacts(metadata)

    for key in ("docker_worker_last_error_interpreted", "docker_worker_last_error_category"):
        classification = metadata.get(key)
        if not classification:
            continue

        canonical = _canonicalize_worker_classification(classification)
        if canonical:
            metadata[key] = canonical
            continue

        if isinstance(classification, str) and _contains_worker_stall_signal(classification):
            metadata[key] = "worker_stalled"

    return cleaned, metadata


def _scrub_residual_worker_warnings(
    messages: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Rewrite lingering ``worker stalled`` banners into actionable guidance."""

    rewritten: list[str] = []
    aggregated_metadata: dict[str, str] = {}

    for message in messages:
        processed, structured = _normalise_structured_worker_banner(
            message, aggregated_metadata
        )

        if structured:
            message = processed

        if not isinstance(message, str):
            rewritten.append(message)
            continue

        lowered = message.casefold()
        if any(sentinel in lowered for sentinel in _WORKER_GUIDANCE_SENTINELS):
            rewritten.append(message)
            continue

        if not _contains_worker_stall_signal(message):
            rewritten.append(message)
            continue

        if "docker desktop reported repeated restarts" in lowered:
            rewritten.append(message)
            continue

        cleaned, metadata = _normalise_docker_warning(message)
        if cleaned:
            rewritten.append(cleaned)
        else:
            rewritten.append(message)

        for key, value in metadata.items():
            aggregated_metadata.setdefault(key, value)

    return rewritten, aggregated_metadata


_WORKER_BANNER_CONTEXT_FIELDS: tuple[str, ...] = (
    "context",
    "component",
    "component_name",
    "componentname",
    "component_id",
    "componentid",
    "component_identifier",
    "componentidentifier",
    "component_display_name",
    "componentdisplayname",
    "component_friendly_name",
    "componentfriendlyname",
    "component_uid",
    "componentuid",
    "component_guid",
    "componentguid",
    "component_slug",
    "componentslug",
    "display_name",
    "displayname",
    "source",
    "origin",
    "worker",
    "module",
    "service",
    "scope",
    "subsystem",
    "target",
    "unit",
    "pipeline",
    "channel",
    "namespace",
)

_WORKER_BANNER_CONTEXT_NORMALIZED_KEYS: frozenset[str] = frozenset(
    re.sub(r"[^a-z0-9]", "", field.lower())
    for field in _WORKER_BANNER_CONTEXT_FIELDS
)

_WORKER_BANNER_CONTEXT_PRIORITIES: Mapping[str, int] = {
    "context": 120,
    "worker": 115,
    "component": 90,
    "componentname": 90,
    "componentid": 88,
    "componentidentifier": 88,
    "componentdisplayname": 85,
    "componentfriendlyname": 85,
    "componentuid": 83,
    "componentguid": 83,
    "componentslug": 82,
    "displayname": 80,
    "module": 75,
    "service": 70,
    "origin": 65,
    "source": 65,
    "namespace": 60,
    "target": 55,
    "unit": 55,
    "pipeline": 55,
    "channel": 55,
}


@dataclass(order=True)
class WorkerBannerCandidate:
    """Represents a potential sanitised worker banner extracted from payloads."""

    priority: int
    cleaned: str = field(compare=False)
    metadata: dict[str, str] = field(compare=False)
    raw: str = field(compare=False)


def _gather_worker_banner_context_hints(
    mapping: Mapping[str, object],
) -> list[str]:
    """Return normalised worker context hints extracted from *mapping*."""

    ranked: list[tuple[int, int, str]] = []

    for index, (key, value) in enumerate(mapping.items()):
        if not isinstance(value, str):
            continue

        normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
        if normalized_key not in _WORKER_BANNER_CONTEXT_NORMALIZED_KEYS:
            continue

        candidate = _normalize_worker_context_candidate(value)
        if not candidate or _is_worker_context_noise(candidate):
            continue

        priority = _WORKER_BANNER_CONTEXT_PRIORITIES.get(normalized_key, 10)
        ranked.append((priority, index, candidate))

    if not ranked:
        return []

    ranked.sort(key=lambda item: (-item[0], item[1]))

    ordered: list[str] = []
    seen: set[str] = set()
    for _priority, _index, candidate in ranked:
        lowered = candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(candidate)

    return ordered


def _score_worker_banner_candidate(raw: str, contexts: Sequence[str]) -> int:
    """Return a deterministic priority score for a worker banner candidate."""

    base = len(raw)
    if contexts:
        base += 25 * len(contexts)
    return base


def _build_worker_banner_candidate(
    raw: str, contexts: Sequence[str]
) -> WorkerBannerCandidate | None:
    """Construct a :class:`WorkerBannerCandidate` from ``raw`` log payloads."""

    cleaned, extracted = _normalise_docker_warning(raw)
    if not cleaned:
        cleaned = _sanitize_worker_banner_text(raw)

    if not cleaned:
        cleaned = _WORKER_STALLED_PRIMARY_NARRATIVE

    metadata: dict[str, str] = dict(extracted)

    if contexts:
        unique_contexts: list[str] = []
        seen: set[str] = set()
        for context in contexts:
            normalized = _normalize_worker_context_candidate(context)
            if not normalized or _is_worker_context_noise(normalized):
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique_contexts.append(normalized)

        if unique_contexts:
            metadata.setdefault("docker_worker_context", unique_contexts[0])
            metadata.setdefault(
                "docker_worker_contexts",
                ", ".join(unique_contexts),
            )

    fingerprint = _fingerprint_worker_banner(raw)
    if fingerprint:
        metadata.setdefault("docker_worker_last_error_banner_signature", fingerprint)
        metadata.setdefault("docker_worker_last_error_banner_fingerprint", fingerprint)

    metadata.setdefault("docker_worker_health", metadata.get("docker_worker_health", "flapping"))
    metadata.setdefault("docker_worker_last_error_banner_source", "structured")

    score = -_score_worker_banner_candidate(raw, contexts)

    return WorkerBannerCandidate(
        priority=score,
        cleaned=cleaned,
        metadata=metadata,
        raw=raw,
    )


def _extract_worker_banner_candidates_from_structured_payload(
    payload: object,
    inherited_contexts: Sequence[str] | None = None,
) -> list[WorkerBannerCandidate]:
    """Return sanitised banner candidates discovered within ``payload``."""

    contexts = list(inherited_contexts or ())
    candidates: list[WorkerBannerCandidate] = []

    if isinstance(payload, MappingABC):
        local_contexts = contexts + _gather_worker_banner_context_hints(payload)
        for child in payload.values():
            candidates.extend(
                _extract_worker_banner_candidates_from_structured_payload(
                    child, local_contexts
                )
            )
        return candidates

    if isinstance(payload, SequenceABC) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        for child in payload:
            candidates.extend(
                _extract_worker_banner_candidates_from_structured_payload(child, contexts)
            )
        return candidates

    if isinstance(payload, str):
        candidate_text = payload.strip()
        if not candidate_text:
            return candidates

        lowered = candidate_text.casefold()
        if any(sentinel in lowered for sentinel in _WORKER_GUIDANCE_SENTINELS):
            return candidates

        if (
            "docker desktop automatically restarted" in lowered
            and "after it stalled" in lowered
        ):
            return candidates

        if _contains_worker_stall_signal(candidate_text):
            candidate = _build_worker_banner_candidate(payload, contexts)
            if candidate:
                candidates.append(candidate)
            return candidates

        if candidate_text[0] in "[{":
            try:
                decoded = json.loads(candidate_text)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return candidates
            return _extract_worker_banner_candidates_from_structured_payload(
                decoded, contexts
            )

    return candidates


def _normalise_structured_worker_banner(
    message: object, metadata: MutableMapping[str, str]
) -> tuple[object, bool]:
    """Normalise structured payloads that embed worker stall diagnostics."""

    candidates = _extract_worker_banner_candidates_from_structured_payload(message)
    if not candidates:
        return message, False

    best = min(candidates)
    for key, value in best.metadata.items():
        metadata.setdefault(key, value)

    return best.cleaned, True


_WORKER_BANNER_FIELD_PATTERN = re.compile(
    r"""
    (?P<key>[A-Za-z0-9_.-]+)
    \s*(?:=|:)\s*
    (?P<value>
        "(?P<double>(?:\\.|[^"\\])*)"
        |
        '(?P<single>(?:\\.|[^'\\])*)'
        |
        (?P<bare>[^\s;,]+)
    )
    """,
    re.VERBOSE,
)


_WORKER_BANNER_CAUSE_PATTERN = re.compile(
    r"(?:due\s+to|because(?:\s+of)?)\s+(?P<reason>[^;.,()]+)",
    re.IGNORECASE,
)


def _derive_worker_banner_subject(raw_text: str, normalized: str) -> tuple[str | None, str | None]:
    """Return a ``(subject, reason)`` tuple extracted from worker stall banners."""

    contexts: list[str] = []
    reason: str | None = None

    def _register_context(candidate: str | None) -> None:
        if not candidate:
            return
        cleaned = _clean_worker_metadata_value(candidate)
        if not cleaned:
            return
        collapsed = _collapse_worker_restart_sequences(cleaned)
        collapsed = re.sub(
            r"\bworker\s+stall(?:ed|ing|s)?\b",
            "",
            collapsed,
            flags=re.IGNORECASE,
        )
        collapsed = re.sub(r"\s+", " ", collapsed).strip(" -:;,")
        if not collapsed:
            return
        collapsed = collapsed.strip("[]{}()<>:;,.\"'")
        if not collapsed:
            return
        tokens = [token for token in collapsed.split(" ") if token]
        while tokens and _is_worker_context_noise(tokens[0]):
            tokens.pop(0)
        collapsed = " ".join(tokens)
        if not collapsed:
            return
        if _is_worker_context_noise(collapsed):
            return
        lowered = collapsed.lower()
        if lowered in {"component", "context"}:
            return
        contexts.append(collapsed)

    for match in _WORKER_BANNER_FIELD_PATTERN.finditer(raw_text):
        key = match.group("key")
        if not key:
            continue
        canonical_key = _canonicalize_warning_key(key)
        if canonical_key in {"lasterror", "last_error", "lasterrormessage", "last_error_message"}:
            value = match.group("double") or match.group("single") or match.group("bare") or ""
            _register_context(value)
            continue
        if canonical_key not in _WORKER_BANNER_CONTEXT_FIELDS:
            continue
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        _register_context(value)

    if not contexts:
        prefix_match = re.search(
            r"(?P<prefix>[^;.,()]{1,160})\bworker\s+stall(?:ed|ing|s)?",
            normalized,
            flags=re.IGNORECASE,
        )
        if prefix_match:
            candidate = prefix_match.group("prefix")
            _register_context(candidate)

    if not contexts:
        for token in normalized.split():
            if token.lower().startswith("worker"):
                break
            tentative = re.sub(r"[\s_-]+", " ", token).strip()
            if not tentative:
                continue
            tentative = tentative.strip("[]{}()<>:;,.\"'")

            if "=" in tentative:
                key_part, _, value_part = tentative.partition("=")
                key_token = re.sub(r"[^a-z0-9]+", "", key_part.lower())
                value_token = value_part.strip().strip("\"'")
                if (
                    key_token in _WORKER_CONTEXT_BASE_KEY_TOKENS
                    and value_token
                    and not _is_worker_context_noise(value_token)
                ):
                    contexts.append(value_token)
                    break

            if tentative and not _is_worker_context_noise(tentative):
                contexts.append(tentative)
                break

    if not reason:
        cause_match = _WORKER_BANNER_CAUSE_PATTERN.search(normalized)
        if cause_match:
            raw_reason = _clean_worker_metadata_value(cause_match.group("reason"))
            if raw_reason and not _contains_worker_stall_signal(raw_reason):
                reason = raw_reason

    subject = contexts[0] if contexts else None
    if subject:
        subject = re.sub(r"^(?:the|an|a)\s+", "", subject, flags=re.IGNORECASE).strip()
        if subject:
            subject = subject.strip("[]{}()<>:;,.\"'")
            subject = re.sub(r"\s+", " ", subject)

    return subject or None, reason


def _format_worker_restart_reason(reason: str | None) -> str | None:
    """Return a cleaned clause describing *reason* for a worker restart."""

    if not reason:
        return None

    cleaned = _clean_worker_metadata_value(reason)
    if not cleaned:
        return None

    normalized = _normalise_worker_stalled_phrase(cleaned)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .;:,-\u2013\u2014")
    if not normalized:
        return None

    lowered = normalized.casefold()
    if _contains_worker_stall_signal(normalized):
        return None

    for prefix in ("due to ", "because of ", "because "):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].lstrip()
            lowered = normalized.casefold()
            break

    if not normalized:
        return None

    return normalized


def _render_worker_banner_narrative(
    subject: str | None, reason: str | None
) -> str:
    """Return a natural language description for a worker restart banner."""

    base = _WORKER_STALLED_PRIMARY_NARRATIVE
    if subject:
        normalized_subject = subject
        if not re.search(r"\bworker\b", normalized_subject, re.IGNORECASE):
            normalized_subject = f"{normalized_subject} worker"
        normalized_subject = re.sub(r"\s+", " ", normalized_subject).strip()
        base = (
            f"Docker Desktop automatically restarted the {normalized_subject} after it stalled"
        )

    formatted_reason = _format_worker_restart_reason(reason)
    if formatted_reason:
        base = f"{base} due to {formatted_reason}"

    return base


def _canonicalize_worker_narrative(value: str) -> str:
    """Collapse contextual worker narratives into the canonical guidance."""

    collapsed = re.sub(r"\s+", " ", value).strip()
    lowered = collapsed.casefold()
    prefix = "docker desktop automatically restarted the "
    if lowered.startswith(prefix) and " a background worker " not in lowered:
        return _WORKER_STALLED_PRIMARY_NARRATIVE
    return collapsed


def _sanitize_worker_banner_text(raw_text: str | None) -> str:
    """Return a banner with ``worker stalled`` phrasing rewritten for humans."""

    if not raw_text:
        return _WORKER_STALLED_PRIMARY_NARRATIVE

    canonical_prefix = "docker desktop automatically restarted"
    raw_collapsed = re.sub(r"\s+", " ", raw_text).strip()
    if raw_collapsed.casefold().startswith(canonical_prefix):
        return raw_collapsed

    normalized = _normalise_worker_stalled_phrase(raw_text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return _WORKER_STALLED_PRIMARY_NARRATIVE

    if normalized.casefold().startswith(canonical_prefix):
        return normalized

    subject, reason = _derive_worker_banner_subject(raw_text, normalized)

    lowered = normalized.casefold()
    if _contains_worker_stall_signal(normalized) or "worker stalled" in lowered or (
        "stall" in lowered and "restart" in lowered
    ):
        narrative = _render_worker_banner_narrative(subject, reason)
        if "worker stalled" in narrative.casefold():
            return _WORKER_STALLED_PRIMARY_NARRATIVE
        return narrative

    return normalized


def _enforce_worker_banner_sanitization(
    messages: Iterable[str], metadata: MutableMapping[str, str]
) -> list[str]:
    """Ensure lingering ``worker stalled; restarting`` banners are normalised.

    Despite the aggressive normalisation performed earlier in the pipeline we
    have occasionally observed Docker Desktop emitting warning strings late in
    the processing chain that still contain the literal ``worker stalled;
    restarting`` banner.  These typically appear when ancillary tooling echoes
    cached diagnostics or when third-party wrappers duplicate stderr streams
    after our primary scrubbers have executed.  To guarantee we never surface
    the raw banner to end users we perform a final pass that rewrites any
    lingering matches into the structured narrative generated by
    :func:`_normalise_docker_warning`.  Metadata derived from the sanitisation is
    merged without clobbering fields that have already been populated upstream.
    """

    harmonised: list[str] = []

    for message in messages:
        if isinstance(message, str):
            lowered_message = message.casefold()
            if any(
                sentinel in lowered_message
                for sentinel in _WORKER_GUIDANCE_SENTINELS
            ):
                harmonised.append(message)
                continue

        processed, structured = _normalise_structured_worker_banner(message, metadata)

        if structured:
            message = processed

        if not isinstance(message, str):
            harmonised.append(message)
            continue

        normalized = _normalise_worker_stalled_phrase(message)
        banner_detected = bool(_WORKER_STALLED_BANNER_PATTERN.search(normalized))
        signal_detected = banner_detected or _contains_worker_stall_signal(normalized)

        if not signal_detected:
            harmonised.append(message)
            continue

        cleaned, extracted = _normalise_docker_warning(message)
        if cleaned:
            harmonised.append(cleaned)
        else:
            harmonised.append(_WORKER_STALLED_PRIMARY_NARRATIVE)

        if not extracted:
            extracted = {}
        for key, value in extracted.items():
            metadata.setdefault(key, value)

        metadata.setdefault("docker_worker_health", "flapping")

        # ``_normalise_docker_warning`` may emit the canonical guidance without
        # an explicit context when the raw banner lacked structured metadata.
        # Ensure we always retain the original banner for diagnostics so future
        # enrichment stages can infer error codes or restart telemetry.
        preserved_raw: str | None = None
        preserved = extracted.get("docker_worker_last_error_banner_preserved")
        sanitized_preserved: str | None = None

        if preserved:
            preserved_raw = preserved
            sanitized_preserved = _sanitize_worker_banner_text(preserved)

        if not sanitized_preserved:
            candidate = normalized.strip()
            if candidate and _contains_worker_stall_signal(candidate):
                preserved_raw = preserved_raw or candidate
                sanitized_preserved = _sanitize_worker_banner_text(candidate)

        if sanitized_preserved:
            existing_preserved = metadata.get(
                "docker_worker_last_error_banner_preserved"
            )
            if not existing_preserved or _contains_worker_stall_signal(
                existing_preserved
            ):
                metadata[
                    "docker_worker_last_error_banner_preserved"
                ] = sanitized_preserved

            preserved_samples = metadata.get(
                "docker_worker_last_error_banner_preserved_samples"
            )
            if (
                not preserved_samples
                or _contains_worker_stall_signal(preserved_samples)
            ):
                metadata[
                    "docker_worker_last_error_banner_preserved_samples"
                ] = sanitized_preserved

        if preserved_raw:
            raw_candidate = re.sub(r"\s+", " ", preserved_raw).strip()
            if raw_candidate:
                metadata.setdefault(
                    "docker_worker_last_error_banner_preserved_raw",
                    raw_candidate,
                )
                metadata.setdefault(
                    "docker_worker_last_error_banner_preserved_raw_samples",
                    raw_candidate,
                )

        signature = extracted.get("docker_worker_last_error_banner_signature")
        if not signature:
            signature = _fingerprint_worker_banner(normalized)
        if signature:
            metadata.setdefault("docker_worker_last_error_banner_signature", signature)

    _redact_worker_banner_artifacts(metadata)

    return harmonised


def _stitch_worker_banner_fragments(messages: Sequence[object]) -> list[object]:
    """Merge split worker stall banners emitted as adjacent log lines."""

    entries = list(messages)
    stitched: list[object] = []
    total = len(entries)
    index = 0

    while index < total:
        current = entries[index]

        if not isinstance(current, str):
            stitched.append(current)
            index += 1
            continue

        if not _looks_like_worker_restart_fragment(current):
            stitched.append(current)
            index += 1
            continue

        combined = current
        consumed = 0
        probe = index + 1

        while probe < total:
            candidate = entries[probe]
            if not isinstance(candidate, str):
                break

            joined = f"{combined.rstrip()} {candidate.lstrip()}"
            normalized = _normalize_worker_banner_characters(joined)
            lowered = normalized.casefold()

            if "restart" in lowered:
                combined = joined
                consumed = probe - index
                break

            if not candidate.strip():
                combined = joined
                consumed = probe - index
                probe += 1
                continue

            if _looks_like_worker_restart_fragment(candidate) or _contains_worker_stall_signal(candidate):
                combined = joined
                consumed = probe - index
                probe += 1
                continue

            break

        stitched.append(combined)
        index += consumed + 1 if consumed else 1

    return stitched


def _looks_like_truncated_worker_error_token(value: str | None) -> bool:
    """Return ``True`` when *value* appears to be a truncated worker error."""

    if not value:
        return True

    candidate = value.strip()
    if not candidate:
        return True

    def _count_quotes(text: str, quote: str) -> int:
        count = 0
        escaped = False
        for char in text:
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                count += 1
            escaped = False
        return count

    if _count_quotes(candidate, '"') % 2 == 1 or _count_quotes(candidate, "'") % 2 == 1:
        return True

    opening_tokens = {"{": "}", "[": "]", "(": ")"}
    closing_tokens = {v: k for k, v in opening_tokens.items()}
    stack: list[str] = []
    escaped = False
    for char in candidate:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if escaped:
            escaped = False
            continue
        if stack and stack[-1] in {'"', "'"}:
            if char == stack[-1]:
                stack.pop()
            elif char == "\\":
                escaped = True
            continue
        if char in {'"', "'"}:
            stack.append(char)
            continue
        if char in opening_tokens:
            stack.append(opening_tokens[char])
            continue
        if char in closing_tokens:
            if not stack or stack[-1] != char:
                return True
            stack.pop()

    if stack and any(token not in {'"', "'"} for token in stack):
        return True

    lowered = candidate.casefold()
    if lowered.startswith('{"message"') and not candidate.endswith('}'):
        return True
    if lowered.startswith('"message"') and 'stalled' not in lowered and 'restart' not in lowered:
        return True

    return False


def _looks_like_worker_restart_fragment(message: str) -> bool:
    """Return ``True`` when *message* appears to be a truncated stall banner."""

    if not message:
        return False

    normalized = _normalize_worker_banner_characters(message)
    harmonised = _normalise_worker_stalled_phrase(normalized)
    collapsed = re.sub(r"\s+", " ", harmonised).strip()
    if not collapsed:
        return False

    lowered = collapsed.casefold()
    if any(sentinel in lowered for sentinel in _WORKER_GUIDANCE_SENTINELS):
        return False

    if "restart" in lowered:
        return False

    if "worker stalled" not in lowered:
        return False

    if len(collapsed) > 200:
        return False

    return bool(_WORKER_STALLED_FRAGMENT_PATTERN.search(collapsed))


def _decode_worker_base64_fragment(value: str) -> str | None:
    """Decode base64-encoded worker stall payloads when present."""

    if not value:
        return None

    candidate = value.strip()
    if len(candidate) < 32:
        return None

    compact = re.sub(r"\s+", "", candidate)
    if len(compact) % 4:
        return None

    if not (
        _WORKER_BASE64_PATTERN.fullmatch(compact)
        or _WORKER_BASE64_URLSAFE_PATTERN.fullmatch(compact)
    ):
        return None

    decode_attempts = (
        lambda data: base64.b64decode(data, validate=True),
        lambda data: base64.b64decode(data, altchars=b"-_", validate=True),
    )

    for decoder in decode_attempts:
        try:
            decoded_bytes = decoder(compact)
        except (binascii.Error, ValueError):
            continue

        if not decoded_bytes:
            continue

        try:
            decoded_text = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            decoded_text = decoded_bytes.decode("utf-8", "replace")

        normalized = decoded_text.strip()
        if not normalized:
            continue

        collapsed = _normalise_worker_stalled_phrase(normalized)
        if _contains_worker_stall_signal(collapsed):
            return normalized

    return None


def _guarantee_worker_banner_suppression(
    messages: Iterable[str], metadata: MutableMapping[str, str]
) -> list[str]:
    """Replace any literal ``worker stalled; restarting`` phrases with guidance.

    The sanitisation pipeline is intentionally defensive and multi-layered, yet
    empirical telemetry from Windows hosts has shown that ancillary tooling can
    occasionally replay cached diagnostics after the primary normalisation
    passes have executed.  When that happens a verbatim ``worker stalled;
    restarting`` banner can re-enter the stream via logging adapters or custom
    stdout/stderr multiplexers.  To guarantee end users never observe the raw
    banner, perform a final sweep that rewrites any literal match into the
    structured narrative returned by :func:`_normalise_docker_warning` while
    preserving previously captured metadata.
    """

    safeguarded: list[str] = []
    stitched_messages = _stitch_worker_banner_fragments(list(messages))

    for message in stitched_messages:
        processed, structured = _normalise_structured_worker_banner(message, metadata)

        if structured:
            message = processed

        if not isinstance(message, str):
            safeguarded.append(message)
            continue

        normalized = _normalize_worker_banner_characters(message)
        if (
            not _contains_literal_worker_restart_banner(message, normalized=normalized)
            and not _looks_like_literal_worker_restart_banner(message)
            and not _looks_like_worker_restart_fragment(message)
        ):
            safeguarded.append(message)
            continue

        cleaned, extracted = _normalise_docker_warning(message)
        if cleaned:
            safeguarded.append(cleaned)
        else:
            safeguarded.append(_WORKER_STALLED_PRIMARY_NARRATIVE)

        if extracted:
            for key, value in extracted.items():
                metadata.setdefault(key, value)

        metadata.setdefault("docker_worker_health", "flapping")

    return safeguarded


def _finalize_worker_banner_sequences(
    messages: Iterable[object], metadata: MutableMapping[str, str]
) -> list[object]:
    """Ensure raw ``worker stalled; restarting`` banners never escape."""

    sanitized: list[object] = []

    for message in messages:
        if not isinstance(message, str):
            sanitized.append(message)
            continue

        normalized = _normalize_worker_banner_characters(message)
        if not _contains_literal_worker_restart_banner(message, normalized=normalized):
            sanitized.append(message)
            continue

        cleaned, extracted = _normalise_docker_warning(message)

        sanitized.append(cleaned or _WORKER_STALLED_PRIMARY_NARRATIVE)

        if extracted:
            for key, value in extracted.items():
                metadata.setdefault(key, value)

        metadata.setdefault("docker_worker_health", "flapping")

    return sanitized


def _sanitize_worker_json_structure(
    payload: object, metadata: MutableMapping[str, str]
) -> tuple[object, bool]:
    """Recursively scrub ``worker stalled`` banners from JSON-compatible objects."""

    if isinstance(payload, str):
        normalized = _normalise_worker_stalled_phrase(payload)
        lowered = normalized.casefold()
        normalized_banner = _normalize_worker_banner_characters(payload)
        banner_like = _contains_literal_worker_restart_banner(
            payload, normalized=normalized_banner
        ) or ("restart" in lowered and _contains_worker_stall_signal(normalized))

        if not banner_like:
            return payload, False

        cleaned, extracted = _normalise_docker_warning(payload)
        if not cleaned:
            cleaned = _WORKER_STALLED_PRIMARY_NARRATIVE

        if extracted:
            for key, value in extracted.items():
                metadata.setdefault(key, value)

        return cleaned, cleaned != payload

    if isinstance(payload, MappingABC):
        updated: dict[object, object] = {}
        changed = False

        for key, value in payload.items():
            sanitized, mutated = _sanitize_worker_json_structure(value, metadata)
            updated[key] = sanitized
            changed = changed or mutated

        return updated, changed

    if isinstance(payload, SequenceABC) and not isinstance(
        payload, (str, bytes, bytearray, memoryview)
    ):
        updated_sequence: list[object] = []
        changed = False

        for item in payload:
            sanitized, mutated = _sanitize_worker_json_structure(item, metadata)
            updated_sequence.append(sanitized)
            changed = changed or mutated

        if isinstance(payload, tuple):
            return tuple(updated_sequence), changed

        return updated_sequence, changed

    return payload, False


def _sanitize_worker_json_payload(
    raw_text: str, metadata: MutableMapping[str, str]
) -> tuple[str | None, bool]:
    """Sanitise JSON blobs that embed worker stall diagnostics."""

    trimmed = raw_text.strip()
    if not trimmed or trimmed[0] not in "[{":
        return None, False

    try:
        decoded = json.loads(trimmed)
    except ValueError:
        return None, False

    sanitized, changed = _sanitize_worker_json_structure(decoded, metadata)
    if not changed:
        return None, False

    rendered = json.dumps(sanitized, separators=(",", ":"), ensure_ascii=False)
    return rendered, True


def _canonicalize_worker_classification(value: object) -> str | None:
    """Return a canonical worker classification token derived from *value*."""

    text_value = _coerce_textual_value(value)
    if not text_value:
        return None

    cleaned = _clean_worker_metadata_value(text_value)
    if not cleaned:
        return None

    lowered = cleaned.casefold()
    token_candidate = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")

    if token_candidate in _WORKER_ERROR_CODE_NORMALISATION:
        return _WORKER_ERROR_CODE_NORMALISATION[token_candidate]

    if lowered in _WORKER_ERROR_CODE_NORMALISATION:
        return _WORKER_ERROR_CODE_NORMALISATION[lowered]

    harmonised = _normalise_worker_stalled_phrase(cleaned)
    harmonised_lower = harmonised.casefold()
    harmonised_token = re.sub(r"[^a-z0-9]+", "_", harmonised_lower).strip("_")

    if harmonised_token in _WORKER_ERROR_CODE_NORMALISATION:
        return _WORKER_ERROR_CODE_NORMALISATION[harmonised_token]

    harmonised_collapsed = re.sub(r"\s+", " ", harmonised_lower).strip()
    narrative_code = _WORKER_ERROR_NARRATIVE_LOOKUP.get(harmonised_collapsed)
    if narrative_code:
        return narrative_code

    for pattern, code, _narrative in _WORKER_ERROR_NORMALISERS:
        if pattern.search(harmonised_lower) or pattern.search(lowered):
            return "worker_stalled" if code == "stalled_restart" else code

    if _contains_worker_stall_signal(cleaned):
        return "worker_stalled"

    return None


def _finalize_worker_banner_metadata(metadata: MutableMapping[str, str]) -> None:
    """Purge residual literal stall banners from metadata artefacts."""

    for key, raw_value in list(metadata.items()):
        text_value = _coerce_textual_value(raw_value)
        if not text_value:
            continue

        if key in _WORKER_METADATA_CLASSIFICATION_KEYS:
            canonical = _canonicalize_worker_classification(text_value)
            if canonical:
                metadata[key] = canonical
                continue

            if _contains_worker_stall_signal(text_value):
                metadata[key] = "worker_stalled"
                continue

            cleaned = _clean_worker_metadata_value(text_value)
            if cleaned:
                metadata[key] = cleaned
            continue

        original_value = text_value

        json_sanitized, mutated = _sanitize_worker_json_payload(text_value, metadata)
        if mutated and json_sanitized is not None:
            normalized_json = _normalize_worker_banner_characters(json_sanitized)
            lowered_json = normalized_json.casefold()
            if not _contains_literal_worker_restart_banner(
                json_sanitized, normalized=normalized_json
            ) and not (
                "restart" in lowered_json and _contains_worker_stall_signal(normalized_json)
            ):
                metadata[key] = json_sanitized
                digest = _fingerprint_worker_banner(original_value)
                if digest:
                    fingerprint_key = f"{key}_fingerprint"
                    metadata.setdefault(fingerprint_key, digest)
                continue

            text_value = json_sanitized

        normalized = _normalize_worker_banner_characters(text_value)
        lowered = normalized.casefold()
        if not _contains_literal_worker_restart_banner(
            text_value, normalized=normalized
        ) and not ("restart" in lowered and _contains_worker_stall_signal(normalized)):
            continue

        sanitized_value = _sanitize_worker_banner_text(text_value)
        digest = _fingerprint_worker_banner(original_value)

        if sanitized_value and sanitized_value != metadata.get(key):
            metadata[key] = sanitized_value
        elif isinstance(raw_value, (bytes, bytearray, memoryview)):
            metadata[key] = text_value

        fingerprint_key = f"{key}_fingerprint"
        if digest and not metadata.get(fingerprint_key):
            metadata[fingerprint_key] = digest


_WORKER_METADATA_CLASSIFICATION_KEYS: frozenset[str] = frozenset(
    {
        "docker_worker_last_error_category",
        "docker_worker_last_error_interpreted",
    }
)


_WORKER_METADATA_SANITIZE_RULES: tuple[tuple[str, Literal["single", "multi"]], ...] = (
    ("docker_worker_last_error", "single"),
    ("docker_worker_last_error_original", "single"),
    ("docker_worker_last_error_raw", "single"),
    ("docker_worker_last_error_banner", "single"),
    ("docker_worker_last_error_narrative", "single"),
    ("docker_worker_last_error_summary", "single"),
    ("docker_worker_last_error_interpreted", "single"),
    ("docker_worker_last_error_details", "single"),
    ("docker_worker_last_error_remediation", "single"),
    ("docker_worker_last_error_structured_message", "single"),
    ("docker_worker_last_error_samples", "multi"),
    ("docker_worker_last_error_original_samples", "multi"),
    ("docker_worker_last_error_raw_samples", "multi"),
    ("docker_worker_last_error_banner_samples", "multi"),
    ("docker_worker_last_error_narrative_samples", "multi"),
    ("docker_worker_last_error_summary_samples", "multi"),
    ("docker_worker_last_error_details_samples", "multi"),
    ("docker_worker_last_error_remediation_samples", "multi"),
    ("docker_worker_last_error_structured_message_samples", "multi"),
)


# Connector tokens that appear between ``worker stalled`` and ``restarting``.
#
# ``Docker Desktop`` on Windows localises portions of its diagnostics based on
# the host locale.  When non-English locales are active we routinely observe the
# ``worker stalled; restarting`` banner rendered with fullwidth punctuation or
# inverted question/exclamation marks.  Normalising all of the observed
# separators ensures the sanitisation pipeline continues to collapse the warning
# into the canonical guidance regardless of the user's locale.
_WORKER_ASCII_RESTART_CONNECTORS: tuple[str, ...] = (
    ";",
    ":",
    ",",
    ".",
    "-",
    "->",
    "=>",
    "|",
    "/",
    "\\",
    "…",
    "⋯",
    "·",
    "•",
)

_WORKER_UNICODE_RESTART_CONNECTORS: tuple[str, ...] = (
    "—",
    "–",
    "―",
    "→",
    "⇒",
    "／",
    "＼",
)

_WORKER_PUNCTUATION_RESTART_CONNECTORS: tuple[str, ...] = (
    "!",
    "?",
    "！？",
    "？！",
    "!?",
    "?!",
    "！",
    "？",
    "，",
    "．",
    "｡",
    "。",
    "、",
    "﹔",
    "﹕",
    "﹒",
    "﹑",
    "﹗",
    "﹖",
    "；",
    "：",
    "‧",
    "・",
    "･",
    "／",
    "＼",
)

_WORKER_LITERAL_RESTART_CONNECTORS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *_WORKER_ASCII_RESTART_CONNECTORS,
            *_WORKER_UNICODE_RESTART_CONNECTORS,
            *_WORKER_PUNCTUATION_RESTART_CONNECTORS,
        )
    )
)

_WORKER_SORTED_RESTART_CONNECTORS: tuple[str, ...] = tuple(
    sorted(_WORKER_LITERAL_RESTART_CONNECTORS, key=len, reverse=True)
)

_WORKER_RESTART_CONNECTOR_PATTERN = "|".join(
    re.escape(token) for token in _WORKER_SORTED_RESTART_CONNECTORS
)

_WORKER_WHITESPACE_CONNECTOR_PATTERN = r"\s+"

_WORKER_FRAGMENT_CONNECTORS = [
    re.escape(token) for token in _WORKER_SORTED_RESTART_CONNECTORS
]
_WORKER_FRAGMENT_CONNECTORS.append("-+")
_WORKER_FRAGMENT_CONNECTORS.append(_WORKER_WHITESPACE_CONNECTOR_PATTERN)

_WORKER_RESTART_SEPARATOR_PATTERN = (
    rf"(?:\s*(?:{_WORKER_RESTART_CONNECTOR_PATTERN})\s*|{_WORKER_WHITESPACE_CONNECTOR_PATTERN})"
)

_WORKER_FRAGMENT_CONNECTOR_PATTERN = "|".join(_WORKER_FRAGMENT_CONNECTORS)

_WORKER_RESTART_CONNECTOR_STRIPPER = re.compile(
    rf"^(?:{_WORKER_RESTART_SEPARATOR_PATTERN})",
    re.IGNORECASE,
)

_WORKER_RESTART_SUFFIX_PATTERN = re.compile(
    rf"(?P<prefix>\b(?:worker\s+)?{_WORKER_STALL_ROOT_PATTERN})"
    rf"\s*(?:{_WORKER_RESTART_SEPARATOR_PATTERN})\s*"
    r"re[-\s]?start(?:ed|ing)?(?P<tail>[^\r\n]*)",
    re.IGNORECASE,
)


_WORKER_STALLED_FRAGMENT_PATTERN = re.compile(
    rf"""
    worker[\s_-]+stalled
    (?:
        \s*(?:{_WORKER_FRAGMENT_CONNECTOR_PATTERN})
    )?
    \s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


_WORKER_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_WORKER_BASE64_URLSAFE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")


def _looks_like_literal_worker_restart_banner(message: str) -> bool:
    """Return ``True`` when *message* resembles the raw stall banner."""

    if not message:
        return False

    normalized = _normalize_worker_banner_characters(message)
    harmonised = _normalise_worker_stalled_phrase(normalized)
    normalized = re.sub(r"\s+", " ", harmonised).strip()
    if not normalized:
        return False

    lowered = normalized.casefold()
    if any(sentinel in lowered for sentinel in _WORKER_GUIDANCE_SENTINELS):
        return False

    if "worker" not in lowered or "restart" not in lowered:
        return False

    canonical = re.sub(
        r"re\s*[-_/]?\s*start(?:\s*[-_/]?\s*(ed|ing))?",
        lambda match: "restart" + (match.group(1) or ""),
        lowered,
    )
    canonical = re.sub(r"\s+", "", canonical)

    stall_roots = (
        "workerstall",
        "workerstalls",
        "workerstalled",
        "workerstalling",
        "workersstall",
        "workersstalled",
        "workersstalling",
    )
    restart_roots = ("restart", "restarted", "restarting")

    for stall_root in stall_roots:
        if stall_root not in canonical:
            continue
        for connector in _WORKER_LITERAL_RESTART_CONNECTORS:
            connector_token = connector.replace(" ", "")
            if not connector_token:
                continue
            for restart_root in restart_roots:
                target = f"{stall_root}{connector_token}{restart_root}"
                if target in canonical:
                    return True

    return False


def _collapse_worker_restart_sequences(value: str) -> str:
    """Rewrite ``stalled; restarting`` phrases into natural language."""

    if not value:
        return value

    def _canonicalize_prefix(prefix: str) -> str:
        token = prefix.strip()
        if not token:
            return "stalled"

        lowered = token.lower()
        if lowered.startswith("worker"):
            canonical = "worker stalled"
        else:
            canonical = "stalled"

        if token.isupper():
            return canonical.upper()
        if token[0].isupper():
            return canonical.title() if canonical != "worker stalled" else "Worker stalled"
        return canonical

    def _normalise_tail(prefix: str, tail: str) -> str:
        cleaned = tail.strip()
        if cleaned:
            cleaned = _WORKER_RESTART_CONNECTOR_STRIPPER.sub("", cleaned, count=1)
            cleaned = cleaned.strip()
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        scheduling_tokens = ("in ", "within ", "after ", "before ", "pending", "again", "backoff", "once ")
        contextual_tokens = ("due to", "because", "as ", "when ", "while ", "from ")

        if lowered.startswith(contextual_tokens):
            return cleaned
        if lowered.startswith(scheduling_tokens):
            return f"(restart {cleaned})"
        if cleaned.startswith(("(", "[")):
            return cleaned
        if prefix.lower().startswith("worker ") and not cleaned.lower().startswith("worker"):
            return cleaned
        return cleaned

    def _rewrite(match: re.Match[str]) -> str:
        prefix = _canonicalize_prefix(match.group("prefix") or "")
        tail = match.group("tail") or ""
        normalised_tail = _normalise_tail(prefix, tail)
        if normalised_tail:
            combined = f"{prefix} {normalised_tail}".strip()
        else:
            combined = prefix
        return re.sub(r"\s{2,}", " ", combined)

    collapsed = _WORKER_RESTART_SUFFIX_PATTERN.sub(_rewrite, value)
    return re.sub(r"\s{2,}", " ", collapsed).strip()


def _merge_worker_banner_fingerprints(
    existing: object, candidates: Iterable[str]
) -> str | None:
    """Merge worker banner fingerprints into a stable, sorted string."""

    collected: set[str] = {candidate for candidate in candidates if candidate}

    if isinstance(existing, str):
        parts = [segment.strip() for segment in existing.split(",") if segment.strip()]
        collected.update(parts)
    elif isinstance(existing, IterableABC) and not isinstance(
        existing, (str, bytes, bytearray)
    ):
        for token in existing:
            if isinstance(token, str):
                stripped = token.strip()
                if stripped:
                    collected.add(stripped)

    if not collected:
        return None

    return ", ".join(sorted(collected))


_WORKER_METADATA_KEY_SANITISER = re.compile(r"[^0-9A-Za-z]+")


def _sanitize_worker_metadata_key(key: object) -> tuple[object, set[str], bool]:
    """Return a sanitised mapping key when ``worker stalled`` banners bleed into it."""

    if not isinstance(key, str):
        return key, set(), False

    normalized = _normalise_worker_stalled_phrase(key)
    if not _contains_worker_stall_signal(normalized):
        return key, set(), False

    fingerprint = _fingerprint_worker_banner(key)
    narrative = _sanitize_worker_banner_text(key)
    canonical = _canonicalize_worker_narrative(narrative)

    collapsed = _WORKER_METADATA_KEY_SANITISER.sub("_", canonical).strip("_")
    if not collapsed:
        collapsed = "docker_worker_guidance"

    collapsed = collapsed.lower()
    if fingerprint:
        collapsed = f"{collapsed}_{fingerprint}"

    return collapsed, ({fingerprint} if fingerprint else set()), collapsed != key


def _scrub_nested_worker_artifacts(
    value: object, seen: set[int] | None = None
) -> tuple[object, set[str], bool]:
    """Recursively sanitise nested metadata containers that echo worker banners."""

    if isinstance(value, (str, bytes, bytearray, memoryview)):
        textual = _coerce_textual_value(value)
        if textual is None:
            return value, set(), False
        sanitized, digest = _sanitize_worker_metadata_value(textual)
        digests: set[str] = set()
        mutated = False
        if sanitized is None:
            result: object = textual if not isinstance(value, str) else value
        else:
            result = sanitized
            mutated = sanitized != textual or not isinstance(value, str)
        if mutated and digest:
            digests.add(digest)
        return result, digests, mutated

    if value is None:
        return value, set(), False

    if seen is None:
        seen = set()

    identifier = id(value)
    if identifier in seen:
        return value, set(), False

    seen.add(identifier)

    try:
        if isinstance(value, MutableMappingABC):
            mutated = False
            digests: set[str] = set()
            sanitized_items: dict[object, object] = {}
            for key in list(value.keys()):
                sanitized_key, key_digests, key_mutated = _sanitize_worker_metadata_key(key)
                sanitized_child, child_digests, child_mutated = _scrub_nested_worker_artifacts(
                    value[key], seen
                )
                sanitized_items[sanitized_key] = sanitized_child
                mutated = mutated or child_mutated or key_mutated or sanitized_key != key
                digests.update(key_digests)
                digests.update(child_digests)
            if mutated:
                value.clear()
                for sanitized_key, sanitized_child in sanitized_items.items():
                    value[sanitized_key] = sanitized_child
            return value, digests, mutated

        if isinstance(value, MappingABC):
            mutated = False
            digests: set[str] = set()
            sanitized_items: dict[object, object] = {}
            for key, raw in value.items():
                sanitized_key, key_digests, key_mutated = _sanitize_worker_metadata_key(key)
                sanitized_child, child_digests, child_mutated = _scrub_nested_worker_artifacts(
                    raw, seen
                )
                sanitized_items[sanitized_key] = sanitized_child
                mutated = mutated or child_mutated or key_mutated or sanitized_key != key
                digests.update(key_digests)
                digests.update(child_digests)

            if not mutated:
                return value, digests, False

            try:
                reconstructed = value.__class__(sanitized_items)  # type: ignore[call-arg]
            except Exception:
                reconstructed = dict(sanitized_items)

            return reconstructed, digests, True

        if isinstance(value, MutableSequenceABC):
            mutated = False
            digests: set[str] = set()
            for index in range(len(value)):
                sanitized_child, child_digests, child_mutated = _scrub_nested_worker_artifacts(
                    value[index], seen
                )
                if child_mutated:
                    value[index] = sanitized_child
                    mutated = True
                digests.update(child_digests)
            return value, digests, mutated

        if isinstance(value, tuple):
            digests: set[str] = set()
            sanitized_items: list[object] = []
            mutated = False
            for item in value:
                sanitized_child, child_digests, child_mutated = _scrub_nested_worker_artifacts(
                    item, seen
                )
                sanitized_items.append(sanitized_child)
                mutated = mutated or child_mutated
                digests.update(child_digests)
            if mutated:
                return tuple(sanitized_items), digests, True
            return value, digests, False

        if isinstance(value, frozenset):
            digests: set[str] = set()
            sanitized_items: list[object] = []
            mutated = False
            for item in value:
                sanitized_child, child_digests, child_mutated = _scrub_nested_worker_artifacts(
                    item, seen
                )
                sanitized_items.append(sanitized_child)
                mutated = mutated or child_mutated
                digests.update(child_digests)
            if mutated:
                return frozenset(sanitized_items), digests, True
            return value, digests, False

        if isinstance(value, set):
            digests: set[str] = set()
            sanitized_items: list[object] = []
            mutated = False
            for item in list(value):
                sanitized_child, child_digests, child_mutated = _scrub_nested_worker_artifacts(
                    item, seen
                )
                sanitized_items.append(sanitized_child)
                mutated = mutated or child_mutated
                digests.update(child_digests)
            if mutated:
                value.clear()
                for sanitized_item in sanitized_items:
                    value.add(sanitized_item)
            return value, digests, mutated

        if isinstance(value, SequenceABC):
            digests: set[str] = set()
            sanitized_items: list[object] = []
            mutated = False
            for item in value:
                sanitized_child, child_digests, child_mutated = _scrub_nested_worker_artifacts(
                    item, seen
                )
                sanitized_items.append(sanitized_child)
                mutated = mutated or child_mutated
                digests.update(child_digests)
            if mutated:
                try:
                    reconstructed = type(value)(sanitized_items)  # type: ignore[call-arg]
                except Exception:
                    reconstructed = list(sanitized_items)
                return reconstructed, digests, True
            return value, digests, False

        return value, set(), False
    finally:
        seen.discard(identifier)


def _redact_worker_banner_artifacts(metadata: MutableMapping[str, str]) -> None:
    """Scrub lingering worker stall phrases from metadata artefacts.

    Docker diagnostics frequently embed raw banner strings inside metadata
    fields such as ``docker_worker_last_error_banner_preserved_raw`` in
    addition to the cleaned narrative that is surfaced to the user.  The raw
    payload is valuable when debugging new Docker Desktop releases because it
    preserves the exact punctuation emitted by the daemon.  Unfortunately the
    literal ``worker stalled; restarting`` phrase can bleed into user facing
    summaries when the metadata dictionary is later rendered verbatim (for
    example by troubleshooting utilities that dump the entire structure).

    To keep diagnostics noise-free without discarding signal, we rewrite any
    metadata value that still contains a worker stall banner into the canonical
    narrative generated by :func:`_sanitize_worker_banner_text`.  The original
    payload is replaced with a digest so operators can still correlate logs when
    investigating regressions.
    """

    candidate_keys = [
        "docker_worker_last_error_banner_raw",
        "docker_worker_last_error_banner_raw_samples",
        "docker_worker_last_error_banner_preserved_raw",
        "docker_worker_last_error_banner_preserved_raw_samples",
        "docker_worker_last_error_structured_message_raw",
    ]

    sanitized_primary_keys: set[str] = {
        "docker_worker_last_error_banner_preserved",
        "docker_worker_last_error_banner_preserved_samples",
        "docker_worker_last_error_structured_message",
        "docker_worker_last_error_structured_message_raw",
    }
    sanitized_primary_keys.update(candidate_keys)
    sanitized_primary_keys.update(key for key, _ in _WORKER_METADATA_SANITIZE_RULES)

    for key in candidate_keys:
        raw_value = metadata.get(key)
        text_value = _coerce_textual_value(raw_value)
        if not text_value:
            continue

        sanitized, digest = _sanitize_worker_metadata_value(text_value)
        if sanitized is None:
            if isinstance(raw_value, (bytes, bytearray, memoryview)):
                metadata[key] = text_value
            continue

        if digest and not metadata.get(f"{key}_fingerprint"):
            metadata[f"{key}_fingerprint"] = digest

        metadata[key] = sanitized

    for key, mode in _WORKER_METADATA_SANITIZE_RULES:
        raw_value = metadata.get(key)
        text_value = _coerce_textual_value(raw_value)
        if not text_value:
            continue

        normalized = _normalise_worker_stalled_phrase(text_value)
        if not _contains_worker_stall_signal(normalized):
            continue

        fingerprint_key = f"{key}_fingerprint"
        if mode == "multi":
            sanitized_value, digest = _sanitize_worker_metadata_value(
                text_value, prefer_canonical=True
            )
            if sanitized_value is None:
                continue
            if sanitized_value != text_value:
                metadata[key] = sanitized_value
        else:
            digest = _fingerprint_worker_banner(text_value)
            sanitized_value = _sanitize_worker_banner_text(text_value)
            if sanitized_value and sanitized_value != text_value:
                metadata[key] = _canonicalize_worker_narrative(sanitized_value)

        if digest and not metadata.get(fingerprint_key):
            metadata[fingerprint_key] = digest

    for key, raw_value in list(metadata.items()):
        text_value = _coerce_textual_value(raw_value)
        if text_value is None:
            continue
        if key in sanitized_primary_keys:
            continue
        if key.endswith("_fingerprint"):
            continue

        normalized = _normalise_worker_stalled_phrase(text_value)
        if not _contains_worker_stall_signal(normalized):
            continue

        sanitized_value = _sanitize_worker_banner_text(text_value)
        digest = _fingerprint_worker_banner(text_value)

        if sanitized_value and sanitized_value != text_value:
            metadata[key] = sanitized_value
        elif isinstance(raw_value, (bytes, bytearray, memoryview)) and text_value:
            metadata[key] = text_value

        fingerprint_key = f"{key}_fingerprint"
        if digest and not metadata.get(fingerprint_key):
            metadata[fingerprint_key] = digest

    visited: set[int] = set()
    nested_fingerprints: set[str] = set()

    for key in list(metadata.keys()):
        value = metadata[key]
        sanitized_value, digests, mutated = _scrub_nested_worker_artifacts(value, visited)
        if mutated:
            metadata[key] = sanitized_value
            if isinstance(sanitized_value, str):
                fingerprint_key = f"{key}_fingerprint"
                if digests and not metadata.get(fingerprint_key):
                    metadata[fingerprint_key] = sorted(digests)[0]
        nested_fingerprints.update(digests)

    if nested_fingerprints:
        merged = _merge_worker_banner_fingerprints(
            metadata.get("docker_worker_nested_banner_fingerprints"),
            nested_fingerprints,
        )
        if merged:
            metadata["docker_worker_nested_banner_fingerprints"] = merged


def _sanitize_worker_json_fragment(raw: str) -> tuple[str | None, bool]:
    """Sanitise Docker worker stall banners embedded inside JSON fragments.

    Docker Desktop occasionally serialises worker diagnostics as JSON before
    emitting them through stderr or auxiliary telemetry channels.  When those
    payloads are collected verbatim we need to rewrite any ``worker stalled``
    phrasing without discarding the surrounding structure so that downstream
    tooling can continue to reason about the metadata.  This helper performs a
    tolerant JSON decode, recursively rewrites any string values that resemble a
    worker stall banner, and re-serialises the structure using a stable format
    so diffs remain deterministic.
    """

    if not raw:
        return None, False

    candidate = raw.strip()
    if not candidate or candidate[0] not in "[{" or candidate[-1] not in "]}":
        return None, False

    try:
        decoded = json.loads(candidate)
    except (TypeError, ValueError):
        return None, False

    def _merge_key_collision(existing: object, replacement: object) -> object:
        """Merge two JSON values that map to the same sanitised key."""

        if existing == replacement:
            return existing

        def _as_iterable(value: object) -> list[object]:
            if isinstance(value, list):
                return list(value)
            return [value]

        combined: list[object] = []
        combined.extend(_as_iterable(existing))
        combined.extend(_as_iterable(replacement))

        deduplicated: list[object] = []
        seen: set[str] = set()

        for item in combined:
            try:
                marker = json.dumps(item, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError):
                marker = repr(item)
            if marker in seen:
                continue
            seen.add(marker)
            deduplicated.append(item)

        return deduplicated

    def _rewrite(node: object) -> tuple[object, bool]:
        if isinstance(node, str):
            normalized = _normalise_worker_stalled_phrase(node)
            if _contains_worker_stall_signal(normalized):
                return _sanitize_worker_banner_text(node), True
            return node, False
        if isinstance(node, list):
            mutated = False
            rewritten: list[object] = []
            for item in node:
                replacement, changed = _rewrite(item)
                mutated = mutated or changed
                rewritten.append(replacement)
            if mutated:
                return rewritten, True
            return node, False
        if isinstance(node, dict):
            mutated = False
            rewritten_dict: dict[str, object] = {}
            for key, value in node.items():
                sanitized_key = key
                key_mutated = False

                if isinstance(key, str):
                    replacement_key, _, key_changed = _sanitize_worker_metadata_key(key)
                    sanitized_key = (
                        str(replacement_key)
                        if not isinstance(replacement_key, str)
                        else replacement_key
                    )
                    key_mutated = key_changed

                replacement, changed = _rewrite(value)
                mutated = mutated or changed or key_mutated or sanitized_key != key

                if sanitized_key in rewritten_dict:
                    existing_value = rewritten_dict[sanitized_key]
                    merged = _merge_key_collision(existing_value, replacement)
                    if merged is not existing_value:
                        rewritten_dict[sanitized_key] = merged
                    mutated = True
                else:
                    rewritten_dict[sanitized_key] = replacement

            if mutated:
                return rewritten_dict, True
            return node, False
        return node, False

    rewritten, mutated = _rewrite(decoded)
    if not mutated:
        return None, False

    try:
        # ``sort_keys`` keeps the representation deterministic so that the same
        # payload always produces identical sanitised output irrespective of key
        # ordering in the original JSON blob.
        rendered = json.dumps(rewritten, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None, False

    return rendered, True


def _sanitize_worker_metadata_value(
    value: object, *, prefer_canonical: bool = False
) -> tuple[str | None, str | None]:
    """Return a sanitised worker metadata payload and its fingerprint.

    The helper preserves non-worker content while aggressively rewriting any
    token that contains a stall banner.  Structured sample fields may aggregate
    multiple values separated by semicolons or newlines; those fragments are
    normalised independently so we keep unrelated diagnostics intact.
    """

    text = _coerce_textual_value(value)

    if text is None:
        return None, None

    if not text:
        return None, None

    decoded = _decode_worker_base64_fragment(text)
    if decoded is not None and decoded != text:
        sanitised, digest = _sanitize_worker_metadata_value(
            decoded, prefer_canonical=prefer_canonical
        )
        if sanitised is not None:
            return sanitised, digest
        if digest:
            return None, digest

    digest = _fingerprint_worker_banner(text)

    json_sanitized, json_changed = _sanitize_worker_json_fragment(text)
    if json_changed and json_sanitized:
        return json_sanitized, digest

    separators = re.compile(r"[;\n]\s*")
    parts = [segment.strip() for segment in separators.split(text) if segment.strip()]

    if not parts:
        return None, digest

    mutated = False
    sanitized_parts: list[str] = []

    for segment in parts:
        normalized = _normalise_worker_stalled_phrase(segment)
        if _contains_worker_stall_signal(normalized):
            sanitized_parts.append(_sanitize_worker_banner_text(segment))
            mutated = True
        else:
            sanitized_parts.append(segment)

    if not mutated:
        collapsed_value = _collapse_worker_restart_sequences(text)
        if collapsed_value != text:
            return collapsed_value, digest
        return None, digest

    unique_parts: list[str] = []
    seen: set[str] = set()
    for segment in sanitized_parts:
        collapsed = re.sub(r"\s+", " ", segment).strip()
        if not collapsed:
            continue
        key = collapsed.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_parts.append(collapsed)

    joined = "; ".join(unique_parts) if unique_parts else _WORKER_STALLED_PRIMARY_NARRATIVE
    sanitised_value = _collapse_worker_restart_sequences(joined)

    if prefer_canonical and sanitised_value:
        sanitised_value = _canonicalize_worker_narrative(sanitised_value)

    return sanitised_value, digest


@dataclass
class _WorkerWarningRecord:
    """Capture restart telemetry for an individual Docker worker."""

    context: str | None
    restart_count: int | None = None
    backoff_hint: str | None = None
    backoff_seconds: float | None = None
    last_seen: str | None = None
    last_healthy: str | None = None
    last_error: str | None = None
    last_error_original: str | None = None
    last_error_raw: str | None = None
    last_error_banner: str | None = None
    last_error_banner_raw: str | None = None
    last_error_banner_preserved: str | None = None
    last_error_banner_preserved_raw: str | None = None
    last_error_banner_signature: str | None = None
    occurrences: int = 0
    restart_samples: list[int] = field(default_factory=list)
    backoff_hints: list[str] = field(default_factory=list)
    last_seen_samples: list[str] = field(default_factory=list)
    last_healthy_samples: list[str] = field(default_factory=list)
    last_error_samples: list[str] = field(default_factory=list)
    last_error_original_samples: list[str] = field(default_factory=list)
    last_error_raw_samples: list[str] = field(default_factory=list)
    last_error_banner_samples: list[str] = field(default_factory=list)
    last_error_banner_raw_samples: list[str] = field(default_factory=list)
    last_error_banner_preserved_samples: list[str] = field(default_factory=list)
    last_error_banner_preserved_raw_samples: list[str] = field(default_factory=list)
    last_error_banner_signature_samples: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)

    def update(self, metadata: Mapping[str, str]) -> None:
        """Merge ``metadata`` gleaned from a worker warning into the record."""

        self.occurrences += 1

        context = metadata.get("docker_worker_context")
        if context and not self.context:
            self.context = context.strip()

        restart_value = metadata.get("docker_worker_restart_count")
        restart_count = _coerce_optional_int(restart_value)
        if restart_count is not None:
            self.restart_samples.append(restart_count)
            if self.restart_count is None or restart_count > self.restart_count:
                self.restart_count = restart_count

        backoff_hint = metadata.get("docker_worker_backoff")
        if backoff_hint:
            normalized_hint = backoff_hint.strip()
            if normalized_hint:
                self.backoff_hints.append(normalized_hint)
                candidate_seconds = _estimate_backoff_seconds(normalized_hint)
                if candidate_seconds is not None:
                    if (
                        self.backoff_seconds is None
                        or candidate_seconds > self.backoff_seconds
                        or (
                            candidate_seconds == self.backoff_seconds
                            and not self.backoff_hint
                        )
                    ):
                        self.backoff_seconds = candidate_seconds
                        self.backoff_hint = normalized_hint
                elif not self.backoff_hint:
                    self.backoff_hint = normalized_hint

        last_restart = metadata.get("docker_worker_last_restart")
        if last_restart:
            cleaned_restart = last_restart.strip()
            if cleaned_restart:
                self.last_seen_samples.append(cleaned_restart)
                self.last_seen = cleaned_restart

        healthy_marker = metadata.get("docker_worker_last_healthy")
        if healthy_marker:
            cleaned_healthy = healthy_marker.strip()
            if cleaned_healthy:
                self.last_healthy_samples.append(cleaned_healthy)
                self.last_healthy = cleaned_healthy

        last_error = metadata.get("docker_worker_last_error")
        if last_error:
            cleaned_error = last_error.strip()
            if cleaned_error:
                self.last_error_samples.append(cleaned_error)
                self.last_error = cleaned_error

        original_error = metadata.get("docker_worker_last_error_original")
        if original_error:
            cleaned_original = original_error.strip()
            if cleaned_original:
                self.last_error_original_samples.append(cleaned_original)
                self.last_error_original = cleaned_original

        raw_error = metadata.get("docker_worker_last_error_raw")
        if raw_error:
            cleaned_raw = raw_error.strip()
            if cleaned_raw:
                self.last_error_raw_samples.append(cleaned_raw)
                self.last_error_raw = cleaned_raw

        banner = metadata.get("docker_worker_last_error_banner")
        if banner:
            cleaned_banner = banner.strip()
            if cleaned_banner:
                self.last_error_banner_samples.append(cleaned_banner)
                self.last_error_banner = cleaned_banner

        banner_raw = metadata.get("docker_worker_last_error_banner_raw")
        if banner_raw:
            cleaned_banner_raw = banner_raw.strip()
            if cleaned_banner_raw:
                self.last_error_banner_raw_samples.append(cleaned_banner_raw)
                self.last_error_banner_raw = cleaned_banner_raw

        banner_preserved = metadata.get("docker_worker_last_error_banner_preserved")
        if banner_preserved:
            cleaned_banner_preserved = banner_preserved.strip()
            if cleaned_banner_preserved:
                self.last_error_banner_preserved_samples.append(
                    cleaned_banner_preserved
                )
                self.last_error_banner_preserved = cleaned_banner_preserved

        banner_preserved_raw = metadata.get(
            "docker_worker_last_error_banner_preserved_raw"
        )
        if banner_preserved_raw:
            cleaned_banner_preserved_raw = banner_preserved_raw.strip()
            if cleaned_banner_preserved_raw:
                self.last_error_banner_preserved_raw_samples.append(
                    cleaned_banner_preserved_raw
                )
                self.last_error_banner_preserved_raw = cleaned_banner_preserved_raw

        banner_signature = metadata.get("docker_worker_last_error_banner_signature")
        if banner_signature:
            cleaned_signature = banner_signature.strip()
            if cleaned_signature:
                self.last_error_banner_signature_samples.append(cleaned_signature)
                if not self.last_error_banner_signature:
                    self.last_error_banner_signature = cleaned_signature

        error_code = metadata.get("docker_worker_last_error_code")
        if error_code:
            normalized_code = error_code.strip()
            if normalized_code:
                self.error_codes.append(normalized_code)


class _WorkerWarningAggregator:
    """Accumulate worker restart telemetry across multiple Docker warnings."""

    def __init__(self) -> None:
        self._records: dict[str, _WorkerWarningRecord] = {}
        self._order: list[str] = []
        self._health: str | None = None

    def ingest(self, metadata: Mapping[str, str]) -> None:
        """Record ``metadata`` emitted by ``_normalise_docker_warning``."""

        if not metadata:
            return

        health = metadata.get("docker_worker_health")
        if health and not self._health:
            self._health = health

        context = metadata.get("docker_worker_context")
        normalized_context = context.strip() if isinstance(context, str) else None
        key = normalized_context.casefold() if normalized_context else "__anonymous"

        record = self._records.get(key)
        if record is None:
            record = _WorkerWarningRecord(context=normalized_context)
            self._records[key] = record
            self._order.append(key)
        else:
            if normalized_context and not record.context:
                record.context = normalized_context

        record.update(metadata)

    def finalize(self) -> dict[str, str]:
        """Produce a consolidated metadata mapping for downstream diagnostics."""

        result: dict[str, str] = {}
        if self._health:
            result["docker_worker_health"] = self._health

        records = [self._records[key] for key in self._order if self._records[key]]
        if not records:
            return result

        primary = self._select_primary_record(records)

        total_occurrences = sum(record.occurrences for record in records)
        if total_occurrences:
            result["docker_worker_warning_occurrences"] = str(total_occurrences)

        context_occurrence_entries = [
            f"{record.context}:{record.occurrences}"
            for record in records
            if record.context and record.occurrences
        ]
        if context_occurrence_entries:
            result["docker_worker_context_occurrences"] = ", ".join(
                context_occurrence_entries
            )

        contexts = _coalesce_iterable(
            [record.context for record in records if record.context]
        )
        if primary and primary.context:
            result["docker_worker_context"] = primary.context
        elif contexts:
            result["docker_worker_context"] = contexts[0]
        if len(contexts) > 1:
            result["docker_worker_contexts"] = ", ".join(contexts)

        restart_samples = sorted(
            {
                sample
                for record in records
                for sample in record.restart_samples
                if sample is not None
            }
        )
        if primary and primary.restart_count is not None:
            result["docker_worker_restart_count"] = str(primary.restart_count)
        elif restart_samples:
            result["docker_worker_restart_count"] = str(restart_samples[-1])
        if len(restart_samples) > 1:
            result["docker_worker_restart_count_samples"] = ", ".join(
                str(sample) for sample in restart_samples
            )

        backoff_hints = _coalesce_iterable(
            [hint for record in records for hint in record.backoff_hints]
        )
        if primary and primary.backoff_hint:
            result["docker_worker_backoff"] = primary.backoff_hint
        elif backoff_hints:
            backoff_candidates = [
                (hint, _estimate_backoff_seconds(hint)) for hint in backoff_hints
            ]
            chosen_hint, _ = max(
                backoff_candidates,
                key=lambda item: (
                    item[1] is not None,
                    item[1] or 0.0,
                    len(item[0]),
                ),
            )
            result["docker_worker_backoff"] = chosen_hint
        if len(backoff_hints) > 1:
            result["docker_worker_backoff_options"] = ", ".join(backoff_hints)

        last_restart_markers = _coalesce_iterable(
            [marker for record in records for marker in record.last_seen_samples]
        )
        if primary and primary.last_seen:
            result["docker_worker_last_restart"] = primary.last_seen
        elif last_restart_markers:
            result["docker_worker_last_restart"] = last_restart_markers[-1]
        if len(last_restart_markers) > 1:
            result["docker_worker_last_restart_samples"] = ", ".join(
                last_restart_markers
            )

        healthy_markers = _coalesce_iterable(
            [marker for record in records for marker in record.last_healthy_samples]
        )
        if primary and primary.last_healthy:
            result["docker_worker_last_healthy"] = primary.last_healthy
        elif healthy_markers:
            result["docker_worker_last_healthy"] = healthy_markers[-1]
        if len(healthy_markers) > 1:
            result["docker_worker_last_healthy_samples"] = "; ".join(
                healthy_markers
            )

        last_errors = _coalesce_iterable(
            [error for record in records for error in record.last_error_samples]
        )
        if primary and primary.last_error:
            result["docker_worker_last_error"] = primary.last_error
        elif last_errors:
            result["docker_worker_last_error"] = last_errors[-1]
        if len(last_errors) > 1:
            result["docker_worker_last_error_samples"] = "; ".join(last_errors)

        original_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_original_samples
            ]
        )
        if primary and primary.last_error_original:
            result["docker_worker_last_error_original"] = primary.last_error_original
        elif original_errors:
            result["docker_worker_last_error_original"] = original_errors[-1]
        if len(original_errors) > 1:
            result["docker_worker_last_error_original_samples"] = "; ".join(
                original_errors
            )

        raw_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_raw_samples
            ]
        )
        if primary and primary.last_error_raw:
            result["docker_worker_last_error_raw"] = primary.last_error_raw
        elif raw_errors:
            result["docker_worker_last_error_raw"] = raw_errors[-1]
        if len(raw_errors) > 1:
            result["docker_worker_last_error_raw_samples"] = "; ".join(raw_errors)

        banner_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_banner_samples
            ]
        )
        if primary and primary.last_error_banner:
            result["docker_worker_last_error_banner"] = primary.last_error_banner
        elif banner_errors:
            result["docker_worker_last_error_banner"] = banner_errors[-1]
        if len(banner_errors) > 1:
            result["docker_worker_last_error_banner_samples"] = "; ".join(
                banner_errors
            )

        raw_banner_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_banner_raw_samples
            ]
        )
        if primary and primary.last_error_banner_raw:
            result["docker_worker_last_error_banner_raw"] = (
                primary.last_error_banner_raw
            )
        elif raw_banner_errors:
            result["docker_worker_last_error_banner_raw"] = raw_banner_errors[-1]
        if len(raw_banner_errors) > 1:
            result["docker_worker_last_error_banner_raw_samples"] = "; ".join(
                raw_banner_errors
            )

        preserved_banner_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_banner_preserved_samples
            ]
        )
        if primary and primary.last_error_banner_preserved:
            result["docker_worker_last_error_banner_preserved"] = (
                primary.last_error_banner_preserved
            )
        elif preserved_banner_errors:
            result["docker_worker_last_error_banner_preserved"] = (
                preserved_banner_errors[-1]
            )
        if len(preserved_banner_errors) > 1:
            result["docker_worker_last_error_banner_preserved_samples"] = "; ".join(
                preserved_banner_errors
            )

        preserved_raw_banner_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_banner_preserved_raw_samples
            ]
        )
        if primary and primary.last_error_banner_preserved_raw:
            result["docker_worker_last_error_banner_preserved_raw"] = (
                primary.last_error_banner_preserved_raw
            )
        elif preserved_raw_banner_errors:
            result["docker_worker_last_error_banner_preserved_raw"] = (
                preserved_raw_banner_errors[-1]
            )
        if len(preserved_raw_banner_errors) > 1:
            result[
                "docker_worker_last_error_banner_preserved_raw_samples"
            ] = "; ".join(preserved_raw_banner_errors)

        signature_errors = _coalesce_iterable(
            [
                signature
                for record in records
                for signature in record.last_error_banner_signature_samples
            ]
        )
        if primary and primary.last_error_banner_signature:
            result["docker_worker_last_error_banner_signature"] = (
                primary.last_error_banner_signature
            )
        elif signature_errors:
            result["docker_worker_last_error_banner_signature"] = signature_errors[-1]
        if len(signature_errors) > 1:
            result["docker_worker_last_error_banner_signature_samples"] = "; ".join(
                signature_errors
            )

        error_codes = _coalesce_iterable(
            [code for record in records for code in record.error_codes]
        )
        if error_codes:
            result["docker_worker_last_error_code"] = error_codes[0]
        if len(error_codes) > 1:
            result["docker_worker_last_error_codes"] = ", ".join(error_codes)

        # Enrich the aggregate with a synthesised health narrative so callers can
        # consistently reason about severity without re-implementing the
        # classification pipeline that ``_normalise_docker_warning`` uses for
        # single warnings.  ``_compose_worker_flapping_guidance`` mutates the
        # mapping in-place and returns the rendered summary which ensures follow
        # up consumers receive the same canonical guidance for both individual
        # and aggregated worker diagnostics.  Defensive guards prevent the
        # aggregation step from ever masking the original metadata when the
        # classifier cannot derive additional context.
        if result.get("docker_worker_health"):
            try:
                summary = _compose_worker_flapping_guidance(result)
            except Exception as exc:  # pragma: no cover - defensive safety net
                LOGGER.debug(
                    "Failed to synthesise aggregated worker guidance: %s",
                    exc,
                    exc_info=True,
                )
            else:
                if summary and "docker_worker_health_summary" not in result:
                    result["docker_worker_health_summary"] = summary

        return result

    @staticmethod
    def _select_primary_record(
        records: Sequence[_WorkerWarningRecord],
    ) -> _WorkerWarningRecord | None:
        if not records:
            return None

        return max(
            records,
            key=lambda record: (
                record.restart_count is not None,
                record.restart_count or 0,
                record.backoff_seconds is not None,
                record.backoff_seconds or 0.0,
                1 if record.last_error else 0,
                len(record.context or ""),
            ),
        )


def _normalize_warning_collection(messages: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Normalise warning ``messages`` and capture associated metadata."""

    normalized_entries: list[tuple[str, str | None]] = []
    metadata: dict[str, str] = {}
    seen: set[str] = set()

    worker_aggregator = _WorkerWarningAggregator()

    for message in messages:
        cleaned, extracted = _normalise_docker_warning(message)
        if extracted:
            worker_aggregator.ingest(extracted)
            for key, value in extracted.items():
                if key.startswith("docker_worker_"):
                    continue
                metadata[key] = value
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        severity = None
        if extracted:
            severity = extracted.get("docker_worker_health_severity")
        normalized_entries.append((cleaned, severity))

    worker_metadata = worker_aggregator.finalize()
    metadata.update(worker_metadata)

    normalized = [message for message, _ in normalized_entries]

    summary = worker_metadata.get("docker_worker_health_summary") if worker_metadata else None
    if isinstance(summary, str):
        summary = summary.strip()
    else:
        summary = None

    if summary:
        filtered_messages: list[str] = []
        for message, severity in normalized_entries:
            if _looks_like_worker_guidance(message):
                normalized_severity = (severity or "").strip().lower()
                if normalized_severity != "info":
                    continue
            filtered_messages.append(message)
        filtered_messages.insert(0, summary)
        normalized = _coalesce_iterable(filtered_messages)
    else:
        normalized = _coalesce_iterable(normalized)

    return normalized, metadata


def _looks_like_worker_guidance(message: str) -> bool:
    """Return ``True`` when *message* resembles synthetic worker guidance."""

    if not message:
        return False

    collapsed = re.sub(r"\s+", " ", message).strip().lower()
    if not collapsed:
        return False

    if "docker desktop" not in collapsed:
        return False
    if "worker" not in collapsed:
        return False

    if not any(token in collapsed for token in ("restart", "restarted", "restarting", "stall", "stalled", "flapping")):
        return False

    return True


def _reclassify_worker_warnings_for_info(
    warnings: Iterable[str], metadata: Mapping[str, str]
) -> tuple[list[str], list[str]]:
    """Move mild worker stall warnings into informational diagnostics."""

    if metadata.get("docker_worker_health") != "flapping":
        return [], list(warnings)

    severity = metadata.get("docker_worker_health_severity")
    if severity != "info":
        return [], list(warnings)

    summary = metadata.get("docker_worker_health_summary")

    info_messages: list[str] = []
    remaining: list[str] = []

    worker_messages: list[str] = []
    for warning in warnings:
        if _looks_like_worker_guidance(warning):
            worker_messages.append(warning)
        else:
            remaining.append(warning)

    if worker_messages:
        if summary:
            info_messages.append(summary)
        else:
            info_messages.extend(worker_messages)

    return info_messages, remaining


def _reclassify_worker_guidance_messages(
    *,
    warnings: MutableSequence[str],
    errors: MutableSequence[str],
    infos: MutableSequence[str],
    metadata: MutableMapping[str, str],
) -> None:
    """Promote worker guidance derived from errors or notices into warnings.

    Docker Desktop occasionally surfaces the ``worker stalled`` banner through
    stderr (which we classify as an error) or as part of general notices.
    After normalisation those payloads are rendered as high level remediation
    messages.  Presenting the same banner across multiple severities dilutes its
    usefulness and makes it harder for Windows developers to distinguish
    legitimate Docker failures from recoverable worker churn.  This helper moves
    any synthetic worker guidance into the warnings channel while retaining
    non-worker diagnostics in their original severity buckets.
    """

    def _extract(collection: MutableSequence[str]) -> tuple[list[str], list[str]]:
        retained: list[str] = []
        reclassified: list[str] = []
        for message in list(collection):
            if _looks_like_worker_guidance(message):
                reclassified.append(message)
            else:
                retained.append(message)
        return retained, reclassified

    retained_errors, promoted_from_errors = _extract(errors)
    retained_infos, promoted_from_infos = _extract(infos)

    if promoted_from_errors or promoted_from_infos:
        # Extend warnings in-place to preserve any later callers that keep the
        # list reference.  Normalisation via ``_coalesce_iterable`` occurs after
        # this helper executes so ordering remains deterministic while removing
        # duplicates.
        warnings.extend(promoted_from_errors)
        warnings.extend(promoted_from_infos)

        if promoted_from_errors:
            metadata.setdefault(
                "docker_worker_guidance_promoted_from_errors",
                str(len(promoted_from_errors)),
            )
        if promoted_from_infos:
            metadata.setdefault(
                "docker_worker_guidance_promoted_from_infos",
                str(len(promoted_from_infos)),
            )

    errors[:] = retained_errors
    infos[:] = retained_infos

def _parse_key_value_lines(payload: str) -> dict[str, str]:
    """Return key/value mappings parsed from ``payload`` lines."""

    parsed: dict[str, str] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").lower()
        parsed[normalized_key] = value.strip()
    return parsed


@dataclass(frozen=True)
class _WindowsServiceSpec:
    """Describe a Windows service that Docker Desktop relies on."""

    name: str
    friendly_name: str
    severity: Literal["error", "warning"]


_WINDOWS_VIRTUALIZATION_SERVICES: tuple[_WindowsServiceSpec, ...] = (
    _WindowsServiceSpec(
        name="vmcompute",
        friendly_name="Hyper-V Host Compute Service",
        severity="error",
    ),
    _WindowsServiceSpec(
        name="hns",
        friendly_name="Windows Host Network Service",
        severity="error",
    ),
    _WindowsServiceSpec(
        name="LxssManager",
        friendly_name="WSL Session Manager Service",
        severity="error",
    ),
    _WindowsServiceSpec(
        name="vmms",
        friendly_name="Hyper-V Virtual Machine Management Service",
        severity="warning",
    ),
    _WindowsServiceSpec(
        name="com.docker.service",
        friendly_name="Docker Desktop Service",
        severity="error",
    ),
)


def _build_powershell_encoded_command(script: str) -> list[str]:
    """Return a PowerShell invocation that executes *script* via ``-EncodedCommand``."""

    if not script:
        raise ValueError("PowerShell script payload must not be empty")

    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        encoded,
    ]


def _parse_windows_service_payload(payload: str) -> list[dict[str, str]]:
    """Return structured service metadata parsed from PowerShell JSON."""

    if not payload:
        return []

    sanitized = _strip_control_sequences(payload).strip()
    if not sanitized:
        return []

    try:
        decoded = json.loads(sanitized)
    except json.JSONDecodeError:
        LOGGER.debug("Failed to decode Windows service probe payload: %s", sanitized)
        return []

    entries: list[dict[str, str]] = []
    if isinstance(decoded, MappingABC):
        decoded = [decoded]

    if not isinstance(decoded, IterableABC):
        return []

    for item in decoded:
        if not isinstance(item, MappingABC):
            continue
        name = str(
            item.get("Name")
            or item.get("name")
            or item.get("ServiceName")
            or ""
        ).strip()
        status = str(item.get("Status") or item.get("status") or "").strip()
        start_type = str(
            item.get("StartType")
            or item.get("startType")
            or item.get("start_type")
            or ""
        ).strip()
        if not name:
            continue
        entries.append({
            "name": name,
            "status": status,
            "start_type": start_type,
        })

    return entries


def _collect_windows_service_health(
    timeout: float,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Inspect critical Windows services that influence Docker Desktop stability."""

    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, str] = {}

    service_literals = ", ".join(f"'{spec.name}'" for spec in _WINDOWS_VIRTUALIZATION_SERVICES)
    script = f"""
Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
$serviceNames = @({service_literals})
$results = foreach ($serviceName in $serviceNames) {{
    $service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($null -eq $service) {{
        [PSCustomObject]@{{ Name = $serviceName; Status = 'Missing'; StartType = 'Unknown' }}
    }} else {{
        $startType = try {{ $service.StartType.ToString() }} catch {{ 'Unknown' }}
        [PSCustomObject]@{{
            Name = $service.Name
            Status = $service.Status.ToString()
            StartType = $startType
        }}
    }}
}}
$results | ConvertTo-Json -Compress
""".strip()

    command = _build_powershell_encoded_command(script)
    proc, failure = _run_command(command, timeout=timeout)

    if failure:
        warnings.append(f"Unable to inspect Windows service health: {failure}")
        return warnings, errors, metadata

    if proc is None:
        return warnings, errors, metadata

    if proc.stderr:
        stderr_message = proc.stderr.strip()
        if stderr_message:
            warnings.append(
                "PowerShell reported issues while probing Docker Desktop services: %s"
                % stderr_message
            )

    if proc.returncode not in {0, None}:
        warnings.append(
            "PowerShell exited with code %s while probing Docker Desktop services"
            % proc.returncode
        )

    if proc.stdout:
        metadata["windows_service_probe_raw"] = _strip_control_sequences(proc.stdout).strip()

    parsed_entries = _parse_windows_service_payload(proc.stdout or "")
    if not parsed_entries:
        warnings.append(
            "Unable to parse Windows service status output; run 'Get-Service' manually to verify Docker dependencies."
        )
        return warnings, errors, metadata

    seen_services: dict[str, dict[str, str]] = {}
    for entry in parsed_entries:
        name = entry.get("name", "")
        if not name:
            continue
        normalized_key = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        status = entry.get("status", "")
        start_type = entry.get("start_type", "")
        metadata[f"windows_service_{normalized_key}_status"] = status
        metadata[f"windows_service_{normalized_key}_start_type"] = start_type
        if normalized_key == "vmcompute" and status:
            metadata.setdefault("vmcompute_status", status)
        seen_services[name.lower()] = entry

    message_registry: set[str] = set()

    def _append_message(collection: list[str], message: str) -> None:
        normalized = (message or "").strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in message_registry:
            return
        message_registry.add(key)
        collection.append(normalized)

    for spec in _WINDOWS_VIRTUALIZATION_SERVICES:
        entry = seen_services.get(spec.name.lower())
        if entry is None:
            message = (
                f"{spec.friendly_name} ({spec.name}) is not installed. "
                "Enable the component via Windows Features and restart Docker Desktop."
            )
            target = errors if spec.severity == "error" else warnings
            _append_message(target, message)
            continue

        status = entry.get("status", "")
        start_type = entry.get("start_type", "")
        status_lower = status.strip().lower()
        start_lower = start_type.strip().lower()

        unhealthy = status_lower not in {"running", "startpending", "starting"}
        disabled = start_lower == "disabled" or (
            spec.name.lower() == "com.docker.service" and start_lower == "manual"
        )

        if unhealthy or disabled:
            message_parts = [
                f"{spec.friendly_name} ({spec.name}) is {status or 'unavailable'}.",
            ]
            if start_type:
                message_parts.append(f"Start type: {start_type}.")
            message_parts.append(
                "Restart the service from an elevated PowerShell session using 'Start-Service %s' "
                "and reboot if the issue persists." % spec.name
            )
            message = " ".join(message_parts)
            target = errors if spec.severity == "error" else warnings
            _append_message(target, message)

    return warnings, errors, metadata


def _parse_bcdedit_configuration(payload: str) -> dict[str, str]:
    """Return boot configuration entries extracted from ``bcdedit`` output."""

    entries: dict[str, str] = {}

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or set(line) <= {"-"}:
            continue
        match = re.match(r"^(?P<key>[A-Za-z0-9._-]+)\s+(?P<value>.+)$", line)
        if not match:
            continue
        key = match.group("key").strip().lower()
        value = match.group("value").strip()
        if not key or not value:
            continue
        entries[key] = value

    return entries


def _parse_wsl_distribution_table(payload: str) -> list[dict[str, str | bool]]:
    """Return WSL distribution metadata parsed from ``wsl.exe -l -v`` output."""

    distributions: list[dict[str, str | bool]] = []

    header_consumed = False
    for raw_line in payload.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not header_consumed:
            header_consumed = True
            columns = [column.upper() for column in re.split(r"\s{2,}", line.strip())[:3]]
            if columns == ["NAME", "STATE", "VERSION"]:
                continue
        is_default = line.lstrip().startswith("*")
        normalized = line.lstrip("*").strip()
        if not normalized:
            continue
        parts = re.split(r"\s{2,}", normalized)
        if len(parts) < 3:
            collapsed = re.sub(r"\s+", " ", normalized)
            parts = collapsed.split(" ")
            if len(parts) < 3:
                continue
            name = " ".join(parts[:-2])
            state, version = parts[-2:]
        else:
            name, state, version = parts[0], parts[1], parts[2]

        distributions.append(
            {
                "name": name.strip(),
                "state": state.strip(),
                "version": version.strip(),
                "is_default": is_default,
            }
        )

    return distributions


def _collect_windows_virtualization_insights(timeout: float = 6.0) -> tuple[list[str], list[str], dict[str, str]]:
    """Gather virtualization diagnostics relevant to Docker Desktop on Windows."""

    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, str] = {}

    status_proc, failure = _run_command(["wsl.exe", "--status"], timeout=timeout)
    if failure:
        warnings.append(f"Unable to query WSL status: {failure}")
    elif status_proc is not None:
        if status_proc.stdout.strip():
            metadata["wsl_status_raw"] = status_proc.stdout.strip()
        parsed = _parse_key_value_lines(status_proc.stdout)
        default_version = parsed.get("default_version")
        if default_version:
            metadata["wsl_default_version"] = default_version
            if default_version and not default_version.startswith("2"):
                errors.append(
                    "WSL default version is set to %s. Docker Desktop requires WSL 2 for stable operation. "
                    "Run 'wsl --set-default-version 2' from an elevated PowerShell session and reboot."
                    % default_version
                )
        wsl_version = parsed.get("wsl_version")
        if wsl_version:
            metadata["wsl_version"] = wsl_version
        if not parsed and status_proc.stdout:
            lower = status_proc.stdout.lower()
            if "not installed" in lower or "not enabled" in lower:
                errors.append(
                    "Windows Subsystem for Linux is not fully enabled. Enable the 'Virtual Machine Platform' and 'Windows Subsystem for Linux' optional features and restart."
                )

    list_proc, failure = _run_command(["wsl.exe", "-l", "-v"], timeout=timeout)
    if failure:
        warnings.append(f"Unable to enumerate WSL distributions: {failure}")
    elif list_proc is not None and list_proc.stdout:
        metadata["wsl_list_raw"] = list_proc.stdout.strip()
        distributions = _parse_wsl_distribution_table(list_proc.stdout)
        default_distro: str | None = None
        docker_states: dict[str, str] = {}
        for item in distributions:
            name = str(item.get("name", "")).strip()
            state = str(item.get("state", "")).strip()
            version = str(item.get("version", "")).strip()
            if not name:
                continue
            key_prefix = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
            if key_prefix:
                if state:
                    metadata[f"wsl_distro_{key_prefix}_state"] = state
                if version:
                    metadata[f"wsl_distro_{key_prefix}_version"] = version
            if bool(item.get("is_default")):
                metadata["wsl_default_distribution"] = name
                default_distro = name
                if state.lower() not in {"running", "starting"}:
                    warnings.append(
                        "Default WSL distribution '%s' is %s. Start the distribution or switch Docker Desktop to a running distribution via Settings > Resources > WSL Integration."
                        % (name, state or "stopped")
                    )
            normalized_name = name.lower()
            if normalized_name in {"docker-desktop", "docker-desktop-data"}:
                docker_states[normalized_name] = state
                if state.lower() not in {"running", "starting"}:
                    errors.append(
                        "WSL distribution '%s' is %s. Start Docker Desktop and ensure its WSL integration is healthy."
                        % (name, state or "stopped")
                    )
        if not distributions:
            warnings.append(
                "WSL reported no installed distributions; Docker Desktop cannot operate without the 'docker-desktop' distribution."
            )
        elif {"docker-desktop", "docker-desktop-data"} - set(docker_states):
            missing = sorted({"docker-desktop", "docker-desktop-data"} - set(docker_states))
            errors.append(
                "Required Docker Desktop WSL distributions are missing: %s. Re-run 'wsl --install' or reinstall Docker Desktop."
                % ", ".join(missing)
            )
        if default_distro is None and distributions:
            warnings.append(
                "No default WSL distribution detected. Assign one with 'wsl --set-default <distro>' to avoid Docker context issues."
            )

    hyperv_cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "(Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All).State",
    ]
    hyperv_proc, failure = _run_command(hyperv_cmd, timeout=timeout)
    if failure:
        warnings.append(f"Unable to inspect Hyper-V feature state: {failure}")
    elif hyperv_proc is not None:
        lines = [line.strip() for line in hyperv_proc.stdout.splitlines() if line.strip()]
        hyperv_state = lines[-1] if lines else ""
        if hyperv_state:
            metadata["hyper_v_state"] = hyperv_state
            if hyperv_state.lower() not in {"enabled", "enablepending"}:
                errors.append(
                    "Hyper-V is %s. Enable Hyper-V (and its management tools) from Windows Features, reboot, and relaunch Docker Desktop."
                    % hyperv_state
                )

    vmp_cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "(Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State",
    ]
    vmp_proc, failure = _run_command(vmp_cmd, timeout=timeout)
    if failure:
        warnings.append(f"Unable to inspect Virtual Machine Platform state: {failure}")
    elif vmp_proc is not None:
        lines = [line.strip() for line in vmp_proc.stdout.splitlines() if line.strip()]
        vmp_state = lines[-1] if lines else ""
        if vmp_state:
            metadata["virtual_machine_platform_state"] = vmp_state
            if vmp_state.lower() not in {"enabled", "enablepending"}:
                errors.append(
                    "Windows 'Virtual Machine Platform' feature is %s. Enable it via 'OptionalFeatures.exe', reboot, and restart Docker Desktop."
                    % vmp_state
                )

    service_warnings, service_errors, service_metadata = _collect_windows_service_health(
        timeout
    )
    if service_warnings:
        warnings.extend(service_warnings)
    if service_errors:
        errors.extend(service_errors)
    if service_metadata:
        metadata.update(service_metadata)

    bcdedit_cmd = ["bcdedit.exe", "/enum", "{current}"]
    bcdedit_proc, failure = _run_command(bcdedit_cmd, timeout=timeout)
    if failure:
        warnings.append(f"Unable to inspect boot configuration: {failure}")
    elif bcdedit_proc is not None:
        entries = _parse_bcdedit_configuration(bcdedit_proc.stdout)
        hypervisor_mode = entries.get("hypervisorlaunchtype")
        if hypervisor_mode:
            metadata["hypervisor_launch_type"] = hypervisor_mode
            normalized_mode = hypervisor_mode.strip().lower()
            if normalized_mode not in {"auto", "automatic"}:
                errors.append(
                    "Boot configuration sets hypervisorlaunchtype to %s. Enable the hypervisor by running 'bcdedit /set hypervisorlaunchtype auto' from an elevated PowerShell session and reboot."
                    % hypervisor_mode
                )

    return warnings, errors, metadata


def _extract_worker_error_codes_from_metadata(
    metadata: Mapping[str, str]
) -> set[str]:
    """Return the set of worker error codes recorded in ``metadata``."""

    codes: set[str] = set()

    primary = metadata.get("docker_worker_last_error_code")
    if isinstance(primary, str):
        normalized = primary.strip().upper()
        if normalized:
            codes.add(normalized)

    for token in _split_metadata_values(metadata.get("docker_worker_last_error_codes")):
        normalized = token.strip().upper()
        if normalized:
            codes.add(normalized)

    return codes


def _is_virtualization_error_code(code: str) -> bool:
    """Return ``True`` when *code* signals a virtualization issue on Windows."""

    if not code:
        return False

    normalized = code.strip().upper()
    if not normalized:
        return False

    if normalized in _VIRTUALIZATION_ERROR_CODE_EXACT_MATCHES:
        return True

    return normalized.startswith(_VIRTUALIZATION_ERROR_CODE_PREFIXES)


def _should_collect_windows_virtualization_followups(
    metadata: Mapping[str, str], context: RuntimeContext
) -> bool:
    """Return ``True`` when Docker diagnostics warrant virtualization checks."""

    if not (context.is_windows or context.is_wsl):
        return False

    if any(
        key in metadata
        for key in ("wsl_status_raw", "hyper_v_state", "virtual_machine_platform_state")
    ):
        # Virtualization telemetry has already been collected; avoid duplicate work.
        return False

    severity = metadata.get("docker_worker_health_severity", "").strip().lower()

    error_codes = _extract_worker_error_codes_from_metadata(metadata)
    virtualization_codes = {
        code for code in error_codes if _is_virtualization_error_code(code)
    }
    if virtualization_codes:
        return True

    # Only fall back to textual heuristics when diagnostics consider the worker
    # unstable.  This avoids running expensive Windows commands for benign,
    # transient worker churn that already recovered.
    if severity and severity not in {"warning", "error"}:
        return False

    textual_hints: list[str] = []
    for key in (
        "docker_worker_last_error",
        "docker_worker_last_error_original",
        "docker_worker_last_error_raw",
        "docker_worker_health_summary",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            textual_hints.append(value)

    if not textual_hints:
        return False

    combined = " ".join(textual_hints).lower()
    virtualization_tokens = (
        "wsl",
        "hyper-v",
        "hyperv",
        "virtualization",
        "vmcompute",
        "hypervisor",
    )

    return any(token in combined for token in virtualization_tokens)


def _coerce_optional_int(value: object) -> int | None:
    """Convert *value* to ``int`` when possible."""

    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def _estimate_backoff_seconds(value: str | None) -> float | None:
    """Approximate the restart backoff interval extracted from Docker warnings."""

    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None

    candidate = candidate.strip(";.,:)")
    candidate = candidate.strip("()[]{}")
    candidate = candidate.strip()
    if not candidate:
        return None

    prefix_match = _APPROX_PREFIX_PATTERN.match(candidate)
    if prefix_match:
        candidate = candidate[prefix_match.end() :].lstrip()
    suffix_match = _APPROX_SUFFIX_PATTERN.search(candidate)
    if suffix_match and suffix_match.end() == len(candidate):
        candidate = candidate[: suffix_match.start()].rstrip()

    if not candidate:
        return None

    condensed = candidate.replace(" ", "")
    go_components = list(_GO_DURATION_COMPONENT_PATTERN.finditer(condensed))
    if go_components:
        reconstructed = "".join(match.group(0) for match in go_components)
        if reconstructed.lower() == condensed.lower():
            total = 0.0
            for match in go_components:
                try:
                    numeric = float(match.group("value"))
                except (TypeError, ValueError):
                    return None
                unit = match.group("unit").lower()
                if unit == "h":
                    total += numeric * 3600.0
                elif unit == "m":
                    total += numeric * 60.0
                elif unit == "s":
                    total += numeric
                else:  # pragma: no cover - defensive
                    return None
            return total

    clock_candidate = _interpret_clock_duration(candidate)
    if clock_candidate:
        _, seconds = clock_candidate
        return seconds

    match = _BACKOFF_INTERVAL_PATTERN.search(candidate)
    if not match:
        return None
    raw = match.group("number")
    unit = (match.group("unit") or "s").lower()
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None

    if unit in {"ms", "msec", "milliseconds"}:
        return numeric / 1000.0
    if unit in {"s", "sec", "secs", "seconds"}:
        return numeric
    if unit in {"m", "min", "mins", "minutes"}:
        return numeric * 60.0
    if unit in {"h", "hr", "hrs", "hours"}:
        return numeric * 3600.0
    return None


def _render_backoff_seconds(seconds: float) -> str:
    """Convert a numeric duration in seconds into a compact textual hint."""

    if seconds <= 0:
        return "0s"

    remaining = float(seconds)
    parts: list[str] = []

    for label, factor in (("d", 86400.0), ("h", 3600.0), ("m", 60.0)):
        if remaining >= factor - 1e-9:
            units = int(remaining // factor)
            if units:
                parts.append(f"{units}{label}")
                remaining -= units * factor

    if remaining > 0:
        if remaining < 1.0 and not parts:
            millis = int(round(remaining * 1000.0))
            if millis:
                parts.append(f"{millis}ms")
            else:
                parts.append("0s")
        else:
            if abs(remaining - round(remaining)) < 1e-6:
                parts.append(f"{int(round(remaining))}s")
            else:
                precise = ("%0.3f" % remaining).rstrip("0").rstrip(".")
                parts.append(f"{precise}s")
    elif not parts:
        parts.append("0s")

    return " ".join(parts)


def _derive_numeric_backoff_hint(key: str, value: str) -> str | None:
    """Derive a human-readable backoff hint from numeric telemetry fields."""

    candidate = value.strip()
    if not candidate:
        return None

    normalized = candidate.replace(",", "")
    try:
        numeric = float(normalized)
    except ValueError:
        return None

    key_lower = key.lower()

    def _matches(tokens: Iterable[str]) -> bool:
        return any(token in key_lower for token in tokens)

    seconds: float
    if _matches({"millisecond", "_ms", "ms"}):
        seconds = numeric / 1000.0
    elif _matches({"second", "_sec", "_s"}):
        seconds = numeric
    elif _matches({"minute", "_min", "_m"}):
        seconds = numeric * 60.0
    elif _matches({"hour", "_hr", "_h"}):
        seconds = numeric * 3600.0
    else:
        return None

    return _render_backoff_seconds(seconds)


@dataclass(frozen=True)
class WorkerRestartTelemetry:
    """Structured representation of Docker worker health metadata."""

    context: str | None
    restart_count: int | None
    backoff_hint: str | None
    last_seen: str | None
    last_healthy: str | None
    last_error: str | None
    last_error_original: str | None = None
    last_error_raw: str | None = None
    last_error_banner: str | None = None
    last_error_banner_raw: str | None = None
    last_error_banner_preserved: str | None = None
    last_error_banner_preserved_raw: str | None = None
    last_error_banner_signature: str | None = None
    warning_occurrences: int = 0
    context_occurrences: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    contexts: tuple[str, ...] = field(default_factory=tuple)
    restart_samples: tuple[int, ...] = field(default_factory=tuple)
    backoff_options: tuple[str, ...] = field(default_factory=tuple)
    last_restart_samples: tuple[str, ...] = field(default_factory=tuple)
    last_healthy_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_original_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_raw_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_banner_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_banner_raw_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_banner_preserved_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_banner_preserved_raw_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_banner_signature_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_codes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> "WorkerRestartTelemetry":
        context = metadata.get("docker_worker_context")
        contexts = _split_metadata_values(metadata.get("docker_worker_contexts"))
        if context:
            contexts = _coalesce_iterable([context, *contexts])
        restart_samples = _parse_int_sequence(
            metadata.get("docker_worker_restart_count_samples")
        )
        raw_backoff_options = [
            _normalise_backoff_hint(option)
            for option in _split_metadata_values(
                metadata.get("docker_worker_backoff_options")
            )
            if option
        ]
        backoff_options = _coalesce_iterable(
            option for option in raw_backoff_options if option
        )
        last_restart_samples = tuple(
            _split_metadata_values(metadata.get("docker_worker_last_restart_samples"))
        )
        last_healthy_samples = tuple(
            _split_metadata_values(metadata.get("docker_worker_last_healthy_samples"))
        )
        last_error_samples = tuple(
            _split_metadata_values(metadata.get("docker_worker_last_error_samples"))
        )
        last_error_original = metadata.get("docker_worker_last_error_original")
        last_error_original_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_original_samples")
            )
        )
        last_error_raw = metadata.get("docker_worker_last_error_raw")
        last_error_raw_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_raw_samples")
            )
        )
        last_error_banner = metadata.get("docker_worker_last_error_banner")
        last_error_banner_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_banner_samples")
            )
        )
        last_error_banner_raw = metadata.get("docker_worker_last_error_banner_raw")
        last_error_banner_raw_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_banner_raw_samples")
            )
        )
        last_error_banner_preserved = metadata.get(
            "docker_worker_last_error_banner_preserved"
        )
        last_error_banner_preserved_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_banner_preserved_samples")
            )
        )
        last_error_banner_preserved_raw = metadata.get(
            "docker_worker_last_error_banner_preserved_raw"
        )
        last_error_banner_preserved_raw_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_banner_preserved_raw_samples")
            )
        )
        last_error_banner_signature = metadata.get(
            "docker_worker_last_error_banner_signature"
        )
        last_error_banner_signature_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_banner_signature_samples")
            )
        )
        error_codes = _split_metadata_values(
            metadata.get("docker_worker_last_error_codes")
        )
        primary_code = metadata.get("docker_worker_last_error_code")
        if primary_code:
            error_codes = _coalesce_iterable([primary_code, *error_codes])
        else:
            error_codes = _coalesce_iterable(error_codes)

        normalized_backoff = metadata.get("docker_worker_backoff")
        if normalized_backoff:
            normalized_backoff = _normalise_backoff_hint(normalized_backoff)

        warning_occurrences = _coerce_optional_int(
            metadata.get("docker_worker_warning_occurrences")
        ) or 0

        raw_context_occurrences = metadata.get("docker_worker_context_occurrences")
        context_occurrence_pairs: list[tuple[str, int]] = []
        for token in _split_metadata_values(raw_context_occurrences):
            if ":" not in token:
                continue
            name, raw_count = token.split(":", 1)
            cleaned_name = _clean_worker_metadata_value(name)
            if not cleaned_name:
                continue
            try:
                count_value = int(raw_count.strip())
            except ValueError:
                continue
            context_occurrence_pairs.append((cleaned_name, count_value))

        return cls(
            context=context,
            restart_count=_coerce_optional_int(
                metadata.get("docker_worker_restart_count")
            ),
            backoff_hint=normalized_backoff,
            last_seen=metadata.get("docker_worker_last_restart"),
            last_healthy=metadata.get("docker_worker_last_healthy"),
            last_error=metadata.get("docker_worker_last_error"),
            last_error_original=last_error_original,
            last_error_raw=last_error_raw,
            last_error_banner=last_error_banner,
            last_error_banner_raw=last_error_banner_raw,
            last_error_banner_preserved=last_error_banner_preserved,
            last_error_banner_preserved_raw=last_error_banner_preserved_raw,
            last_error_banner_signature=last_error_banner_signature,
            warning_occurrences=warning_occurrences,
            context_occurrences=tuple(context_occurrence_pairs),
            contexts=tuple(contexts),
            restart_samples=restart_samples,
            backoff_options=tuple(backoff_options),
            last_restart_samples=last_restart_samples,
            last_healthy_samples=last_healthy_samples,
            last_error_samples=last_error_samples,
            last_error_original_samples=tuple(last_error_original_samples),
            last_error_raw_samples=tuple(last_error_raw_samples),
            last_error_banner_samples=tuple(last_error_banner_samples),
            last_error_banner_raw_samples=tuple(last_error_banner_raw_samples),
            last_error_banner_preserved_samples=tuple(
                last_error_banner_preserved_samples
            ),
            last_error_banner_preserved_raw_samples=tuple(
                last_error_banner_preserved_raw_samples
            ),
            last_error_banner_signature_samples=tuple(
                last_error_banner_signature_samples
            ),
            last_error_codes=tuple(error_codes),
        )

    @property
    def backoff_seconds(self) -> float | None:
        """Best-effort conversion of the Docker restart backoff to seconds."""

        return _estimate_backoff_seconds(self.backoff_hint)

    @property
    def max_restart_count(self) -> int | None:
        """Return the highest restart count observed across metadata samples."""

        candidates: list[int] = []
        if self.restart_count is not None:
            candidates.append(self.restart_count)
        candidates.extend(self.restart_samples)
        if not candidates:
            return None
        return max(candidates)

    @property
    def all_contexts(self) -> tuple[str, ...]:
        """Return unique worker contexts with the primary context prioritised."""

        contexts: list[str] = []
        if self.context:
            contexts.append(self.context)
        contexts.extend(self.contexts)
        if not contexts:
            return ()
        return tuple(_coalesce_iterable(contexts))

    @property
    def all_last_restarts(self) -> tuple[str, ...]:
        """Return the set of observed restart markers."""

        markers: list[str] = []
        if self.last_seen:
            markers.append(self.last_seen)
        markers.extend(self.last_restart_samples)
        if not markers:
            return ()
        return tuple(_coalesce_iterable(markers))

    @property
    def all_last_healthy(self) -> tuple[str, ...]:
        """Return the set of timestamps when Docker reported the worker healthy."""

        markers: list[str] = []
        if self.last_healthy:
            markers.append(self.last_healthy)
        markers.extend(self.last_healthy_samples)
        if not markers:
            return ()
        return tuple(_coalesce_iterable(markers))

    @property
    def all_last_errors(self) -> tuple[str, ...]:
        """Return the set of observed error messages for the worker."""

        errors: list[str] = []
        if self.last_error:
            errors.append(self.last_error)
        errors.extend(self.last_error_samples)
        if not errors:
            return ()
        return tuple(_coalesce_iterable(errors))

    @property
    def max_backoff_seconds(self) -> float | None:
        """Return the slowest restart backoff advertised by Docker."""

        candidates: list[float] = []
        primary = self.backoff_seconds
        if primary is not None:
            candidates.append(primary)
        for option in self.backoff_options:
            seconds = _estimate_backoff_seconds(option)
            if seconds is not None:
                candidates.append(seconds)
        if not candidates:
            return None
        return max(candidates)


@dataclass(frozen=True)
class WorkerHealthAssessment:
    """Classification of Docker worker restart telemetry."""

    severity: Literal["info", "warning", "error"]
    headline: str
    details: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """Compose a human readable message from the assessment components."""

        segments = [self.headline]

        detail_segments = [detail.strip() for detail in self.details if detail]
        if detail_segments:
            segments.append("Additional context: " + " ".join(detail_segments))

        reason_segments = [reason.strip() for reason in self.reasons if reason]
        if reason_segments:
            segments.append("Diagnostic signals: " + " ".join(reason_segments))

        if self.remediation:
            segments.append(" ".join(hint.strip() for hint in self.remediation if hint))

        return " ".join(segment for segment in segments if segment)


_CRITICAL_WORKER_ERROR_KEYWORDS = (
    "fatal",
    "panic",
    "exhausted",
    "cannot allocate",
    "unrecoverable",
    "out of memory",
    "corrupted",
)


def _is_critical_worker_error(message: str | None) -> bool:
    """Return ``True`` when *message* indicates a severe worker failure."""

    if not message:
        return False
    lowered = message.lower()
    return any(keyword in lowered for keyword in _CRITICAL_WORKER_ERROR_KEYWORDS)


def _apply_error_code_guidance(
    codes: Iterable[str],
    *,
    register_reason: Callable[[str, str], None],
    detail_collector: list[str],
    remediation_collector: list[str],
) -> dict[str, str]:
    """Enrich worker assessments with remediation guidance for known error codes."""

    metadata: dict[str, str] = {}
    seen: set[str] = set()

    for raw_code in codes:
        if not raw_code:
            continue
        normalized = raw_code.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        directive = _WORKER_ERROR_CODE_GUIDANCE.get(normalized)
        if directive is None:
            directive = _derive_generic_error_code_guidance(normalized)
        if directive is None:
            continue

        reason_key = f"error_code_{normalized.lower()}"
        register_reason(reason_key, directive.reason)

        if directive.detail:
            detail = directive.detail.strip()
            if detail and detail not in detail_collector:
                if not detail.endswith("."):
                    detail += "."
                detail_collector.append(detail)

        for step in directive.remediation:
            normalized_step = step.strip()
            if not normalized_step:
                continue
            if normalized_step not in remediation_collector:
                remediation_collector.append(normalized_step)

        for key, value in directive.metadata.items():
            metadata.setdefault(key, value)

    return metadata


def _classify_worker_flapping(
    telemetry: WorkerRestartTelemetry,
    context: RuntimeContext,
) -> WorkerHealthAssessment:
    """Categorise Docker worker restart telemetry into actionable guidance."""

    details: list[str] = []
    remediation: list[str] = []
    severity_reasons: dict[str, str] = {}

    def _register_reason(key: str, message: str) -> None:
        if key in severity_reasons:
            return
        normalized = message.rstrip()
        if not normalized.endswith("."):
            normalized += "."
        severity_reasons[key] = normalized

    contexts = telemetry.all_contexts
    if contexts:
        if len(contexts) == 1:
            details.append(f"Affected component: {contexts[0]}.")
        else:
            joined = ", ".join(contexts)
            details.append(f"Affected components: {joined}.")

    occurrence_count = telemetry.warning_occurrences
    if occurrence_count:
        plural = "s" if occurrence_count != 1 else ""
        details.append(
            f"Docker emitted {occurrence_count} warning{plural} about stalled Docker Desktop workers during diagnostics."
        )
        if occurrence_count >= 4:
            _register_reason(
                "warning_frequency",
                "Docker emitted four or more warnings about stalled Docker Desktop workers during a single diagnostics run",
            )

    if telemetry.context_occurrences:
        rendered_context_occurrences = ", ".join(
            f"{name} ({count})" for name, count in telemetry.context_occurrences
        )
        details.append(
            f"Warning frequency by component: {rendered_context_occurrences}."
        )

    max_restart = telemetry.max_restart_count
    if max_restart is not None:
        plural = "s" if max_restart != 1 else ""
        details.append(
            f"Docker recorded up to {max_restart} restart{plural} during diagnostics."
        )
        additional_samples = [
            sample
            for sample in telemetry.restart_samples
            if sample != max_restart
        ]
        if additional_samples:
            rendered = ", ".join(str(sample) for sample in additional_samples)
            details.append(
                f"Additional restart counts observed across repeated runs: {rendered}."
            )
        if max_restart >= 6:
            _register_reason(
                "excessive_restarts",
                "Docker recorded at least six worker restarts during diagnostics",
            )
        elif max_restart >= 4:
            _register_reason(
                "sustained_restarts",
                "Docker recorded four or more worker restarts during diagnostics",
            )
    elif telemetry.restart_samples:
        rendered = ", ".join(str(sample) for sample in telemetry.restart_samples)
        details.append(f"Docker reported restart counts during diagnostics: {rendered}.")

    backoff_hint = telemetry.backoff_hint
    if backoff_hint:
        details.append(f"Backoff interval advertised by Docker: {backoff_hint}.")
    extra_backoff = [
        option
        for option in telemetry.backoff_options
        if option and option != backoff_hint
    ]
    if extra_backoff:
        rendered = ", ".join(extra_backoff)
        details.append(f"Additional backoff intervals observed: {rendered}.")

    max_backoff_seconds = telemetry.max_backoff_seconds
    if max_backoff_seconds is not None and max_backoff_seconds >= 60:
        descriptor = backoff_hint or extra_backoff[0] if extra_backoff else None
        if descriptor is None:
            descriptor = f"approximately {int(round(max_backoff_seconds))}s"
        _register_reason(
            "prolonged_backoff",
            f"Docker advertised a restart backoff of at least {descriptor}, indicating sustained recovery attempts",
        )

    restart_markers = telemetry.all_last_restarts
    if restart_markers:
        if len(restart_markers) == 1:
            details.append(f"Last restart marker: {restart_markers[0]}.")
        else:
            preview = restart_markers[:3]
            rendered = ", ".join(preview)
            if len(restart_markers) > len(preview):
                rendered += ", …"
            details.append(f"Restart markers captured from Docker diagnostics: {rendered}.")

    healthy_markers = telemetry.all_last_healthy
    if healthy_markers:
        details.append(
            f"Docker last reported the worker as healthy at {healthy_markers[0]}."
        )
        if len(healthy_markers) > 1:
            preview = healthy_markers[1:3]
            rendered = ", ".join(preview)
            if len(healthy_markers) > 3:
                rendered += ", …"
            details.append(f"Additional healthy timestamps observed: {rendered}.")

    raw_last_errors = telemetry.all_last_errors
    if raw_last_errors:
        sanitized_errors: list[str] = []
        for message in raw_last_errors:
            if _contains_worker_stall_signal(message):
                sanitized_errors.append(_WORKER_STALLED_PRIMARY_NARRATIVE)
            else:
                sanitized_errors.append(message)
        last_errors = tuple(_coalesce_iterable(sanitized_errors))
    else:
        last_errors = ()
    if last_errors:
        details.append(f"Most recent worker error: {last_errors[0]}.")
        if len(last_errors) > 1:
            preview = last_errors[1:3]
            rendered = ", ".join(preview)
            if len(last_errors) > 3:
                rendered += ", …"
            details.append(f"Additional errors encountered: {rendered}.")

    critical_errors = [message for message in last_errors if _is_critical_worker_error(message)]
    if critical_errors:
        _register_reason(
            "critical_error",
            f"Docker reported a critical worker error: {critical_errors[0]}",
        )

    error_codes = tuple(_coalesce_iterable(telemetry.last_error_codes))
    normalized_codes = {
        code.upper()
        for code in error_codes
        if isinstance(code, str) and code.strip()
    }
    if error_codes:
        primary_code = error_codes[0]
        label = _WORKER_ERROR_CODE_LABELS.get(
            primary_code,
            primary_code.replace("_", " "),
        )
        details.append(
            f"Docker categorised the recent worker issue as {label}."
        )

        code_reason_map = {
            "restart_loop": "Docker classified the worker as being stuck in a restart loop",
            "healthcheck_failure": "Docker reported repeated health-check failures for the worker",
        }
        for code in error_codes:
            reason = code_reason_map.get(code)
            if reason:
                _register_reason(f"code:{code}", reason)

        sustained_backoff = (
            max_backoff_seconds is not None
            and max_backoff_seconds >= _SUSTAINED_BACKOFF_THRESHOLD
        )
        if "stalled_restart" in error_codes and (
            (max_restart is not None and max_restart >= 3)
            or occurrence_count >= 2
            or sustained_backoff
        ):
            _register_reason(
                "persistent_stalls",
                "Docker repeatedly restarted the worker after stalls, indicating instability",
            )

    guidance_metadata = _apply_error_code_guidance(
        error_codes,
        register_reason=_register_reason,
        detail_collector=details,
        remediation_collector=remediation,
    )

    sustained_backoff = (
        max_backoff_seconds is not None
        and max_backoff_seconds >= _SUSTAINED_BACKOFF_THRESHOLD
    )

    severity: Literal["info", "warning", "error"]
    benign_codes_only = not normalized_codes or normalized_codes <= _BENIGN_WORKER_ERROR_CODES
    if benign_codes_only and normalized_codes:
        for code in normalized_codes:
            severity_reasons.pop(f"error_code_{code.lower()}", None)

    if severity_reasons:
        severity = "error"
    else:
        mild_recovery = (
            occurrence_count <= 2
            and (max_restart is None or max_restart <= 3)
            and not sustained_backoff
            and benign_codes_only
            and not critical_errors
        )
        severity = "info" if mild_recovery else "warning"

    if severity == "info":
        if (
            occurrence_count > 1
            or (max_restart is not None and max_restart > 1)
            or (normalized_codes & _BENIGN_WORKER_ERROR_CODES)
        ):
            headline = (
                "Docker Desktop reported it recovered from transient worker stalls and the background worker is stable."
            )
            details.insert(
                0,
                "Docker Desktop recovered from transient worker stalls and reports the background worker is stable.",
            )
            guidance_metadata.setdefault("docker_worker_health_state", "stabilising")
        else:
            headline = (
                "Docker Desktop reported it briefly restarted a background worker and it is healthy."
            )
            details.insert(
                0,
                "Docker Desktop recovered from a brief background worker restart and reports it is healthy.",
            )
            guidance_metadata.setdefault("docker_worker_health_state", "recovered")

        remediation.append(
            "No immediate action is required, but monitor Docker Desktop if the message reappears."
        )
    elif severity == "warning":
        headline = (
            "Docker Desktop reported worker restarts but indicated they are recovering automatically."
        )
        remediation.append(
            "Monitor Docker Desktop and re-run bootstrap if instability persists."
        )
        if context.is_wsl:
            remediation.append(
                "Ensure WSL 2 is enabled for the distribution, install the latest WSL kernel update, and enable Docker Desktop's WSL integration for the distribution in settings."
            )
        elif context.is_windows:
            remediation.append(
                "Enable the Hyper-V and Virtual Machine Platform Windows features, allocate sufficient CPU and memory to Docker Desktop, and restart Docker Desktop after applying changes."
            )
        else:
            remediation.append(
                "Restart the Docker daemon and inspect host virtualization services for resource starvation or crashes."
            )
    else:
        headline = (
            "Docker Desktop worker processes are repeatedly restarting and may not stabilize without intervention."
        )
        details.insert(
            0,
            "Docker Desktop reported persistent background worker restarts that are not recovering automatically.",
        )
        if context.is_wsl:
            remediation.append(
                "Ensure WSL 2 is enabled for the distribution, install the latest WSL kernel update, and enable Docker Desktop's WSL integration for the distribution in settings."
            )
        elif context.is_windows:
            remediation.append(
                "Enable the Hyper-V and Virtual Machine Platform Windows features, allocate sufficient CPU and memory to Docker Desktop, and restart Docker Desktop after applying changes."
            )
        else:
            remediation.append(
                "Restart the Docker daemon and inspect host virtualization services for resource starvation or crashes."
            )

    return WorkerHealthAssessment(
        severity=severity,
        headline=headline,
        details=tuple(details),
        remediation=tuple(remediation),
        reasons=tuple(severity_reasons.values()),
        metadata=guidance_metadata,
    )


def _post_process_docker_health(
    *,
    metadata: dict[str, str],
    context: RuntimeContext,
    timeout: float = 6.0,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Augment diagnostics when Docker reports unhealthy background workers."""

    worker_health = metadata.get("docker_worker_health")
    if worker_health != "flapping":
        return [], [], {}

    telemetry = WorkerRestartTelemetry.from_metadata(metadata)
    assessment = _classify_worker_flapping(telemetry, context)

    warnings: list[str] = []
    errors: list[str] = []
    additional_metadata: dict[str, str] = {}

    summary = assessment.render()
    severity = assessment.severity

    if assessment.metadata:
        additional_metadata.update(assessment.metadata)

    if assessment.reasons:
        additional_metadata["docker_worker_health_reasons"] = "; ".join(
            assessment.reasons
        )

    virtualization_warnings: list[str] = []
    virtualization_errors: list[str] = []
    virtualization_metadata: dict[str, str] = {}

    if context.is_wsl or context.is_windows:
        vw_warnings, vw_errors, vw_metadata = _collect_windows_virtualization_insights(
            timeout=timeout
        )
        virtualization_warnings.extend(vw_warnings)
        virtualization_errors.extend(vw_errors)
        virtualization_metadata.update(vw_metadata)

    virtualization_findings_present = bool(
        virtualization_warnings or virtualization_errors or virtualization_metadata
    )

    if severity == "info" and virtualization_findings_present:
        severity = "warning"
        summary = (
            "Docker Desktop reported worker restarts and host virtualization diagnostics "
            "flagged configuration issues that require attention. Review the warnings "
            "below to stabilise Docker Desktop before retrying."
        )

    additional_metadata["docker_worker_health_severity"] = severity
    if summary:
        sanitized_summary = _enforce_worker_banner_sanitization(
            [summary],
            additional_metadata,
        )
        if sanitized_summary:
            summary = sanitized_summary[0]
        additional_metadata["docker_worker_health_summary"] = summary
    if virtualization_metadata:
        additional_metadata.update(virtualization_metadata)

    message_registry: set[str] = set()

    def _append_unique(collection: list[str], message: str) -> None:
        normalized = (message or "").strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in message_registry:
            return
        message_registry.add(key)
        collection.append(normalized)

    if severity == "info":
        return warnings, errors, additional_metadata

    if summary:
        target = warnings if severity == "warning" else errors
        _append_unique(target, summary)

    if virtualization_warnings:
        sanitized_virtualization_warnings = _enforce_worker_banner_sanitization(
            virtualization_warnings,
            additional_metadata,
        )
        for message in sanitized_virtualization_warnings:
            _append_unique(warnings, message)

    if virtualization_errors:
        sanitized_virtualization_errors = _enforce_worker_banner_sanitization(
            virtualization_errors,
            additional_metadata,
        )
        if severity == "error":
            for message in sanitized_virtualization_errors:
                _append_unique(errors, message)
        else:
            for message in sanitized_virtualization_errors:
                _append_unique(
                    warnings,
                    f"Virtualization issue detected: {message}",
                )

    return warnings, errors, additional_metadata


def _normalize_docker_warnings(value: object) -> tuple[list[str], dict[str, str]]:
    """Normalise Docker warnings into unique, user-friendly strings."""

    return _normalize_warning_collection(_iter_docker_warning_messages(value))


def _extract_json_document(stdout: str, stderr: str) -> tuple[str | None, list[str], dict[str, str]]:
    """Extract the JSON payload and structured warnings from Docker command output.

    Docker Desktop – especially on Windows – occasionally prefixes formatted
    JSON output with warning banners such as ``WARNING: worker stalled;
    restarting``.  Earlier bootstrap logic attempted to ``json.loads`` the raw
    ``stdout`` payload which failed in these situations and left the user with a
    cryptic parsing error.  This helper tolerantly searches ``stdout`` for the
    first decodable JSON document while collecting any surrounding text as human
    readable warnings.  ``stderr`` content is also normalised into warnings and
    annotated metadata so that diagnostics remain actionable.
    """

    decoder = json.JSONDecoder()
    warnings: list[str] = []

    def _normalise_stream(value: str | None) -> str:
        if not value:
            return ""
        return value.replace("\r", "\n")

    stdout_normalized = _normalise_stream(stdout)
    stderr_normalized = _normalise_stream(stderr)

    json_fragment: str | None = None

    if stdout_normalized:
        search_text = stdout_normalized
        index = 0
        length = len(search_text)
        while index < length:
            char = search_text[index]
            if char not in "[{":
                index += 1
                continue
            try:
                _, end = decoder.raw_decode(search_text[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            start = index
            finish = index + end
            json_fragment = search_text[start:finish]
            prefix = search_text[:start]
            suffix = search_text[finish:]
            warnings.extend(_iter_docker_warning_messages(prefix))
            warnings.extend(_iter_docker_warning_messages(suffix))
            break
        if json_fragment is None:
            warnings.extend(_iter_docker_warning_messages(search_text))

    if json_fragment is None:
        warnings.extend(_iter_docker_warning_messages(stderr_normalized))
        normalized_warnings, metadata = _normalize_warning_collection(warnings)
        return None, normalized_warnings, metadata

    warnings.extend(_iter_docker_warning_messages(stderr_normalized))
    normalized_warnings, warning_metadata = _normalize_warning_collection(warnings)
    return json_fragment.strip(), normalized_warnings, warning_metadata


def _parse_docker_json(
    proc: subprocess.CompletedProcess[str],
    command: str,
) -> tuple[object | None, list[str], dict[str, str]]:
    """Return decoded JSON output, normalised warnings, and metadata."""

    payload, collected_warnings, warning_metadata = _extract_json_document(
        proc.stdout, proc.stderr
    )

    if not payload:
        collected_warnings.append(f"docker {command} produced no JSON output")
        normalized_warnings, metadata = _normalize_warning_collection(collected_warnings)
        metadata.update(warning_metadata)
        return None, normalized_warnings, metadata

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        collected_warnings.append(f"Failed to parse docker {command} payload: {exc}")
        normalized_warnings, metadata = _normalize_warning_collection(collected_warnings)
        metadata.update(warning_metadata)
        return None, normalized_warnings, metadata

    return decoded, collected_warnings, warning_metadata


def _summarize_docker_command_failure(
    proc: subprocess.CompletedProcess[str],
    command: str,
) -> tuple[str, list[str], dict[str, str]]:
    """Return a sanitized failure message and extracted warning metadata."""

    components = []
    if proc.stderr:
        components.append(proc.stderr)
    if proc.stdout:
        components.append(proc.stdout)
    combined = "\n".join(components)

    normalized_stream = _strip_control_sequences(combined).replace("\r", "\n")
    raw_lines = list(_coalesce_warning_lines(normalized_stream))

    raw_warnings: list[str] = []
    residual_lines: list[str] = []

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        cleaned_warning, _ = _normalise_docker_warning(stripped)
        if cleaned_warning:
            raw_warnings.append(stripped)
            continue

        normalized = _normalise_worker_stalled_phrase(stripped)
        if _contains_worker_stall_signal(normalized):
            normalized_error, detail, _ = _normalise_worker_error_message(
                normalized,
                raw_original=stripped,
            )
            residual_lines.append(detail or normalized_error or normalized)
        else:
            residual_lines.append(normalized)

    normalized_warnings, metadata = _normalize_warning_collection(raw_warnings)
    residual_text = "; ".join(line for line in residual_lines if line)
    detail_text = residual_text or "; ".join(normalized_warnings)

    message = (
        f"docker {command} returned non-zero exit code {proc.returncode}: "
        f"{detail_text or 'no diagnostic output provided'}"
    )

    return message, normalized_warnings, metadata


def _probe_docker_environment(cli_path: Path, timeout: float) -> tuple[dict[str, str], list[str], list[str]]:
    """Gather Docker daemon metadata and associated warnings/errors."""

    metadata: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []

    version_proc, failure = _run_docker_command(cli_path, ["version", "--format", "{{json .}}"], timeout=timeout)
    if failure:
        errors.append(failure)
        return metadata, warnings, errors

    if version_proc is None:
        return metadata, warnings, errors

    if version_proc.returncode != 0:
        message, failure_warnings, failure_metadata = _summarize_docker_command_failure(
            version_proc,
            "version",
        )
        warnings.extend(failure_warnings)
        metadata.update(failure_metadata)
        errors.append(message)
        return metadata, warnings, errors

    version_data, version_warnings, version_metadata = _parse_docker_json(
        version_proc, "version"
    )
    warnings.extend(version_warnings)
    metadata.update(version_metadata)

    if isinstance(version_data, dict):
        client_data = version_data.get("Client", {}) if isinstance(version_data, dict) else {}
        server_data = version_data.get("Server", {}) if isinstance(version_data, dict) else {}
        client_version = str(client_data.get("Version", "")).strip()
        api_version = str(client_data.get("ApiVersion", "")).strip()
        server_version = str(server_data.get("Version", "")).strip()
        if client_version:
            metadata["client_version"] = client_version
        if api_version:
            metadata["api_version"] = api_version
        if server_version:
            metadata["server_version"] = server_version
        else:
            errors.append(
                "Docker daemon appears to be unavailable; version output omitted server details. "
                "Start Docker Desktop or connect to a reachable daemon."
            )

    if errors:
        return metadata, warnings, errors

    info_proc, failure = _run_docker_command(cli_path, ["info", "--format", "{{json .}}"], timeout=timeout)
    if failure:
        errors.append(failure)
        return metadata, warnings, errors

    if info_proc is None:
        return metadata, warnings, errors

    if info_proc.returncode != 0:
        message, failure_warnings, failure_metadata = _summarize_docker_command_failure(
            info_proc,
            "info",
        )
        warnings.extend(failure_warnings)
        metadata.update(failure_metadata)
        errors.append(message)
        return metadata, warnings, errors

    info_data, info_warnings, info_metadata = _parse_docker_json(info_proc, "info")
    warnings.extend(info_warnings)
    metadata.update(info_metadata)

    if isinstance(info_data, dict):
        for key, metadata_key in (
            ("ServerVersion", "server_version"),
            ("OperatingSystem", "operating_system"),
            ("OSType", "os_type"),
            ("Architecture", "architecture"),
            ("DockerRootDir", "root_dir"),
        ):
            value = str(info_data.get(key, "")).strip()
            if value and metadata_key not in metadata:
                metadata[metadata_key] = value

        warnings_field = info_data.get("Warnings")
        normalized_warnings, warning_metadata = _normalize_docker_warnings(warnings_field)
        warnings.extend(normalized_warnings)
        metadata.update(warning_metadata)

        if _is_windows() or _is_wsl():
            context = str(info_data.get("Name", "")).strip()
            if context and context.lower() not in {"docker-desktop", "desktop-linux"}:
                warnings.append(
                    "Docker context '%s' is active; Docker Desktop typically uses 'docker-desktop'. "
                    "Verify that the desired context is selected before launching sandboxes."
                    % context
                )
    elif info_data is not None:  # pragma: no cover - unexpected payloads
        warnings.append("docker info returned an unexpected payload structure")

    return metadata, warnings, errors


def _infer_missing_docker_skip_reason(context: RuntimeContext) -> str | None:
    """Return a descriptive reason for skipping Docker verification if appropriate."""

    assume_no = os.getenv(_DOCKER_ASSUME_NO_ENV)
    if assume_no and assume_no.strip().lower() in {"1", "true", "yes", "on"}:
        return f"Docker diagnostics disabled via {_DOCKER_ASSUME_NO_ENV}"

    if context.inside_container and not context.is_windows:
        runtime_label = context.container_runtime or "container"
        indicators = ", ".join(context.container_indicators) or "no explicit indicators"
        return (
            "Detected execution inside a %s-managed environment (%s) without Docker CLI access; "
            "assuming the host manages containers and skipping Docker diagnostics."
            % (runtime_label, indicators)
        )

    if context.is_ci:
        ci_label = ", ".join(context.ci_indicators) or "CI"
        return (
            "Detected continuous integration environment (%s) without Docker CLI access; "
            "skipping Docker diagnostics."
            % ci_label
        )

    return None


def _collect_docker_diagnostics(timeout: float = 12.0) -> DockerDiagnosticResult:
    """Inspect the Docker environment and return detailed diagnostics."""

    context = _detect_runtime_context()
    metadata: dict[str, str] = context.to_metadata()

    cli_path, cli_warnings = _discover_docker_cli()
    warnings = list(cli_warnings)
    info_messages: list[str] = []
    errors: list[str] = []

    if cli_path is None:
        skip_reason = _infer_missing_docker_skip_reason(context)
        if skip_reason:
            metadata["skip_reason"] = skip_reason
            return DockerDiagnosticResult(
                cli_path=None,
                available=False,
                errors=(),
                warnings=(),
                infos=(),
                metadata=metadata,
                skipped=True,
                skip_reason=skip_reason,
            )

        virtualization_warnings: list[str] = []
        virtualization_errors: list[str] = []
        virtualization_metadata: dict[str, str] = {}

        if context.is_windows or context.is_wsl:
            vw_warnings, vw_errors, vw_metadata = _collect_windows_virtualization_insights(
                timeout=timeout
            )
            virtualization_warnings.extend(vw_warnings)
            virtualization_errors.extend(vw_errors)
            virtualization_metadata.update(vw_metadata)

        errors.append(
            "Docker CLI executable was not found. Install Docker Desktop or ensure 'docker' is on PATH."
        )

        if virtualization_warnings:
            warnings.extend(virtualization_warnings)
        if virtualization_errors:
            errors.extend(virtualization_errors)
        if virtualization_metadata:
            metadata.update(virtualization_metadata)

        return DockerDiagnosticResult(
            cli_path=None,
            available=False,
            errors=tuple(_coalesce_iterable(errors)),
            warnings=tuple(_coalesce_iterable(warnings)),
            infos=tuple(info_messages),
            metadata=metadata,
        )

    metadata["cli_path"] = str(cli_path)

    probe_metadata, probe_warnings, probe_errors = _probe_docker_environment(cli_path, timeout)
    metadata.update(probe_metadata)
    warnings.extend(probe_warnings)
    errors.extend(probe_errors)

    health_warnings, health_errors, health_metadata = _post_process_docker_health(
        metadata=metadata,
        context=context,
    )
    warnings.extend(health_warnings)
    errors.extend(health_errors)
    metadata.update(health_metadata)

    if _should_collect_windows_virtualization_followups(metadata, context):
        vw_warnings, vw_errors, vw_metadata = _collect_windows_virtualization_insights(
            timeout=timeout
        )
        if vw_warnings:
            warnings.extend(vw_warnings)
        if vw_errors:
            errors.extend(vw_errors)
        for key, value in vw_metadata.items():
            metadata.setdefault(key, value)

    warnings, worker_metadata = _scrub_residual_worker_warnings(warnings)
    for key, value in worker_metadata.items():
        metadata.setdefault(key, value)

    errors, error_worker_metadata = _scrub_residual_worker_warnings(errors)
    for key, value in error_worker_metadata.items():
        metadata.setdefault(key, value)

    info_messages, info_worker_metadata = _scrub_residual_worker_warnings(info_messages)
    for key, value in info_worker_metadata.items():
        metadata.setdefault(key, value)

    _reclassify_worker_guidance_messages(
        warnings=warnings,
        errors=errors,
        infos=info_messages,
        metadata=metadata,
    )

    info_updates, remaining_warnings = _reclassify_worker_warnings_for_info(warnings, metadata)
    if info_updates:
        info_messages.extend(info_updates)
    warnings = remaining_warnings

    warnings = _enforce_worker_banner_sanitization(warnings, metadata)
    errors = _enforce_worker_banner_sanitization(errors, metadata)
    info_messages = _enforce_worker_banner_sanitization(info_messages, metadata)

    warnings = _guarantee_worker_banner_suppression(warnings, metadata)
    errors = _guarantee_worker_banner_suppression(errors, metadata)
    info_messages = _guarantee_worker_banner_suppression(info_messages, metadata)

    # Windows builds of Docker Desktop occasionally replay cached telemetry
    # blobs after the primary sanitisation passes have completed.  Those blobs
    # may contain verbatim ``worker stalled; restarting`` banners in metadata
    # fields such as ``docker_worker_last_error_banner_raw`` even though the
    # human facing warnings have been rewritten already.  Apply one final sweep
    # over the metadata to guarantee that every lingering banner is converted
    # into the canonical guidance narrative before returning diagnostics to the
    # caller.  This keeps downstream tooling from leaking the raw banner when
    # rendering the metadata dictionary verbatim (for example in Windows event
    # viewers or CI upload artefacts).
    _redact_worker_banner_artifacts(metadata)

    warnings = _finalize_worker_banner_sequences(warnings, metadata)
    errors = _finalize_worker_banner_sequences(errors, metadata)
    info_messages = _finalize_worker_banner_sequences(info_messages, metadata)

    _finalize_worker_banner_metadata(metadata)

    warnings = _coalesce_iterable(warnings)
    errors = _coalesce_iterable(errors)
    info_messages = _coalesce_iterable(info_messages)

    available = not errors

    return DockerDiagnosticResult(
        cli_path=cli_path,
        available=available,
        errors=tuple(errors),
        warnings=tuple(warnings),
        infos=tuple(info_messages),
        metadata=metadata,
    )


def _bool_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _verify_docker_environment() -> None:
    """Perform Docker diagnostics and surface actionable guidance to the user."""

    if _bool_env_flag(_DOCKER_SKIP_ENV):
        LOGGER.info(
            "Skipping Docker diagnostics due to %s environment override",
            _DOCKER_SKIP_ENV,
        )
        return

    diagnostics = _collect_docker_diagnostics()

    if diagnostics.skipped:
        message = diagnostics.skip_reason or "Docker diagnostics skipped"
        LOGGER.info("Skipping Docker diagnostics: %s", message)
        extra_context = {
            key: value
            for key, value in diagnostics.metadata.items()
            if key not in {"skip_reason", "cli_path"}
        }
        if extra_context:
            LOGGER.debug("Docker runtime context: %s", extra_context)
        return

    for notice in diagnostics.infos:
        LOGGER.info("Docker diagnostic notice: %s", notice)

    for warning in diagnostics.warnings:
        LOGGER.warning("Docker diagnostic warning: %s", warning)

    if diagnostics.available:
        details = {
            key: value
            for key, value in diagnostics.metadata.items()
            if key in {"server_version", "operating_system", "os_type", "architecture", "cli_path"}
            and value
        }
        if details:
            summary = ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
            LOGGER.info("Docker daemon reachable (%s)", summary)
        return

    for error in diagnostics.errors:
        LOGGER.warning("Docker diagnostic error: %s", error)

    if _bool_env_flag(_DOCKER_REQUIRE_ENV):
        raise BootstrapError(
            "Docker environment verification failed and %s is set. "
            "Review the diagnostic warnings above before retrying." % _DOCKER_REQUIRE_ENV
        )


def _run_bootstrap(config: BootstrapConfig) -> None:
    resolved_env_file = _prepare_environment(config)

    _verify_docker_environment()

    from menace.bootstrap_policy import PolicyLoader
    from menace.environment_bootstrap import EnvironmentBootstrapper
    import startup_checks
    from startup_checks import run_startup_checks
    from menace.bootstrap_defaults import ensure_bootstrap_defaults

    created, env_file = ensure_bootstrap_defaults(
        startup_checks.REQUIRED_VARS,
        repo_root=_REPO_ROOT,
        env_file=resolved_env_file,
    )
    if created:
        LOGGER.info("Persisted generated defaults to %s", env_file)

    loader = PolicyLoader()
    auto_install = startup_checks.auto_install_enabled()
    env_requested = os.getenv("MENACE_BOOTSTRAP_PROFILE")
    requested = env_requested or ("minimal" if not auto_install else None)
    policy = loader.resolve(
        requested=requested,
        auto_install_enabled=auto_install,
    )
    run_startup_checks(skip_stripe_router=config.skip_stripe_router, policy=policy)
    EnvironmentBootstrapper(policy=policy).bootstrap()
    LOGGER.info("Environment bootstrap completed successfully")
    logging.shutdown()
    if sys.stdout is not None and sys.stdout.isatty():
        print("Bootstrap complete. You can close this window.")
        sys.stdout.flush()
    _wait_for_windows_console_visibility()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = BootstrapConfig.from_namespace(args)
    _configure_logging(config.log_level)
    try:
        _run_bootstrap(config)
    except BootstrapError as exc:
        LOGGER.error("bootstrap aborted: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        LOGGER.warning("Bootstrap interrupted by user")
        raise SystemExit(130)
    except Exception as exc:  # pragma: no cover - safety net
        LOGGER.exception("Unexpected error during bootstrap")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
