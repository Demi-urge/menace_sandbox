"""Cross-platform installer for Menace as a system service."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from dynamic_path_router import resolve_path

SERVICE_NAME = "menace"


def _install_systemd() -> None:
    service_file = resolve_path("systemd/menace.service")
    target = Path("/etc/systemd/system/menace.service")
    if os.geteuid() != 0:
        print("Systemd installation requires root privileges")
        return
    try:
        shutil.copy(service_file, target)
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", SERVICE_NAME], check=True)
        print(f"Installed systemd service at {target}")
    except Exception as exc:  # pragma: no cover - best effort
        raise RuntimeError(f"failed installing systemd service: {exc}") from exc


def _install_windows() -> None:
    exe = sys.executable
    script = resolve_path("service_supervisor.py")
    cmd = [
        "sc",
        "create",
        SERVICE_NAME,
        f"binPath= \"{exe} {script}\"",
        "start= auto",
    ]
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        print("Windows service created. Start with 'sc start menace'.")
    except Exception as exc:  # pragma: no cover - best effort
        raise RuntimeError(f"failed installing windows service: {exc}") from exc


def _load_env(path: str | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path:
        return env
    p = Path(path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _generate_k8s(
    path: Path,
    *,
    env_file: str | None = ".env",
    env: dict[str, str] | None = None,
    volumes: list[str] | None = None,
    cpu: str | None = None,
    memory: str | None = None,
) -> None:
    """Write a Kubernetes deployment manifest with optional extras."""

    env_vars = _load_env(env_file)
    env_vars.update(env or {})

    volume_specs = list(volumes or [])
    if "MENACE_VOLUMES" in env_vars:
        volume_specs.extend(
            [v.strip() for v in env_vars.pop("MENACE_VOLUMES").split(",") if v.strip()]
        )

    cpu = cpu or env_vars.pop("CPU_LIMIT", None)
    memory = memory or env_vars.pop("MEMORY_LIMIT", None)

    container: dict[str, object] = {
        "name": "menace",
        "image": "menace:latest",
        "command": ["python", "-m", "menace.service_supervisor"],
    }
    if env_vars:
        container["env"] = [
            {"name": k, "value": v} for k, v in sorted(env_vars.items())
        ]

    volume_mounts = []
    volumes_section = []
    for idx, spec in enumerate(volume_specs):
        if ":" not in spec:
            continue
        host, mount = spec.split(":", 1)
        name = f"vol{idx}"
        volume_mounts.append({"name": name, "mountPath": mount})
        volumes_section.append(
            {"name": name, "hostPath": {"path": host, "type": "DirectoryOrCreate"}}
        )
    if volume_mounts:
        container["volumeMounts"] = volume_mounts

    if cpu or memory:
        limits: dict[str, str] = {}
        if cpu:
            limits["cpu"] = str(cpu)
        if memory:
            limits["memory"] = str(memory)
        container["resources"] = {"limits": limits}

    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "menace"},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "menace"}},
            "template": {
                "metadata": {"labels": {"app": "menace"}},
                "spec": {"containers": [container]},
            },
        },
    }
    if volumes_section:
        manifest["spec"]["template"]["spec"]["volumes"] = volumes_section

    import yaml

    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(
        f"Kubernetes manifest written to {path}. "
        f"Enable with 'kubectl apply -f {path}'."
    )


def _generate_swarm(
    path: Path,
    *,
    env_file: str | None = ".env",
    env: dict[str, str] | None = None,
    volumes: list[str] | None = None,
    cpu: str | None = None,
    memory: str | None = None,
) -> None:
    """Write a Docker Swarm compose file with extras."""

    env_vars = _load_env(env_file)
    env_vars.update(env or {})

    volume_specs = list(volumes or [])
    if "MENACE_VOLUMES" in env_vars:
        volume_specs.extend(
            [v.strip() for v in env_vars.pop("MENACE_VOLUMES").split(",") if v.strip()]
        )

    cpu = cpu or env_vars.pop("CPU_LIMIT", None)
    memory = memory or env_vars.pop("MEMORY_LIMIT", None)

    service: dict[str, object] = {
        "image": "menace:latest",
        "command": "python -m menace.service_supervisor",
        "deploy": {"restart_policy": {"condition": "any"}},
    }
    if env_vars:
        service["environment"] = [f"{k}={v}" for k, v in sorted(env_vars.items())]
    if volume_specs:
        service["volumes"] = volume_specs
    if cpu or memory:
        limits: dict[str, str] = {}
        if cpu:
            limits["cpus"] = str(cpu)
        if memory:
            limits["memory"] = str(memory)
        service["deploy"]["resources"] = {"limits": limits}

    manifest = {"version": "3.8", "services": {"menace": service}}

    import yaml

    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(
        f"Compose file written to {path}. "
        f"Deploy with 'docker stack deploy -c {path} menace'."
    )


def main(argv: list[str] | None = None) -> None:
    """Install Menace as a service or generate orchestrator manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orchestrator",
        choices=["k8s", "swarm"],
        help="Generate Kubernetes or Docker Swarm manifests",
    )
    parser.add_argument("--output", help="Output manifest path")
    parser.add_argument("--env-file", default=".env", help="Environment file")
    parser.add_argument(
        "--volume",
        action="append",
        help="Volume mount specification HOST:CONTAINER (can be repeated)",
    )
    parser.add_argument("--cpu-limit", help="CPU limit for the container")
    parser.add_argument("--memory-limit", help="Memory limit for the container")
    args = parser.parse_args(argv)

    if args.orchestrator == "k8s":
        out = Path(args.output or "menace-deployment.yaml")
        _generate_k8s(
            out,
            env_file=args.env_file,
            volumes=args.volume,
            cpu=args.cpu_limit,
            memory=args.memory_limit,
        )
        return
    if args.orchestrator == "swarm":
        out = Path(args.output or "docker-compose.yml")
        _generate_swarm(
            out,
            env_file=args.env_file,
            volumes=args.volume,
            cpu=args.cpu_limit,
            memory=args.memory_limit,
        )
        return

    sys_platform = platform.system()
    if sys_platform == "Windows":
        _install_windows()
    elif sys_platform in {"Linux", "Darwin"}:
        _install_systemd()
    else:
        print(f"unsupported platform: {sys_platform}")


if __name__ == "__main__":
    main()
