from __future__ import annotations

"""Interface for model deployment and cloning."""

import logging
import shutil
from pathlib import Path

class ModelDeployer:
    """Simple file based deployer used by :class:`CrossModelComparator`."""

    def __init__(self, models_dir: str = "models", deploy_path: str = "active_model") -> None:
        self.models_dir = Path(models_dir)
        self.deploy_path = Path(deploy_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def deploy_model(self, name: str) -> None:  # pragma: no cover - filesystem
        """Deploy *name* as the active model by copying files."""
        try:
            src = self.models_dir / name
            if self.deploy_path.exists():
                if self.deploy_path.is_dir():
                    shutil.rmtree(self.deploy_path)
                else:
                    self.deploy_path.unlink()
            if src.is_dir():
                shutil.copytree(src, self.deploy_path)
            else:
                shutil.copy(src, self.deploy_path)
            self.logger.info("deployed model %s", name)
        except Exception as exc:
            self.logger.error("deployment failed: %s", exc)

    def clone_model(self, name: str) -> None:  # pragma: no cover - filesystem
        """Create a clone of *name* for backup before deployment."""
        try:
            src = self.models_dir / name
            clone = self.models_dir / f"{name}_clone"
            if clone.exists():
                if clone.is_dir():
                    shutil.rmtree(clone)
                else:
                    clone.unlink()
            if src.is_dir():
                shutil.copytree(src, clone)
            else:
                shutil.copy(src, clone)
            self.logger.info("cloned model %s", name)
        except Exception as exc:
            self.logger.error("clone failed: %s", exc)

    def retire_model(self, name: str) -> None:  # pragma: no cover - filesystem
        """Move *name* to a retired directory."""
        try:
            src = self.models_dir / name
            dest_dir = self.models_dir / "retired"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / name
            if src.exists():
                shutil.move(str(src), str(dest))
                self.logger.info("retired model %s", name)
        except Exception as exc:
            self.logger.error("retire failed: %s", exc)


__all__ = ["ModelDeployer"]
