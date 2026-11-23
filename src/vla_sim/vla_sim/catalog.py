"""Model catalog: reads configs/models.yaml (drives the model dropdown and process launch)."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    server: str
    port: int
    raw: dict
    repo_root: Path | None = None

    @property
    def description(self) -> str:
        return self.raw.get("description", "")

    def checkpoint_cached(self) -> bool | None:
        """True if the checkpoint is already in the local openpi cache (None: not a gs:// checkpoint)."""
        ck = self.raw.get("checkpoint", "")
        if not ck.startswith("gs://"):
            return None
        cache = openpi_data_home(self.repo_root)
        local = cache / ck[len("gs://"):]
        # openpi downloads into a scratch dir and renames on completion, and holds <dir>.lock meanwhile
        return (local / "params").is_dir() and not any(local.parent.glob(local.name + ".partial*"))

    def server_command(self, repo_root: Path) -> list[str]:
        if self.server == "openpi":
            cmd = [str(repo_root / "scripts" / "serve_openpi.sh"), self.raw["openpi_config"], str(self.port)]
            return cmd + ([self.raw["checkpoint"]] if self.raw.get("checkpoint") else [])
        raise ValueError(f"unknown server type '{self.server}' for model '{self.name}'")


def openpi_data_home(repo_root: Path | None) -> Path:
    """openpi checkpoint cache, resolved like scripts/serve_openpi.sh: $OPENPI_DATA_HOME, else
    OPENPI_DATA_HOME=... in <repo>/.env (git-ignored, machine-specific), else ~/.cache/openpi."""
    value = os.environ.get("OPENPI_DATA_HOME")
    env_file = repo_root / ".env" if repo_root else None
    if not value and env_file is not None and env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENPI_DATA_HOME="):
                value = line.split("=", 1)[1].strip()
    return Path(value or "~/.cache/openpi").expanduser()


def load_catalog(repo_root: Path) -> list[ModelSpec]:
    data = yaml.safe_load((repo_root / "configs" / "models.yaml").read_text()) or {}
    return [
        ModelSpec(name=name, server=cfg["server"], port=int(cfg["port"]), raw=cfg, repo_root=repo_root)
        for name, cfg in (data.get("models") or {}).items()
    ]
