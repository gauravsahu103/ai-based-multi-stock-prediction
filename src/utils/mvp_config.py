from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def mvp_config_path() -> Path:
    return project_root() / "config" / "mvp.yaml"


@lru_cache(maxsize=1)
def load_mvp_config() -> dict[str, Any]:
    config_file = mvp_config_path()

    if not config_file.exists():
        raise FileNotFoundError(
            f"MVP config file not found: {config_file}"
        )

    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("MVP config must be a mapping")

    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return project_root() / path


def storage_path(key: str) -> Path:
    config = load_mvp_config()
    storage = config.get("storage")

    if not isinstance(storage, dict):
        raise KeyError("'storage' section missing in mvp.yaml")

    if key not in storage:
        raise KeyError(f"Storage key '{key}' missing in mvp.yaml")

    return resolve_project_path(storage[key])


def modeling_dir(day: int) -> Path:
    return storage_path("modeling")/f"day{day}"

def evaluation_dir(day: int) -> Path:
    return storage_path("evaluation")/ f"day{day}"


def processed_dir() -> Path:
    return storage_path("processed")

def features_technical_dir(day: int) -> Path:
    return storage_path("features") /f"day{day}"



def targets_day_dir(day: int) -> Path:
    return storage_path("targets")/ f"day{day}"


def models_dir() -> Path:
    return project_root()/ "models"


def predictions_dir() -> Path:
    return storage_path("predictions")



