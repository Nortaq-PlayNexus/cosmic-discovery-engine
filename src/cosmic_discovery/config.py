"""Configuration loading from YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file, overlaid with environment variables.

    Environment variables prefixed with CDE_ override YAML values.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Flatten and apply env overrides
    _apply_env_overrides(raw)
    return raw


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Overlay CDE_* environment variables onto the config dict in place."""
    prefix = "CDE_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        _set_nested(config, path, _parse_env_value(value))


def _set_nested(d: dict[str, Any], path: list[str], value: Any) -> None:
    for part in path[:-1]:
        d = d.setdefault(part, {})
    d[path[-1]] = value


def _parse_env_value(value: str) -> Any:
    """Try to parse env var value as bool, int, float, or keep as string."""
    low = value.lower()
    if low in ("true", "yes", "1"):
        return True
    if low in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def get_data_dir(config: dict[str, Any]) -> Path:
    """Resolve and create the data directory."""
    raw = os.environ.get("CDE_DATA_DIR", config.get("database", {}).get("path", "./data"))
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_db_path(config: dict[str, Any]) -> Path:
    """Resolve the SQLite database path."""
    data_dir = get_data_dir(config)
    db_name = config.get("database", {}).get("db_name", "cosmic_discovery.db")
    return data_dir / db_name
