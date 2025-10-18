"""
Generic utility helpers shared across the CLI commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_directory(path: Path) -> Path:
    """
    Create `path` (and parents) if it does not exist and return it for convenience.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    """
    Read a JSON file with UTF-8 encoding.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """
    Write JSON content with UTF-8 encoding and pretty indentation.
    """
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_yaml(path: Path) -> Any:
    """
    Read YAML data from `path` using yaml.safe_load.
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    """
    Convenience wrapper to write UTF-8 text.
    """
    path.write_text(content, encoding="utf-8")
