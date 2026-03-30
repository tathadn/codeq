"""Shared helpers: logging, serialization, config."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Generator

import yaml
from pydantic import BaseModel


def setup_logging(level: str = "INFO", name: str = "codeq") -> logging.Logger:
    """Configure and return a logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        name: Logger name.

    Returns:
        Configured Logger instance.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    return logging.getLogger(name)


def load_jsonl(path: Path) -> Generator[dict, None, None]:
    """Yield records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed JSON objects.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(record: dict, path: Path) -> None:
    """Append a single record to a JSONL file (creates file if absent).

    Args:
        record: Dictionary to serialize and append.
        path: Destination JSONL file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_json(path: Path) -> Any:
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    """Load a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML as a dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_class: type[BaseModel], path: Path) -> BaseModel:
    """Load and validate a YAML config file into a Pydantic model.

    Args:
        config_class: Pydantic BaseModel subclass to validate against.
        path: Path to the YAML config file.

    Returns:
        Validated config instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValidationError: If YAML content does not match the schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = load_yaml(path)
    return config_class(**raw)
