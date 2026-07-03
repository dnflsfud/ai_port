"""Centralized logging configuration for the cc2 harness.

Call ``setup_logging()`` once at the top of an entry-point script to get
structured output across the ``src.*`` package. Safe to call multiple times
— re-invocation replaces handlers rather than duplicating them.

Example::

    from src.logging_config import setup_logging
    setup_logging()  # defaults to INFO on stderr
    setup_logging(level="DEBUG")  # verbose (e.g., missing-sheet warnings)
    setup_logging(log_file="./outputs/run.log")  # also mirror to disk
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """Configure the root logger for cc2 harness entry points."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # Remove any handlers installed by previous setup_logging() or imports so
    # re-invocation is idempotent.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
