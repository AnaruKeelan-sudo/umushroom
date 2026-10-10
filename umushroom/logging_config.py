"""Centralized logging configuration.

Using the logging module (not print) gives timestamps, log levels,
and the ability to redirect output to a file for audit trails —
important if you need to show a competition judge exactly what the
system decided and when.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    root = logging.getLogger("umushroom")
    root.setLevel(level.upper())
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"umushroom.{name}")
