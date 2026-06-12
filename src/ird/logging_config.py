"""Project-wide logging setup.

Call :func:`configure_logging` once at an application entry point. Library
modules should obtain loggers via ``logging.getLogger(__name__)`` and never
configure handlers themselves.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure a single console handler for the ``ird`` logger tree."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("ird")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
