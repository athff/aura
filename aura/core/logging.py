"""
core/logging.py
---------------
AURA's centralized logging setup, built on the Python standard library.

Use it from any module like this:

    from aura.core.logging import get_logger
    logger = get_logger(__name__)   # e.g. logger name "aura.core.engine"

All AURA loggers share ONE console handler configured here, so output is
consistent and we avoid duplicate lines. AURA never logs API keys, tokens, or
complete sensitive configuration -- only contextual messages useful for
local development and debugging.
"""

import logging
import sys

# Root namespace for every AURA logger, e.g. "aura.core.engine".
AURA_LOGGER_NAME = "aura"

# Human-friendly, single-line console format.
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """
    Set up AURA's console logging (safe to call any number of times).

    A StreamHandler is attached to the "aura" logger. Because the handler
    exists after the first call, later calls are no-ops (no duplicate lines).
    """
    root = logging.getLogger(AURA_LOGGER_NAME)
    if root.handlers:  # already configured
        return

    root.setLevel(level.upper())  # e.g. "INFO", "WARNING", "ERROR"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.propagate = False  # avoid double-printing to the root logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger under the AURA namespace, configuring logging if needed.

    `name` should normally be the calling module's ``__name__``. The special
    value ``__main__`` is mapped to ``aura.main`` so scripts still belong to
    the AURA namespace.
    """
    configure_logging()

    if name == "__main__":
        return logging.getLogger("aura.main")
    if name.startswith(AURA_LOGGER_NAME + "."):
        return logging.getLogger(name)

    # Keep arbitrary callers in the AURA namespace for consistent filtering.
    return logging.getLogger(f"{AURA_LOGGER_NAME}.{name}")