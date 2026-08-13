"""Tests for AURA's centralized logging (core/logging.py).

We only assert the wiring -- logger names stay in the "aura" namespace and the
shared handler is idempotent -- so the suite stays free of real log IO.
"""

import logging

from aura.core.logging import AURA_LOGGER_NAME, configure_logging, get_logger


def test_logger_names_stay_in_aura_namespace() -> None:
    assert get_logger("aura.core.engine").name == "aura.core.engine"
    # Arbitrary caller names are prefixed so filtering always works.
    assert get_logger("some.module").name == "aura.some.module"
    # The entry point's special __main__ is mapped into the namespace too.
    assert get_logger("__main__").name == "aura.main"


def test_shared_root_is_aura() -> None:
    assert AURA_LOGGER_NAME == "aura"
    # A logger for a nested core module lives under the "aura" root.
    assert get_logger("aura.core.engine").name.startswith(AURA_LOGGER_NAME)


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger(AURA_LOGGER_NAME)
    configure_logging()
    before = len(root.handlers)
    # A second call must be a no-op: it must NOT add another handler.
    configure_logging()
    assert len(root.handlers) == before
    # AURA never propagates to the root logger (avoids duplicate console lines).
    assert not root.propagate
    # AURA's own configured handler is present among whatever pytest attached.
    assert any(
        isinstance(h, logging.StreamHandler) and h.formatter is not None
        for h in root.handlers
    )


def test_loggers_are_plain_logging_loggers() -> None:
    logger = get_logger("aura.test.extra")
    assert isinstance(logger, logging.Logger)