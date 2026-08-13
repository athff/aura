"""Tests for AURA's error hierarchy in core/errors.py.

These are trivial by design: the hierarchy lives in the CORE layer so the engine
and brains can raise rich, typed errors without knowing about FastAPI. We assert
the structure so nobody accidentally flattens it back into bare exceptions.
"""

import pytest

from aura.core.errors import AuraError, BrainError, ProviderError


def test_brain_error_is_an_aura_error() -> None:
    assert issubclass(BrainError, AuraError)


def test_provider_error_is_an_aura_error() -> None:
    assert issubclass(ProviderError, AuraError)


def test_aura_error_is_an_exception() -> None:
    assert issubclass(AuraError, Exception)


def test_brain_error_is_not_provider_error() -> None:
    # They are siblings: a BrainError must never be mis-caught as a ProviderError.
    assert not issubclass(BrainError, ProviderError)
    assert not issubclass(ProviderError, BrainError)


def test_can_raise_and_catch_typed_errors() -> None:
    with pytest.raises(AuraError):
        raise ProviderError("provider boom")
    with pytest.raises(BrainError):
        raise BrainError("empty reply")


def test_errors_carry_a_message_and_chain_cause() -> None:
    try:
        try:
            raise KeyError("original")
        except KeyError as cause:
            raise ProviderError("wrapped") from cause
    except ProviderError as exc:
        assert str(exc) == "wrapped"
        assert isinstance(exc.__cause__, KeyError)