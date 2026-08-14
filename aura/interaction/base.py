"""
interaction/base.py
-------------------
The CONTRACT for an AURA *interaction*: an input/output modality through which
a human talks to AURA.

An interaction is deliberately tiny and modality-agnostic. It only knows how
to obtain one user utterance (``read``) and how to deliver AURA's reply back
(``write``). Whether the modality is a terminal, a web page, or a future
voice/streaming channel, the rest of AURA (and the ``Session`` driver) depends
only on this interface -- never on a concrete transport.

To add a new modality later (e.g. voice) you implement ``Interaction`` once
(``kind = "voice"``) and reuse the existing ``Session`` unchanged.
"""

from abc import ABC, abstractmethod
from typing import ClassVar


class InteractionEnd(Exception):
    """
    Raised by ``Interaction.read()`` when input is exhausted or closed
    (e.g. EOF / Ctrl-C). This is intentionally distinct from an empty line,
    which is just a blank utterance, so callers can tell \"no more input\" apart
    from \"skip this turn\".
    """


class Interaction(ABC):
    """Abstract base class: the blueprint that every modality implements."""

    # A short classifier naming the modality, e.g. ``"text"`` today and
    # ``"voice"`` in the future. Useful for routing/introspection.
    kind: ClassVar[str] = "unknown"

    @abstractmethod
    def read(self) -> str:
        """
        Block for one user utterance and return it as plain text (whitespace
        trimmed). If the input source is closed/exhausted, raise
        ``InteractionEnd`` so the driver knows the conversation is over.
        """
        raise NotImplementedError

    @abstractmethod
    def write(self, text: str) -> None:
        """
        Deliver text back to the human (usually AURA's reply). The concrete
        modality decides how to present it (print to a terminal, ship over a
        socket, etc.).
        """
        raise NotImplementedError