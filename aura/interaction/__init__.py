"""AURA interaction layer: provider-independent text interaction foundation.

Exposes the abstract ``Interaction`` contract (a modality through which a human
reads and writes with AURA), the ``InteractionEnd`` signal, the concrete
``TextInteraction`` implementation, and the reusable ``Session`` driver that
plugs any ``Interaction`` into any ``Responder`` (e.g. ``AuraEngine``).

This layer is deliberately independent of FastAPI, audio libraries, and the
brain/memory engine -- voice/streaming backends can be added later by
implementing ``Interaction`` without redesign.
"""

from aura.interaction.base import Interaction, InteractionEnd
from aura.interaction.session import Session
from aura.interaction.text import TextInteraction

__all__ = ["Interaction", "InteractionEnd", "Session", "TextInteraction"]