"""
core/errors.py
--------------
AURA's application-level exception hierarchy.

These live in the CORE layer (not the web layer) so the engine and the brain
providers can raise rich, typed errors WITHOUT knowing anything about FastAPI
or the web. It is the web layer's job to translate them into HTTP responses.

Hierarchy:

    AuraError (base for all expected application errors)
      ├─ BrainError    -- the engine could not get a usable reply from the Brain
      └─ ProviderError -- a provider returned a malformed/empty response or its
                          request failed
"""


class AuraError(Exception):
    """Base class for expected, recoverable AURA application errors."""


class BrainError(AuraError):
    """
    Raised by AuraEngine when the Brain fails or returns no usable reply.

    The original exception is preserved as the ``__cause__`` and is logged; it
    is NOT intended to be shown verbatim to end users or the browser.
    """


class ProviderError(AuraError):
    """
    Raised by a concrete Brain when a provider request fails or the response
    is malformed/empty. Keeps provider internals out of user-facing messages.
    """