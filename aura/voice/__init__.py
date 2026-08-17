"""AURA voice runner: bounded voice conversation using local speech engines.

Run a single voice conversation with:
    python -m aura.voice

Wires together the existing ``SoundDeviceHardware``, ``VoskSpeechInput``,
``KokoroSpeechOutput``, ``VoiceInteraction``, ``AuraEngine``, and ``Session``
without modifying any of them. Model paths are configured via environment
variables (see ``__main__.py`` docstring).
"""