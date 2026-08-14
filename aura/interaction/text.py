"""
interaction/text.py
-------------------
The text implementation of the ``Interaction`` contract: a terminal-style
conversation using plain stdin/stdout.

Reads and writes are REQUEST-LEVEL ``input``/``print`` by default, but both
are injected as callables so the class is trivially testable offline and can
be pointed at any text stream (e.g. a pseudo-terminal or a socket) without
any change to the contract.
"""

import builtins
from collections.abc import Callable

from aura.interaction.base import Interaction, InteractionEnd


class TextInteraction(Interaction):
    """A text/terminal interaction: one line in, one line out.

    Parameters
    ----------
    prompt:
        Shown when waiting for input (e.g. ``"You: "``).
    read_line:
        Callable ``prompt -> line``; defaults to the builtin ``input``.
    write_line:
        Callable ``text -> None``; defaults to the builtin ``print``.
    """

    kind = "text"

    def __init__(
        self,
        prompt: str = "You: ",
        read_line: Callable[[str], str] = builtins.input,
        write_line: Callable[[str], None] = print,
    ) -> None:
        self._prompt = prompt
        self._read_line = read_line
        self._write_line = write_line

    def read(self) -> str:
        """
        Read one trimmed user utterance. If the input is closed (EOF) or
        interrupted (Ctrl-C), raise ``InteractionEnd`` instead of returning.
        """
        try:
            line = self._read_line(self._prompt)
        except (EOFError, KeyboardInterrupt):
            raise InteractionEnd from None
        return str(line).strip()

    def write(self, text: str) -> None:
        """Emit ``text`` back to the user via the configured writer."""
        self._write_line(text)