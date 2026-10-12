"""Text to speech engine interface.

Two implementations ship: Piper (neural, natural sounding, needs a voice file that
`setup` downloads once) and Windows SAPI (built into the OS, used as the always
available fallback so the app never fails to speak).
"""

from __future__ import annotations

import abc

import numpy as np


class TTSEngine(abc.ABC):
    """Synthesises text into a mono float32 waveform at the engine sample rate."""

    name: str = "abstract"
    sample_rate: int = 22_050

    @abc.abstractmethod
    def available(self) -> bool:
        """True when the engine can actually produce audio right now."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> np.ndarray | None:
        """Return audio samples in [-1, 1], or None if synthesis failed."""

    def describe(self) -> str:
        return self.name

    def voices(self) -> list[str]:
        return [self.name]

    def close(self) -> None:  # noqa: B027
        """Release any long lived resource. Optional for stateless engines."""


class NullEngine(TTSEngine):
    """Used when speech is switched off, keeps the pipeline branch free."""

    name = "off"

    def available(self) -> bool:
        return False

    def synthesize(self, text: str) -> np.ndarray | None:
        return None
