"""Windows SAPI text to speech.

This talks to the speech engine that ships with Windows, so it needs no download
and no extra binaries. It is noticeably more robotic than Piper, which is exactly
why it is the fallback rather than the default.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import TTSEngine


@lru_cache(maxsize=1)
def _pyttsx3():
    """Import lazily and cache: pyttsx3 is slow to start and may not be installed."""

    import pyttsx3

    return pyttsx3


class SAPIEngine(TTSEngine):
    name = "Windows SAPI"

    def __init__(self) -> None:
        self._engine = None
        self._rate = 0
        self._volume = 1.0

    # ------------------------------------------------------------- lifecycle

    def _ensure(self):
        if self._engine is None:
            engine = _pyttsx3().init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._voice_volume(self._volume))
            self._engine = engine
        return self._engine

    @staticmethod
    def _voice_volume(volume: float) -> float:
        return float(max(0.0, min(volume, 1.0)))

    def available(self) -> bool:
        try:
            self._ensure()
        except Exception:
            return False
        return True

    def configure(self, rate: float = 1.0, volume: float = 1.0) -> None:
        # pyttsx3 rate is words per minute, the default is 200.
        self._rate = int(max(0.1, min(rate, 3.0)) * 200) - 200
        self._volume = float(max(0.0, min(volume, 1.0)))
        if self._engine is not None:
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._voice_volume(self._volume))

    # -------------------------------------------------------------- speaking

    def voices(self) -> list[str]:
        try:
            return [v.name for v in self._ensure().getProperty("voices")]
        except Exception:
            return []

    def synthesize(self, text: str) -> np.ndarray | None:
        """pyttsx3 renders to a wav file, which we read back into a numpy array."""

        if not text.strip():
            return None
        try:
            engine = self._ensure()
        except Exception:
            return None

        handle, path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            return _read_wav(path)
        except Exception:
            return None
        finally:
            Path(path).unlink(missing_ok=True)

    def speak_blocking(self, text: str) -> None:
        try:
            engine = self._ensure()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    def close(self) -> None:
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.stop()
            self._engine = None


def _read_wav(path: str) -> np.ndarray | None:
    file = Path(path)
    if not file.exists() or file.stat().st_size <= 44:
        return None
    with wave.open(path, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        wav.getnchannels()
        width = wav.getsampwidth()

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        return None
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if dtype is np.uint8:
        return (data - 128.0) / 128.0
    return data / float(np.iinfo(dtype).max)
