"""Piper neural text to speech.

Piper runs a small VITS model fully offline. Voice files are downloaded once by
`python -m signbridge.setup` and then read straight off disk, so the app never
touches the network after install.

Two Piper Python APIs are in circulation: the current `piper-tts` (1.3+, OHF-Voice)
and the archived `rhasspy/piper` 1.x line. Both are supported here so the app keeps
working whichever one pip resolved.
"""

from __future__ import annotations

import threading
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

from .base import TTSEngine

# lessac is the default because it is the best sounding voice that is not
# restricted to non-commercial use. See docs/licensing.md before shipping.
DEFAULT_VOICE = "en_US-lessac-medium"


def _load_piper():
    """Return (PiperVoice, synth_kind) for whichever Piper is installed."""

    try:  # current package, piper-tts 1.3+
        from piper import PiperVoice  # type: ignore

        return PiperVoice, "chunks"
    except Exception:
        pass
    try:  # archived rhasspy/piper line
        from piper import PiperVoice  # type: ignore

        return PiperVoice, "wave"
    except Exception as exc:  # pragma: no cover - depends on install
        raise RuntimeError("Piper is not installed. Run: pip install piper-tts") from exc


def _synthesis_config(rate: float):
    """Build a SynthesisConfig when the installed Piper has one.

    normalize_audio is turned off deliberately: the default peak-normalises every
    sentence separately, which sounds like the volume is jumping around.
    """

    try:
        from piper import SynthesisConfig
    except Exception:
        return None
    return SynthesisConfig(
        length_scale=1.0 / max(rate, 0.1),
        normalize_audio=False,
    )


class PiperEngine(TTSEngine):
    name = "Piper (neural)"

    def __init__(
        self,
        voices_dir: Path | None = None,
        voice: str = DEFAULT_VOICE,
        rate: float = 1.0,
    ) -> None:
        self.voices_dir = Path(voices_dir) if voices_dir else Path("voices")
        self.voice_name = voice
        self.rate = rate
        self._voice = None
        self._kind = "wave"
        self._lock = threading.Lock()
        self._sample_rate = 22_050
        self._load_error: str | None = None
        self._warming = False

    # ------------------------------------------------------------- lifecycle

    @property
    def model_path(self) -> Path:
        return self.voices_dir / f"{self.voice_name}.onnx"

    @property
    def config_path(self) -> Path:
        return self.voices_dir / f"{self.voice_name}.onnx.json"

    def available(self) -> bool:
        """Whether this engine can speak, without paying to load the voice.

        Loading a Piper voice reads a neural model off disk and builds an
        inference session, which takes seconds. Doing that here froze the entire
        UI, because engine selection runs on the UI thread: a frozen UI looks
        exactly like a stalled camera, no frames and no error to explain it. So
        this answers from the files and the import, and the weights are loaded
        later on the speech thread by `warm`.
        """

        if self._voice is not None:
            return True
        if not (self.model_path.exists() and self.config_path.exists()):
            return False
        try:
            _load_piper()
        except Exception as exc:
            self._load_error = f"Piper is not available: {exc}"
            return False
        return True

    def warm(self) -> None:
        """Load the voice on a background thread.

        Called after selection so the first sign is not delayed by a model load,
        and so that load never runs on the thread that draws the UI.
        """

        if self._voice is not None or self._warming:
            return
        self._warming = True

        def _work() -> None:
            try:
                self._load()
            finally:
                self._warming = False

        threading.Thread(target=_work, name="piper-warm", daemon=True).start()

    def _load(self):
        if self._voice is not None:
            return self._voice
        if not self.model_path.exists() or not self.config_path.exists():
            self._load_error = (
                f"Voice {self.voice_name} is not downloaded. "
                "Run: python -m signbridge.setup --voices"
            )
            return None
        try:
            piper_voice, kind = _load_piper()
            # Piper is not thread safe, synthesis is serialised in synthesize().
            self._voice = piper_voice.load(str(self.model_path), config_path=str(self.config_path))
            self._kind = kind
            self._sample_rate = int(getattr(self._voice, "sample_rate", 22_050))
            self._load_error = None
        except Exception as exc:
            self._load_error = f"Could not load voice {self.voice_name}: {exc}"
            self._voice = None
        return self._voice

    def set_voice(self, name: str) -> bool:
        if name == self.voice_name and self._voice is not None:
            return True
        self.close()
        self.voice_name = name
        return self._load() is not None

    def set_rate(self, rate: float) -> None:
        self.rate = max(0.1, min(rate, 3.0))

    def voices(self) -> list[str]:
        if not self.voices_dir.exists():
            return []
        return sorted(p.stem for p in self.voices_dir.glob("*.onnx"))

    @property
    def load_error(self) -> str | None:
        """Why this engine cannot load, or None when it can."""

        if self._voice is None:
            self._load()  # populates _load_error if it is still unset
        return None if self._voice is not None else self._load_error

    def resolve(self) -> str | None:
        """Substitute an installed voice when the chosen one is missing.

        A stale voice name in the settings file, left over from a previous
        install, would otherwise mean the app is silent with no obvious cause.
        Any installed voice beats no voice.
        """

        if self.model_path.exists() and self.config_path.exists():
            return None
        available = self.voices()
        if not available or self.voice_name in available:
            return None
        substituted = available[0]
        self.close()
        self.voice_name = substituted
        return substituted

    @property
    def describe(self) -> str:
        if self.load_error:
            return f"{self.name}: unavailable"
        return f"{self.name} / {self.voice_name}"

    # -------------------------------------------------------------- synthesis

    def synthesize(self, text: str) -> np.ndarray | None:
        if not text.strip():
            return None
        voice = self._load()
        if voice is None:
            return None
        try:
            with self._lock:
                if self._kind == "chunks":
                    audio = self._synthesize_chunks(voice, text)
                else:
                    audio = self._synthesize_wave(voice, text)
        except Exception:
            return None
        return audio

    def _synthesize_chunks(self, voice, text: str) -> np.ndarray | None:
        """piper-tts 1.3+ yields AudioChunk objects holding float samples."""

        config = _synthesis_config(self.rate)
        chunks = []
        rate = self._sample_rate
        produced = (
            voice.synthesize(text)
            if config is None
            else voice.synthesize(text, syn_config=config)
        )
        for chunk in produced:
            rate = int(getattr(chunk, "sample_rate", rate))
            arr = getattr(chunk, "audio_float_array", None)
            if arr is not None:
                chunks.append(np.asarray(arr, dtype=np.float32))
                continue
            raw = getattr(chunk, "audio_int16_bytes", None)
            if raw is not None:
                chunks.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
        if not chunks:
            return None
        self.sample_rate = rate
        return np.concatenate(chunks)

    def _synthesize_wave(self, voice, text: str) -> np.ndarray | None:
        """Original piper-tts writes straight into a wave.Wave_write."""

        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            voice.synthesize(text, wav)
        buffer.seek(0)
        with wave.open(buffer, "rb") as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            return None
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if dtype is np.uint8:
            data = (data - 128.0) / 128.0
        else:
            data /= float(np.iinfo(dtype).max)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        self.sample_rate = rate
        return data

    def close(self) -> None:
        self._voice = None
