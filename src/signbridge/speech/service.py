"""Audio playback and the speech worker thread.

`AudioPlayer` owns a sounddevice output stream. `SpeechService` sits on top of it
with a background thread so synthesis, which takes a moment even for a neural
voice, never stalls the camera pipeline.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .base import NullEngine, TTSEngine
from .piper import PiperEngine
from .sapi import SAPIEngine


@dataclass(slots=True)
class SpeechJob:
    text: str
    queued_at: float = 0.0


class AudioPlayer:
    """Thin wrapper over sounddevice with lazy stream creation and duck typing."""

    def __init__(self) -> None:
        self._sd = None
        self._lock = threading.Lock()
        self._volume = 1.0

    @property
    def available(self) -> bool:
        try:
            import sounddevice as sd

            sd.query_devices()
            return True
        except Exception:
            return False

    def _stream(self, sample_rate: int):
        import sounddevice as sd

        return sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            latency="low",
        )

    def play(self, audio: np.ndarray, sample_rate: int, volume: float | None = None) -> None:
        if audio is None or audio.size == 0:
            return
        gain = self._volume if volume is None else volume
        data = np.clip(np.asarray(audio, dtype=np.float32) * gain, -1.0, 1.0)
        with self._lock:
            try:

                with self._stream(sample_rate) as stream:
                    stream.write(data.reshape(-1, 1))
            except Exception:
                # No audio device (headless, RDP, device busy). Losing audio is
                # preferable to taking the whole app down.
                pass

    def set_volume(self, volume: float) -> None:
        self._volume = float(max(0.0, min(volume, 1.0)))


class SpeechService:
    """Queues phrases, synthesises them in order, plays them without blocking."""

    def __init__(self, engine: TTSEngine | None = None) -> None:
        self.engine: TTSEngine = engine or NullEngine()
        self.player = AudioPlayer()
        self._queue: queue.Queue[SpeechJob | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_spoken: str = ""
        self.spoken_count = 0
        self.last_error: str | None = None
        self.on_spoken = None  # optional callback(text)

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------ api

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._queue.put(SpeechJob(text=text, queued_at=time.perf_counter()))

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def set_engine(self, engine: TTSEngine) -> None:
        self.clear()
        self.engine.close()
        self.engine = engine

    @property
    def busy(self) -> bool:
        return not self._queue.empty()

    def status(self) -> str:
        return self.engine.describe()

    # ----------------------------------------------------------------- loop

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                audio = self.engine.synthesize(job.text)
                if audio is not None and audio.size:
                    self.player.play(audio, self.engine.sample_rate)
                    self.last_spoken = job.text
                    self.spoken_count += 1
                    self.last_error = None
                    if self.on_spoken is not None:
                        self.on_spoken(job.text)
                else:
                    self.last_error = "no audio produced"
            except Exception as exc:
                self.last_error = str(exc)


def build_engine(
    kind: str,
    voices_dir=None,
    voice: str | None = None,
    rate: float = 1.0,
) -> TTSEngine:
    """Factory used by the controller so engine switching stays declarative."""

    kind = (kind or "off").lower()
    if kind == "piper":
        return PiperEngine(voices_dir, voice, rate) if voice else PiperEngine(voices_dir, rate=rate)
    if kind == "sapi":
        return SAPIEngine()
    return NullEngine()
