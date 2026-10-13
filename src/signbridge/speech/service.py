"""Audio playback and the speech worker thread.

`AudioPlayer` owns a sounddevice output stream. `SpeechService` sits on top of it
with a background thread so synthesis, which takes a moment even for a neural
voice, never stalls the camera pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .base import NullEngine, TTSEngine
from .piper import PiperEngine
from .sapi import SAPIEngine

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EngineNote:
    """Why the running engine is not the one that was asked for."""

    degraded_from: str = ""
    message: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_from)


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

    def __init__(self, engine: TTSEngine | None = None, note: EngineNote | None = None) -> None:
        self.engine: TTSEngine = engine or NullEngine()
        self.note = note or getattr(self.engine, "note", None) or EngineNote()
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

    def set_engine(self, engine: TTSEngine, note: EngineNote | None = None) -> None:
        self.clear()
        self.engine.close()
        self.engine = engine
        if note is not None:
            self.note = note

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
    note: EngineNote | None = None,
) -> TTSEngine:
    """Build an engine, degrading rather than failing.

    Asking for Piper when its voice has not been downloaded used to produce a
    silent engine: every synthesis returned None and the app simply said nothing,
    with no clue why. An unavailable engine now falls back to the one built into
    Windows, and the reason is recorded on `note` so the UI can say so out loud.
    """

    kind = (kind or "off").lower()
    note = note if note is not None else EngineNote()
    engine = _select(kind, voices_dir, voice, rate, note)
    # The note rides along with the engine so a caller cannot forget to pass it,
    # which is how the reason a fallback happened used to get dropped.
    with contextlib.suppress(AttributeError):
        engine.note = note
    return engine


def _select(kind: str, voices_dir, voice: str | None, rate: float, note: EngineNote) -> TTSEngine:
    if kind == "piper":
        engine = (
            PiperEngine(voices_dir, voice, rate)
            if voice
            else PiperEngine(voices_dir, rate=rate)
        )
        substituted = engine.resolve()
        if substituted:
            note.message = (
                f"Voice {engine.voice_name!r} was not downloaded, so "
                f"{substituted} is being used instead."
            )
        if engine.available():
            return engine
        reason = engine.load_error or "the Piper voice is not available"
        sapi = SAPIEngine()
        if sapi.available():
            sapi.configure(rate=rate)
            note.degraded_from = "piper"
            note.message = f"{reason}. {note.message} Windows SAPI is speaking.".strip()
            log.warning(note.message)
            return sapi
        note.degraded_from = "piper"
        note.message = f"{reason}, and Windows SAPI is unavailable. Speech is off."
        log.error(note.message)
        return NullEngine()

    if kind == "sapi":
        sapi = SAPIEngine()
        if sapi.available():
            sapi.configure(rate=rate)
            return sapi
        note.degraded_from = "sapi"
        note.message = "Windows SAPI is unavailable. Speech is off."
        log.error(note.message)
        return NullEngine()

    return NullEngine()
