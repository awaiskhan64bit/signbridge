"""The realtime pipeline.

One thread drives the whole loop:

    capture -> hand landmarks -> feature encode -> sliding window -> recognizer
            -> temporal decode -> phrase assembly -> speech

Nothing here touches Qt. The UI supplies callbacks and marshals them onto the
GUI thread, which keeps the heavy work off the event loop and makes the pipeline
runnable headless from the tests.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Settings
from ..ml.base import Recognizer
from ..ml.collector import Collector
from ..ml.features import encode_frame, encode_sequence
from ..speech.service import SpeechService, build_engine
from ..vision.camera import CaptureWorker
from ..vision.hands import HandTracker
from ..vision.overlay import draw_hands
from .decoder import DecodedEvent, Event, SignDecoder, window_has_hand
from .types import Frame, SignPrediction, Source

log = logging.getLogger(__name__)

FrameCallback = Callable[[np.ndarray | None, "EngineState"], None]
EventCallback = Callable[[DecodedEvent], None]

EMPTY = SignPrediction(label="", confidence=0.0)


@dataclass(slots=True)
class EngineState:
    """Read only telemetry the UI renders. Copy with :meth:`snapshot`."""

    fps: float = 0.0
    latency_ms: float = 0.0
    hands: int = 0
    tracking: bool = False
    recognizer: str = ""
    speech: str = ""
    source: Source = Source.CAMERA
    last_label: str = ""
    confidence: float = 0.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    window_fill: float = 0.0
    error: str | None = None

    def snapshot(self) -> EngineState:
        return EngineState(
            fps=self.fps,
            latency_ms=self.latency_ms,
            hands=self.hands,
            tracking=self.tracking,
            recognizer=self.recognizer,
            speech=self.speech,
            source=self.source,
            last_label=self.last_label,
            confidence=self.confidence,
            alternatives=list(self.alternatives),
            window_fill=self.window_fill,
            error=self.error,
        )


class SignEngine:
    """Owns every long lived resource and the thread that pumps frames through them."""

    def __init__(
        self,
        settings: Settings,
        recognizer: Recognizer,
        *,
        on_frame: FrameCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.settings = settings
        self.recognizer = recognizer
        self.on_frame = on_frame
        self.on_event = on_event

        self.state = EngineState(
            recognizer=recognizer.status(),
            source=_source_of(settings),
        )
        self.decoder = SignDecoder(
            signs_per_phrase=settings.signs_per_phrase,
            phrase_pause_ms=900.0,
        )
        self.speech = SpeechService(
            build_engine(
                settings.tts_engine,
                settings.voices_dir,
                settings.tts_voice,
                settings.tts_rate,
            )
        )
        self.speech.player.set_volume(settings.tts_volume)
        if self.speech.note.degraded:
            # Surface it immediately rather than leaving the user wondering why
            # the app is silent.
            self.state.error = self.speech.note.message

        self.tracker: HandTracker | None = None
        self.capture = self._build_capture()
        self._collector = Collector(settings.data_dir / "collected")
        self._window: deque[np.ndarray] = deque(maxlen=max(8, settings.window_frames))
        self._stride = max(1, settings.window_frames // 12)
        self._since_predict = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps_window: deque[float] = deque(maxlen=30)
        self._lock = threading.RLock()

    # -------------------------------------------------------------- lifecycle

    def _build_capture(self) -> CaptureWorker:
        s = self.settings
        worker = CaptureWorker(
            source=_source_of(s),
            camera_index=s.camera_index,
            width=s.frame_width,
            height=s.frame_height,
            fps=s.target_fps,
        )
        if s.video_path:
            worker.set_video(s.video_path)
        return worker

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.speech.start()
        self._load_tracker()
        self.capture.start()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.speech.stop()
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None

    def _load_tracker(self) -> None:
        self.tracker = None
        if self.state.source is Source.SIMULATED:
            return  # synthetic frames arrive with their own hands attached
        model = Path(self.settings.models_dir) / "hand_landmarker.task"
        tracker = HandTracker(model, min_confidence=self.settings.hand_confidence)
        tracker.load()
        if tracker.available:
            self.tracker = tracker
        else:
            self.state.error = tracker.error

    # --------------------------------------------------------------- commands

    def switch_source(self, source: Source, video_path: str | Path | None = None) -> None:
        """Swap input without a restart, used by the source selector in the UI."""

        was_running = self._thread is not None
        self.capture.stop()
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None

        self.settings.source = source.value
        if video_path:
            self.settings.video_path = str(video_path)
        self.state.source = source
        self.state.error = None
        self.decoder.reset()
        self._window.clear()

        self.capture = self._build_capture()
        self._load_tracker()
        if was_running:
            self.capture.start()

    def set_recognizer(self, recognizer: Recognizer) -> None:
        with self._lock:
            self.recognizer = recognizer
            self.decoder.reset()
            self._window.clear()
        self.state.recognizer = recognizer.status()

    def set_settings(self, settings: Settings) -> None:
        """Apply changed settings without dropping the running pipeline."""

        self.settings = settings
        self.speech.player.set_volume(settings.tts_volume)
        self.decoder.signs_per_phrase = settings.signs_per_phrase
        self.decoder.phrase_pause_ms = settings.phrase_pause_ms
        current = self.speech.engine.name.lower().split()[0]
        if settings.tts_engine != current:
            self.speech.set_engine(
                build_engine(
                    settings.tts_engine,
                    settings.voices_dir,
                    settings.tts_voice,
                    settings.tts_rate,
                )
            )
            self.state.error = self.speech.note.message or None
        else:
            configure = getattr(self.speech.engine, "set_rate", None)
            if callable(configure):
                configure(settings.tts_rate)

    def flush_phrase(self) -> None:
        for event in self.decoder.flush(time.perf_counter() * 1000.0):
            self._handle_event(event)

    def clear_phrase(self) -> None:
        self.decoder.reset()
        self._window.clear()

    def repeat_last(self) -> None:
        if self.speech.last_spoken:
            self.speech.say(self.speech.last_spoken)

    # ------------------------------------------------------------- recording

    @property
    def collecting(self) -> bool:
        return self._collector.is_recording

    @property
    def collector(self) -> Collector:
        return self._collector

    def start_recording(self, label: str) -> bool:
        """Begin capturing a labelled clip of the user's own signing."""

        return self._collector.start(label)

    def stop_recording(self) -> str | None:
        """Store the clip. Returns its path, or None if there was no real hand."""

        path = self._collector.stop()
        return str(path) if path else None

    def cancel_recording(self) -> None:
        self._collector.cancel()

    def _feed_collector(self, features: np.ndarray) -> None:
        if self._collector.is_recording:
            self._collector.push(features)

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        reported = None
        while not self._stop.is_set():
            frame = self.capture.read(timeout=0.5)
            if frame is None:
                if self.capture.error and self.capture.error != reported:
                    reported = self.capture.error
                    self.state.error = reported
                    self.state.fps = 0.0
                    self.state.last_label = ""
                    self.state.tracking = False
                    self._emit(None)
                continue
            reported = None
            try:
                self._process(frame)
            except Exception as exc:  # one bad frame must not kill the loop
                log.debug("frame failed: %s", exc, exc_info=True)

    def _process(self, frame: Frame) -> None:
        started = time.perf_counter()

        if frame.image is not None and self.settings.mirror:
            frame.image = np.ascontiguousarray(frame.image[:, ::-1])

        if not frame.hands and self.tracker is not None and frame.image is not None:
            import cv2

            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            frame.hands = self.tracker.detect(rgb, frame.timestamp_ms)

        encoded = encode_frame(frame.hands)
        self._window.append(encoded)
        self._feed_collector(encoded)
        self.state.hands = len(frame.hands)
        self.state.tracking = bool(frame.hands)
        self.state.window_fill = len(self._window) / max(1, self._window.maxlen)

        self._since_predict += 1
        if self._since_predict >= self._stride:
            self._since_predict = 0
            self._predict(frame.timestamp_ms)

        image = frame.image
        if image is not None and self.settings.show_skeleton:
            # The frame was already flipped above, so the skeleton is drawn as is.
            draw_hands(image, frame.hands, mirror=False)

        self.state.latency_ms = (time.perf_counter() - started) * 1000.0
        self._fps_window.append(time.perf_counter())
        if len(self._fps_window) > 1:
            span = self._fps_window[-1] - self._fps_window[0]
            self.state.fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0
        self.state.speech = self.speech.status()
        self._emit(image)

    def _predict(self, now_ms: float) -> None:
        window = encode_sequence(list(self._window), self._window.maxlen or 48)

        if not window_has_hand(window):
            # Hand left the frame. Feed an empty prediction so the decoder sees the
            # gap and any pending phrase can flush on its pause timer.
            self.state.last_label = ""
            self.state.confidence = 0.0
            self.state.alternatives = []
            for event in self.decoder.push(EMPTY, now_ms):
                self._handle_event(event)
            return

        with self._lock:
            recognizer = self.recognizer
        prediction = recognizer.predict(window)

        self.state.last_label = prediction.label
        self.state.confidence = prediction.confidence
        self.state.alternatives = prediction.alternatives

        for event in self.decoder.push(prediction, now_ms):
            self._handle_event(event)

    def _handle_event(self, event: DecodedEvent) -> None:
        if event.kind is Event.PHRASE and event.text and (
            self.settings.speak_automatically and self.settings.tts_engine != "off"
        ):
            self.speech.say(event.text)
        if self.on_event is not None:
            self.on_event(event)

    def _emit(self, image: np.ndarray | None) -> None:
        if self.on_frame is not None:
            self.on_frame(image, self.state)


def _source_of(settings: Settings) -> Source:
    try:
        return Source(settings.source)
    except ValueError:
        return Source.CAMERA
