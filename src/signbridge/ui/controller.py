"""QML facing controller.

The engine runs on its own thread and calls plain callbacks. This class is the
boundary: it converts those callbacks into Qt signals so QML can bind to them,
and turns QML invokable calls into engine commands. Nothing expensive runs here,
because everything on this object lives on the GUI thread.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtQuick import QQuickImageProvider

from ..config import Settings
from ..core.decoder import DecodedEvent, Event
from ..core.engine import EngineState, SignEngine
from ..core.types import Source
from ..ml.base import Glossary, Recognizer
from ..ml.gestures import GestureRecognizer
from ..ml.neural import NeuralRecognizer

#: Frames are downscaled to this width before crossing into the GUI thread.
PREVIEW_WIDTH = 960


class FrameProvider(QQuickImageProvider):
    """Hands the latest annotated frame to QML through an image:// URL.

    Going through a provider rather than assigning a QImage to `Image.source`
    keeps the conversion on the GUI thread where it belongs, and avoids relying
    on QImage to url coercion, which is not dependable across Qt versions.
    """

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.Pixmap)
        self._pixmap: QPixmap | None = None

    def publish(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap

    def requestPixmap(self, request_id: str, size, requested_size):
        """Qt calls this, not requestImage, because the provider is a Pixmap one."""

        if self._pixmap is None:
            return QPixmap()
        return self._pixmap


class Controller(QObject):
    """Everything the UI can read or call."""

    # Live telemetry
    frameReady = Signal()
    stateChanged = Signal()
    phraseChanged = Signal()
    historyChanged = Signal()
    errorRaised = Signal(str)
    spokeText = Signal(str)

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._history: list[dict] = []
        self._phrase = ""
        self._frame_id = 0
        self._state = EngineState()
        self._recognizer = self._build_recognizer()
        self.provider = FrameProvider()

        self.engine = SignEngine(
            settings,
            self._recognizer,
            on_frame=self._on_frame,
            on_event=self._on_event,
        )
        self.engine.speech.on_spoken = self._on_spoken

    @Property(int, notify=frameReady)
    def frameId(self) -> int:
        """Bumped whenever a new frame is published."""

        return self._frame_id

    # ------------------------------------------------------------ recognizer

    def _build_recognizer(self) -> Recognizer:
        """Prefer a trained model, fall back to the built in gesture rules."""

        model = Path(self.settings.models_dir) / "sign_classifier.onnx"
        neural = NeuralRecognizer.try_load(model, self.settings.min_sign_confidence)
        if neural is not None and neural.ready:
            return neural
        return GestureRecognizer(self.settings.min_sign_confidence)

    @Property(str, notify=stateChanged)
    def recognizerLabel(self) -> str:
        return self._state.recognizer or self._recognizer.status()

    @Property(str, notify=stateChanged)
    def recognizerKind(self) -> str:
        return "neural" if isinstance(self._recognizer, NeuralRecognizer) else "rules"

    # ------------------------------------------------------------- telemetry

    def _on_frame(self, image: np.ndarray | None, state: EngineState) -> None:
        """Called on the engine thread. QPixmap is built here so the GUI thread
        only ever touches Qt objects from inside a slot."""

        pixmap = None
        if image is not None:
            import cv2

            h, w = image.shape[:2]
            if w > PREVIEW_WIDTH:
                scale = PREVIEW_WIDTH / w
                image = cv2.resize(
                    image, (PREVIEW_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA
                )
                h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # The QImage copy detaches from the numpy buffer, which the next
            # frame reuses, so the pixmap is never left pointing at stale data.
            pixmap = QPixmap.fromImage(_to_qimage(rgb))

        self._state = state
        if pixmap is not None:
            self.provider.publish(pixmap)
            self._frame_id += 1
            self.frameReady.emit()
        self.stateChanged.emit()

    def _on_event(self, event: DecodedEvent) -> None:
        if event.kind is Event.SIGN and event.label:
            # Keep the running phrase visible in the UI as it builds up.
            self._phrase = self.engine.decoder.phrase
            self.phraseChanged.emit()
        elif event.kind is Event.PHRASE and event.text:
            self._history.insert(
                0,
                {
                    "text": event.text,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "spoken": self.settings.tts_engine != "off",
                },
            )
            del self._history[40:]
            self._phrase = ""
            self.phraseChanged.emit()
            self.historyChanged.emit()

    def _on_spoken(self, text: str) -> None:
        self.spokeText.emit(text)

    # ------------------------------------------------------------ properties

    @Property(float, notify=stateChanged)
    def fps(self) -> float:
        return self._state.fps

    @Property(float, notify=stateChanged)
    def latency(self) -> float:
        return self._state.latency_ms

    @Property(int, notify=stateChanged)
    def handCount(self) -> int:
        return self._state.hands

    @Property(bool, notify=stateChanged)
    def tracking(self) -> bool:
        return self._state.tracking

    @Property(float, notify=stateChanged)
    def windowFill(self) -> float:
        return self._state.window_fill

    @Property(str, notify=stateChanged)
    def speechLabel(self) -> str:
        return self._state.speech

    @Property(str, notify=stateChanged)
    def sourceLabel(self) -> str:
        return {
            Source.CAMERA: f"Camera {self.settings.camera_index}",
            Source.VIDEO: self.settings.video_path.rsplit("/", 1)[-1] or "Clip",
            Source.SIMULATED: "Simulated signer",
        }.get(self._state.source, "Camera")

    @Property(str, notify=stateChanged)
    def errorText(self) -> str:
        return self._state.error or ""

    @Property(bool, notify=stateChanged)
    def running(self) -> bool:
        return self.engine._thread is not None

    @Property(str, notify=stateChanged)
    def pendingLabel(self) -> str:
        label = self._state.last_label
        return Glossary.display(label) if label else ""

    @Property(str, notify=stateChanged)
    def pendingSpeech(self) -> str:
        label = self._state.last_label
        return Glossary.say(label) if label else ""

    @Property(float, notify=stateChanged)
    def confidence(self) -> float:
        return self._state.confidence

    @Property("QVariantList", notify=stateChanged)
    def alternatives(self) -> list[dict]:
        return [
            {"label": Glossary.display(label), "score": float(score)}
            for label, score in self._state.alternatives
            if Glossary.is_spoken(label)
        ]

    @Property(str, notify=phraseChanged)
    def phrase(self) -> str:
        return self.engine.decoder.phrase

    @Property("QVariantList", notify=phraseChanged)
    def phraseSigns(self) -> list[dict]:
        return [{"text": Glossary.display(s)} for s in self.engine.decoder.phrase_signs]

    @Property("QVariantList", notify=historyChanged)
    def history(self) -> list[dict]:
        return list(self._history)

    # -------------------------------------------------------------- settings

    @Property(str, notify=stateChanged)
    def ttsEngine(self) -> str:
        return self.settings.tts_engine

    @Property(str, notify=stateChanged)
    def ttsVoice(self) -> str:
        return self.settings.tts_voice

    @Property("QVariantList", notify=stateChanged)
    def voiceOptions(self) -> list[str]:
        engine = self.engine.speech.engine
        voices = engine.voices()
        return voices or [self.settings.tts_voice]

    @Property(float, notify=stateChanged)
    def ttsRate(self) -> float:
        return self.settings.tts_rate

    @Property(float, notify=stateChanged)
    def ttsVolume(self) -> float:
        return self.settings.tts_volume

    @Property(bool, notify=stateChanged)
    def autoSpeak(self) -> bool:
        return self.settings.speak_automatically

    @Property(bool, notify=stateChanged)
    def mirrored(self) -> bool:
        return self.settings.mirror

    @Property(bool, notify=stateChanged)
    def showSkeleton(self) -> bool:
        return self.settings.show_skeleton

    @Property(float, notify=stateChanged)
    def minConfidence(self) -> float:
        return self.settings.min_sign_confidence

    # ----------------------------------------------------------------- slots

    @Slot()
    def toggleRunning(self) -> None:
        if self.engine._thread is not None:
            self.engine.stop()
        else:
            self.engine.start()
        self.stateChanged.emit()

    @Slot(str)
    def setSource(self, name: str) -> None:
        try:
            source = Source(name)
        except ValueError:
            return
        self.engine.switch_source(source)
        self.settings.save()
        self.stateChanged.emit()

    @Slot()
    def flushPhrase(self) -> None:
        self.engine.flush_phrase()

    @Slot()
    def clearPhrase(self) -> None:
        self.engine.clear_phrase()
        self._phrase = ""
        self.phraseChanged.emit()

    @Slot()
    def repeatLast(self) -> None:
        self.engine.repeat_last()

    @Slot()
    def clearHistory(self) -> None:
        self._history.clear()
        self.historyChanged.emit()

    @Slot(str)
    def setTtsEngine(self, engine: str) -> None:
        self.settings.tts_engine = engine
        self.engine.set_settings(self.settings)
        self.settings.save()
        self.stateChanged.emit()

    @Slot(str)
    def setVoice(self, voice: str) -> None:
        self.settings.tts_voice = voice
        configure = getattr(self.engine.speech.engine, "set_voice", None)
        if callable(configure):
            configure(voice)
        self.settings.save()
        self.stateChanged.emit()

    @Slot(float)
    def setRate(self, rate: float) -> None:
        self.settings.tts_rate = rate
        self.engine.set_settings(self.settings)
        self.settings.save()

    @Slot(float)
    def setVolume(self, volume: float) -> None:
        self.settings.tts_volume = volume
        self.engine.speech.player.set_volume(volume)
        self.settings.save()

    @Slot(bool)
    def setAutoSpeak(self, on: bool) -> None:
        self.settings.speak_automatically = on
        self.settings.save()

    @Slot(bool)
    def setMirrored(self, on: bool) -> None:
        self.settings.mirror = on
        self.settings.save()

    @Slot(bool)
    def setSkeleton(self, on: bool) -> None:
        self.settings.show_skeleton = on
        self.settings.save()

    @Slot(float)
    def setMinConfidence(self, value: float) -> None:
        self.settings.min_sign_confidence = value
        self.engine.set_recognizer(
            self._recognizer
            if isinstance(self._recognizer, GestureRecognizer)
            else NeuralRecognizer.try_load(
                Path(self.settings.models_dir) / "sign_classifier.onnx", value
            )
        )
        self.settings.save()

    @Slot(str)
    def setVideoPath(self, path: str) -> None:
        if not path:
            return
        self.engine.switch_source(Source.VIDEO, path)
        self.settings.save()
        self.stateChanged.emit()

    @Slot(str)
    def previewPhrase(self, text: str) -> None:
        """Speak something without a sign, used by the voice tester."""

        self.engine.speech.say(text)

    @Slot(result=str)
    def settingsPath(self) -> str:
        return str(Settings.path())

    @Slot()
    def reloadRecognizer(self) -> None:
        self._recognizer = self._build_recognizer()
        self.engine.set_recognizer(self._recognizer)
        self.stateChanged.emit()


def _to_qimage(rgb: np.ndarray) -> QImage:
    """Wrap an RGB numpy array in a QImage that owns its data."""

    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
