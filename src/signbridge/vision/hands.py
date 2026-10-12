"""Hand landmark extraction.

Wraps MediaPipe so the rest of the app never imports it directly. Two APIs exist
across MediaPipe releases and both are supported here:

  * the newer Tasks API, ``HandLandmarker`` from ``mediapipe.tasks.python``, which
    is what current pip builds ship with
  * the legacy Solutions API, ``mp.solutions.hands``

Landmarks come back normalised to the image (x and y in 0..1, z roughly in the
same scale relative to the wrist), which is exactly the representation the model
was trained on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import LANDMARK_COUNT, Hand, Handedness

log = logging.getLogger(__name__)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


class HandTrackerUnavailable(RuntimeError):
    pass


class HandTracker:
    """Detects up to two hands per frame. Not thread safe, call from one thread."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_hands: int = 2,
        min_confidence: float = 0.55,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_hands = max_hands
        self.min_confidence = min_confidence
        self._impl: str = ""
        self._session: Any = None
        self._mp: Any = None
        self._init_error: str | None = None

    # ------------------------------------------------------------- lifecycle

    def load(self) -> None:
        if self._session is not None:
            return
        self._init_error = None
        if not self.model_path.exists():
            self._init_error = (
                f"Hand model missing at {self.model_path}. "
                "Run: python -m signbridge.setup --models"
            )
            return
        reason = "no tracker implementation loaded"
        try:
            self._init_tasks()
            self._impl = "tasks"
            return
        except Exception as exc:
            reason = f"Tasks API: {exc}"
            log.debug("Tasks API unavailable: %s", exc)
        try:
            self._init_solutions()
            self._impl = "solutions"
            return
        except Exception as exc:
            reason = f"Solutions API: {exc}"
            log.debug("Solutions API unavailable: %s", exc)
        # Both attempts are reported together. Python unbinds the except
        # variable at the end of the block, so it has to be captured here.
        self._init_error = f"MediaPipe hand tracking could not start ({reason})"

    def _init_tasks(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = mp_python.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_python.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._session = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def _init_solutions(self) -> None:
        import mediapipe as mp

        self._session = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._mp = mp

    def close(self) -> None:
        if self._session is None:
            return
        try:
            if self._impl == "tasks":
                self._session.close()
            else:
                self._session.close()
        except Exception:
            pass
        self._session = None

    @property
    def available(self) -> bool:
        self.load()
        return self._session is not None

    @property
    def error(self) -> str | None:
        return self._init_error

    # --------------------------------------------------------------- tracking

    def detect(self, image_rgb: np.ndarray, timestamp_ms: float) -> list[Hand]:
        """Run the tracker on an RGB frame and return hands, largest first."""

        if self._session is None:
            self.load()
        if self._session is None:
            return []

        try:
            if self._impl == "tasks":
                return self._detect_tasks(image_rgb, timestamp_ms)
            return self._detect_solutions(image_rgb)
        except Exception as exc:
            log.debug("detection failed: %s", exc)
            return []

    def _detect_tasks(self, image_rgb: np.ndarray, timestamp_ms: float) -> list[Hand]:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        # MediaPipe VIDEO mode wants a strictly increasing timestamp.
        stamp = int(max(timestamp_ms, self._last_stamp + 1))
        self._last_stamp = stamp
        result = self._session.detect_for_video(mp_image, stamp)
        return _hands_from_tasks(result)

    _last_stamp: int = -1

    def _detect_solutions(self, image_rgb: np.ndarray) -> list[Hand]:
        result = self._session.process(image_rgb)
        return _hands_from_solutions(result)


def _hands_from_tasks(result: Any) -> list[Hand]:
    hands: list[Hand] = []
    landmarks = getattr(result, "landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []
    for i, points in enumerate(landmarks):
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape != (LANDMARK_COUNT, 3):
            continue
        label = Handedness.UNKNOWN
        score = 1.0
        if i < len(handedness) and handedness[i]:
            cat = handedness[i][0] if isinstance(handedness[i], (list, tuple)) else handedness[i]
            label = _handedness(cat.category_name, cat.score)
            score = float(cat.score)
        hands.append(Hand(landmarks=arr, handedness=label, score=score))
    hands.sort(key=_prominence, reverse=True)
    return hands


def _hands_from_solutions(result: Any) -> list[Hand]:
    hands: list[Hand] = []
    multi = getattr(result, "multi_hand_landmarks", None) or []
    handedness = getattr(result, "multi_handedness", None) or []
    for i, points in enumerate(multi):
        arr = np.asarray([(p.x, p.y, p.z) for p in points], dtype=np.float32)
        if arr.shape != (LANDMARK_COUNT, 3):
            continue
        label, score = Handedness.UNKNOWN, 1.0
        if i < len(handedness):
            cat = handedness[i].classification[0]
            label = _handedness(cat.label, cat.score)
            score = float(cat.score)
        hands.append(Hand(landmarks=arr, handedness=label, score=score))
    hands.sort(key=_prominence, reverse=True)
    return hands


def _handedness(name: str, score: float) -> Handedness:
    if name and name.lower().startswith("left"):
        return Handedness.LEFT
    if name and name.lower().startswith("right"):
        return Handedness.RIGHT
    return Handedness.UNKNOWN


def _prominence(hand: Hand) -> float:
    """Bigger, more central hands win when both are in frame."""

    pts = hand.landmarks
    span = float(pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)).sum()
    centrality = float(1.0 - np.linalg.norm(pts[0, :2] - 0.5))
    return span * 0.5 + centrality * 0.5 * hand.score
