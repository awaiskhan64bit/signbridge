"""Body landmark extraction.

SignBridge started with hands alone, which turns out to be the wrong unit. A
flat hand at the forehead and the same flat hand at the chest are different
signs, and nothing in a hand-only feature vector distinguishes them. The body is
the reference the sign is made against, so it has to be tracked too.

This module wraps MediaPipe pose detection in the same shape as
vision.hands.HandTracker: both the current Tasks API and the legacy Solutions API
are tried, failures are reported rather than swallowed, and callers never import
MediaPipe themselves.

Coordinates come back normalised to the image (x and y in 0..1, z relative to
the hips), which is the same space the hand landmarks arrive in, so the two can
be concatenated without rescaling.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: MediaPipe's pose model reports 33 joints.
POSE_LANDMARK_COUNT = 33

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


class PoseTrackerUnavailable(RuntimeError):
    pass


class PoseTracker:
    """Detects one body per frame. Not thread safe, call from one thread.

    Only one person is tracked on purpose. A second person in frame would
    overwrite the first's landmarks and make the body frame jump between
    signers, which the model reads as a sign that teleports.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        min_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        self._impl: str = ""
        self._session: Any = None
        self._mp: Any = None
        self._init_error: str | None = None
        self._last_stamp: int = -1
        self._warned = False

    # ------------------------------------------------------------- lifecycle

    def load(self) -> None:
        if self._session is not None:
            return
        self._init_error = None
        if not self.model_path.exists():
            self._init_error = (
                f"Pose model missing at {self.model_path}. "
                "Run: python -m signbridge.setup --models"
            )
            return
        reason = "no pose tracker implementation loaded"
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
        self._init_error = f"MediaPipe pose tracking could not start ({reason})"

    def _init_tasks(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        options_cls = getattr(vision, "PoseLandmarkerOptions", None)
        if options_cls is None:  # pragma: no cover - pre 1.1 layout
            from mediapipe.tasks import python as mp_python

            options_cls = mp_python.PoseLandmarkerOptions

        from mediapipe.tasks.python import BaseOptions

        running_mode = getattr(vision, "RunningMode", None)
        if running_mode is None:  # pragma: no cover - pre 1.1 layout
            from mediapipe.tasks import python as mp_python

            running_mode = mp_python.RunningMode

        options = options_cls(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=running_mode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._session = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    def _init_solutions(self) -> None:
        import mediapipe as mp

        self._session = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._mp = mp

    def close(self) -> None:
        if self._session is None:
            return
        # MediaPipe's close can raise on a session that never started, and a
        # failure to close during shutdown must not mask the real reason we are
        # shutting down.
        with contextlib.suppress(Exception):
            self._session.close()
        self._session = None

    @property
    def available(self) -> bool:
        self.load()
        return self._session is not None

    @property
    def error(self) -> str | None:
        return self._init_error

    # --------------------------------------------------------------- tracking

    def detect(self, image_rgb: np.ndarray, timestamp_ms: float) -> np.ndarray | None:
        """Return (33, 3) body landmarks for this frame, or None if not found.

        None means no body was visible, which is different from an all-zero
        result: zero is the encoding the model was trained for when a landmark is
        missing, so callers must not convert "no body" into zeros without also
        telling the model that the frame is unusable.
        """

        if self._session is None:
            self.load()
        if self._session is None:
            return None

        try:
            if self._impl == "tasks":
                return self._detect_tasks(image_rgb, timestamp_ms)
            return self._detect_solutions(image_rgb)
        except Exception as exc:
            # Same reasoning as the hand tracker: a swallowed exception looks
            # exactly like "no body in frame", which is how an extraction bug
            # hides for hours.
            if not self._warned:
                self._warned = True
                log.warning("Pose tracking failed, treating every frame as empty: %s", exc)
            log.debug("pose detection failed: %s", exc, exc_info=True)
            return None

    def _detect_tasks(self, image_rgb: np.ndarray, timestamp_ms: float) -> np.ndarray | None:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        stamp = int(max(timestamp_ms, self._last_stamp + 1))
        self._last_stamp = stamp
        result = self._session.detect_for_video(mp_image, stamp)
        marks = getattr(result, "pose_landmarks", None)
        if marks is None:
            marks = getattr(result, "landmarks", None)
        if not marks:
            return None
        return _as_points(marks[0])

    def _detect_solutions(self, image_rgb: np.ndarray) -> np.ndarray | None:
        result = self._session.process(image_rgb)
        marks = getattr(result, "pose_landmarks", None)
        if not marks:
            return None
        return _as_points(marks[0])


def _as_points(raw: Any) -> np.ndarray | None:
    """Coerce one body's landmarks into a (33, 3) float array.

    MediaPipe returns `NormalizedLandmark` objects whose x, y and z are
    attributes. Handing that list to `np.asarray` raises, and a caller that
    catches exceptions cannot tell that apart from "nobody in frame".
    """

    try:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.shape == (POSE_LANDMARK_COUNT, 3):
            return arr
    except (TypeError, ValueError):
        pass

    try:
        points = np.array(
            [[float(p.x), float(p.y), float(p.z)] for p in raw], dtype=np.float32
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return points if points.shape == (POSE_LANDMARK_COUNT, 3) else None