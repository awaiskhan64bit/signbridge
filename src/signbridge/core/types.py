"""Shared dataclasses and enums used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# 21 landmarks per hand, 3 axes (x, y, z) each.
LANDMARK_COUNT = 21
HAND_DIMS = LANDMARK_COUNT * 3


class Handedness(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


class Source(str, Enum):
    CAMERA = "camera"
    VIDEO = "video"
    SIMULATED = "simulated"


@dataclass(slots=True)
class Hand:
    """One detected hand in a single video frame."""

    landmarks: np.ndarray  # (21, 3) normalized coords in [0, 1] / relative depth
    handedness: Handedness = Handedness.UNKNOWN
    score: float = 1.0

    @property
    def fingers_up(self) -> tuple[bool, ...]:
        """Cheap heuristic used for gestures and for the HUD readout.

        A finger counts as extended when its tip sits further from the wrist
        than its own middle joint, measured along the index finger axis as a
        robust reference that does not depend on which hand is showing.
        """
        wrist = self.landmarks[0]
        # Index MCP (5) to wrist defines the palm direction proxy.
        palm_ref = np.linalg.norm(self.landmarks[5] - wrist) + 1e-6
        out: list[bool] = []
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            tip_d = float(np.linalg.norm(self.landmarks[tip] - wrist))
            pip_d = float(np.linalg.norm(self.landmarks[pip] - wrist))
            out.append(tip_d > pip_d and tip_d > palm_ref * 0.55)
        return tuple(out)


@dataclass(slots=True)
class Frame:
    """A processed video frame plus whatever the vision layer found in it."""

    image: np.ndarray | None = None
    hands: list[Hand] = field(default_factory=list)
    #: (33, 3) body landmarks, or None when no body was visible. Kept separate
    #: from `hands` because the two are detected by separate MediaPipe models
    #: and either can succeed while the other fails.
    pose: np.ndarray | None = None
    index: int = 0
    timestamp_ms: float = 0.0

    @property
    def primary(self) -> Hand | None:
        return self.hands[0] if self.hands else None

    def by_handedness(self, wanted: Handedness) -> np.ndarray | None:
        """The landmarks of the hand MediaPipe labelled `wanted`, if present."""

        for hand in self.hands:
            if hand.handedness is wanted:
                return hand.landmarks
        return None


@dataclass(slots=True)
class SignPrediction:
    """One classifier decision for the current temporal window."""

    label: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    @property
    def display(self) -> str:
        return self.label.replace("_", " ").title()


@dataclass(slots=True)
class SpeechChunk:
    """A phrase the pipeline decided is finished and wants spoken."""

    text: str
    signs: list[str] = field(default_factory=list)
    started_at_ms: float = 0.0
