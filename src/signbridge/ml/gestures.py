"""Rule based gesture recognizer.

Not machine learning, and not pretending to be. It exists so the app has a
working vocabulary before any model has been trained, which lets the pipeline,
HUD and speech be verified end to end.

It reads the same feature windows the neural recognizer does and decides from
average finger curl plus how far the thumb reaches.

These six are 100% accurate by construction rather than by skill, and that
distinction matters when comparing against the trained model. Each rule reads
off hand shape, the hand rig in `vision/simulated.py` produces exactly those
shapes, and nothing else can ever be confused for them. The network has to
separate hundreds of words that genuinely look alike, which is a different and
much harder job. A high score here means the rules match the rig, not that
sign recognition is solved.

Thresholds were calibrated by measuring the encoder against that rig, which
reports curls in the range 0.0 (fully extended) to about 0.47 (closed fist) and
thumb reach from 0.7 (tucked) to 1.1 (extended).
"""

from __future__ import annotations

import numpy as np

from ..core.types import SignPrediction
from .base import Recognizer
from .features import COORD_DIMS, FEATURE_DIMS

_CURL = COORD_DIMS
_REACH = _CURL + 5

#: Finger curl at or below this reads as fully extended.
STRAIGHT = 0.15
#: Finger curl at or above this reads as folded into the palm.
FOLDED = 0.33
#: Thumb reach above this reads as extended away from the fist.
THUMB_OUT = 0.90
#: Thumb reach below this reads as tucked across the fingers.
THUMB_TUCKED = 0.85

LABELS = (
    "open_palm",
    "closed_fist",
    "pointing_up",
    "thumbs_up",
    "victory",
    "i_love_you",
)


class GestureRecognizer(Recognizer):
    name = "rule based gestures"

    def __init__(self, min_confidence: float = 0.62, recent_frames: int = 12) -> None:
        self.min_confidence = min_confidence
        self.recent_frames = recent_frames

    @property
    def labels(self) -> list[str]:
        return list(LABELS)

    def predict(self, window: np.ndarray) -> SignPrediction:
        frames = window.reshape(-1, FEATURE_DIMS)
        # Only judge frames where a hand was actually present, otherwise a mostly
        # empty window averages out to a confident-looking neutral hand.
        present = frames[np.abs(frames).sum(axis=1) > 1e-6]
        if len(present) < 3:
            return SignPrediction(label="", confidence=0.0)

        # Judge the most recent frames, not the whole window. A sign is
        # identified by the pose it settles into, and averaging over the approach
        # smears that away into something that looks like a resting hand.
        present = present[-self.recent_frames :]
        curls = present[:, _CURL : _CURL + 5].mean(axis=0)
        reach = present[:, _REACH : _REACH + 5].mean(axis=0)
        scores = self._score(curls, reach)

        order = np.argsort(scores)[::-1]
        best = int(order[0])
        confidence = float(scores[best])
        alternatives = [(LABELS[i], float(scores[i])) for i in order[1:4] if scores[i] > 0.05]
        return SignPrediction(
            label=LABELS[best] if confidence >= self.min_confidence else "",
            confidence=confidence,
            alternatives=alternatives,
        )

    @staticmethod
    def _score(curls: np.ndarray, reach: np.ndarray) -> np.ndarray:
        """Soft membership per gesture, ordered to match LABELS.

        Each test reads 0.5 exactly at its threshold and reaches 1.0 a fixed
        distance past it. That distance is what stops a clearly folded finger
        from scoring only 0.85 and dragging a whole product of three below the
        acceptance cut, while keeping the scores far enough apart for the
        confidence readout to mean something.
        """

        def step(value: float, threshold: float, clear: float) -> float:
            return float(np.clip((value - threshold) / clear + 0.5, 0.0, 1.0))

        def absent(value: float, threshold: float, clear: float) -> float:
            return 1.0 - step(value, threshold, clear)

        # Curls run from 0.0 fully extended to about 0.47 closed, so a fifth of
        # the range is a sensible "clearly past". Thumb reach runs 0.7 to 1.1.
        curl_clear = 0.20
        reach_clear = 0.25

        def folded(value: float) -> float:
            return step(value, FOLDED, curl_clear)

        def straight(value: float) -> float:
            return absent(value, STRAIGHT, curl_clear)

        _thumb, index, middle, ring, pinky = curls
        four = (index, middle, ring, pinky)

        all_straight = float(all(v <= STRAIGHT for v in four))
        all_folded = float(all(v >= FOLDED for v in four))
        thumb_out = step(reach[0], THUMB_OUT, reach_clear)
        thumb_tucked = absent(reach[0], THUMB_TUCKED, reach_clear)

        return np.array(
            [
                # open_palm: every finger out and the thumb out too.
                all_straight * thumb_out,
                # closed_fist: everything folded and the thumb across the fingers.
                all_folded * thumb_tucked,
                # pointing_up: only the index left.
                straight(index) * folded(middle) * folded(ring) * folded(pinky),
                # thumbs_up: fist plus an extended thumb.
                all_folded * thumb_out,
                # victory: index and middle only.
                straight(index) * straight(middle) * folded(ring) * folded(pinky),
                # i_love_you: index and pinky, middle and ring folded.
                straight(index) * straight(pinky) * folded(middle) * folded(ring),
            ],
            dtype=np.float32,
        )
