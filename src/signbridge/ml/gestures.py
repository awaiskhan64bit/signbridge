"""Rule based gesture recognizer.

Not machine learning, and not pretending to be. It exists for two reasons:

  * it gives the app a working vocabulary out of the box, before any model has
    been trained, so the pipeline, HUD and speech can be verified end to end
  * it is a useful control condition when evaluating the trained network

It reads the same feature windows the neural recognizer does and decides from
average finger curl plus how far the thumb reaches.

Thresholds were calibrated by measuring the encoder against the hand rig in
`vision/simulated.py`, which reports curls in the range 0.0 (fully extended) to
about 0.47 (closed fist) and thumb reach from 0.7 (tucked) to 1.1 (extended).

Note that `thumbs_down` is deliberately absent. The feature encoder is rotation
invariant on purpose, which makes a thumbs up and a thumbs down produce identical
features. Telling them apart needs screen-relative orientation, which this layer
deliberately throws away.
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
        """Soft membership per gesture, ordered to match LABELS."""

        def step(value: float, threshold: float) -> float:
            """1 when clearly past the threshold, fading to 0 by twice past it."""
            margin = abs(threshold) * 0.6 + 0.05
            return float(np.clip((value - threshold) / margin + 0.5, 0.0, 1.0))

        def absent(value: float, threshold: float) -> float:
            return 1.0 - step(value, threshold)

        _thumb, index, middle, ring, pinky = curls
        four = np.array([index, middle, ring, pinky], dtype=np.float32)

        all_straight = float(np.all(four <= STRAIGHT))
        all_folded = float(np.all(four >= FOLDED))
        thumb_out = step(reach[0], THUMB_OUT)
        thumb_tucked = absent(reach[0], THUMB_TUCKED)

        return np.array(
            [
                # open_palm: every finger out and the thumb out too.
                all_straight * thumb_out,
                # closed_fist: everything folded and the thumb across the fingers.
                all_folded * thumb_tucked,
                # pointing_up: only the index left.
                absent(index, STRAIGHT)
                * step(middle, FOLDED)
                * step(ring, FOLDED)
                * step(pinky, FOLDED),
                # thumbs_up: fist plus an extended thumb.
                all_folded * thumb_out,
                # victory: index and middle only.
                absent(index, STRAIGHT)
                * absent(middle, STRAIGHT)
                * step(ring, FOLDED)
                * step(pinky, FOLDED),
                # i_love_you: index and pinky, middle and ring folded.
                absent(index, STRAIGHT)
                * absent(pinky, STRAIGHT)
                * step(middle, FOLDED)
                * step(ring, FOLDED),
            ],
            dtype=np.float32,
        )
