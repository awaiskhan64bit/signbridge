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
#:
#: These are measured against the current feature encoder, not carried over. The
#: encoder was rewritten from 80 to 86 dimensions after these thresholds were
#: first set, which moved every derived curl and left gestures silent on the rig.
#: Measured over the fifteen entry book, an extended finger lands anywhere from
#: 0.000 to 0.247 and a folded one from 0.359 to 0.429, so the cut goes in the gap
#: between them rather than at a round number.
STRAIGHT = 0.30
#: Finger curl at or above this reads as folded into the palm.
FOLDED = 0.33
#: How far past a threshold counts as clearly past it.
#:
#: Sharp on purpose. The memberships are multiplied together, so four fingers
#: scoring 0.9 each is 0.66 overall, and a gradual ramp pushed the whole book
#: under the 0.62 the app ships with.
CURL_CLEAR = 0.10

#: Thumb reach above this reads as extended away from the fist.
#:
#: Tucked gestures measure about 0.93 and leading ones about 1.09, so the cut sits
#: between the two measured values. The older 0.90 and 0.85 sat below the tucked
#: value, which is why every leading-thumb sign was named as its tucked twin: an
#: open palm came back as four, a fist as a thumbs up, a pointing finger as a gun.
THUMB_OUT = 1.01
#: Thumb reach below this reads as tucked across the fingers.
THUMB_TUCKED = 1.01
#: Reach is a narrow range, so it needs a sharper ramp than curl to separate.
REACH_CLEAR = 0.05

#: (label, finger pattern, thumb out) for every gesture.
#:
#: The pattern is index, middle, ring, pinky with 0 meaning extended and 1 folded.
#: Thumb out is 1 for a sign that leads with the thumb and 0 for one that tucks it
#: across the palm.
#:
#: One table, because the recogniser's rules and the rig's poses have to agree and
#: keeping two lists in step by hand is how they drift. Every entry is a real
#: one-handed sign, and no two share both a finger pattern and a thumb state,
#: which is the only thing this recogniser can tell apart.
#:
#: The first six are the originals and are never to be removed. The rest are ASL
#: signs for words, so the app can say something rather than name a hand shape.
#: Their hand shapes are the settled pose of each sign: "my name is" is a flat
#: hand on the chest, "name" is the H of both hands, "how are you" is fists turned
#: palm up. The movement itself is not in the hand shape, so a word that is
#: defined by motion rather than posture is approximated by the pose it settles
#: into. That is a real limit of a static matcher and is worth knowing.
GESTURES: tuple[tuple[str, tuple[int, int, int, int], int], ...] = (
    # The original six.
    ("open_palm", (0, 0, 0, 0), 1),
    ("closed_fist", (1, 1, 1, 1), 0),
    ("pointing_up", (1, 0, 0, 0), 0),
    ("thumbs_up", (1, 1, 1, 1), 1),
    ("victory", (0, 0, 1, 1), 0),
    ("i_love_you", (0, 1, 1, 0), 1),
    # Words, as ASL signs.
    ("my", (0, 0, 0, 0), 0),          # flat hand to the chest
    ("name", (0, 1, 1, 1), 0),        # both hands forming an H
    ("is", (0, 0, 0, 1), 0),          # index up, held
    ("i", (0, 0, 0, 1), 1),           # index up, turned back toward the signer
    ("am", (0, 0, 1, 1), 1),          # flat hand, palm out, drawn in
    ("happy", (0, 1, 1, 1), 1),       # flat hand brushing up the chest
    ("how", (1, 1, 1, 0), 0),         # fists, palms up, shrugged
    ("are_you", (1, 1, 0, 1), 1),     # fists turned forward, then a point at you
)

LABELS = tuple(label for label, _, _ in GESTURES)

EXPECTED_GESTURES = len(GESTURES)


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

        # Curls and reach each span a narrow measured range, so both ramps are sharp.
        curl_clear = CURL_CLEAR
        reach_clear = REACH_CLEAR

        def folded(value: float) -> float:
            return step(value, FOLDED, curl_clear)

        def straight(value: float) -> float:
            return absent(value, STRAIGHT, curl_clear)

        _thumb, index, middle, ring, pinky = curls
        four = (index, middle, ring, pinky)

        thumb_out = step(reach[0], THUMB_OUT, reach_clear)
        thumb_tucked = absent(reach[0], THUMB_TUCKED, reach_clear)

        # Each gesture's rule is generated from its entry in GESTURES, so a new
        # sign is one row rather than a new hand written product that has to be
        # kept in step with the rig's pose by hand. A finger is required to be
        # straight or folded exactly as its pattern says, and the thumb factor
        # separates the pairs that share a pattern.
        out = np.empty(len(GESTURES), dtype=np.float32)
        for i, (_label, pattern, want_out) in enumerate(GESTURES):
            score = 1.0
            for value, folded_flag in zip(four, pattern, strict=True):
                score *= folded(value) if folded_flag else straight(value)
            out[i] = score * (thumb_out if want_out else thumb_tucked)
        return out
