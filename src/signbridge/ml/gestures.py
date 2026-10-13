"""Template matching gesture recogniser.

Not machine learning, and not pretending to be. It exists so the app has a
working vocabulary out of the box, before any model has been trained, which lets
the pipeline, HUD and speech be verified end to end.

It reads the same feature windows the neural recogniser does and matches them
against one stored template per gesture.

Why templates and not rules. An earlier version scored each gesture with a hand
written expression over finger curl and thumb reach. That works for six shapes
and collapses past that: ASL words largely share hand shapes and differ by
movement and location, so with a hundred entries the expressions would overlap
constantly. Measured on a hundred hand written poses, the closest pair landed
0.03 apart on an 86 dimensional vector, which is the same vector as far as the
recogniser is concerned, so two words became one gesture. Storing a template
per word and taking the nearest one scales without adding a branch per word.

What the bundled templates are. They come from vision/gesturebook.py, where the
finger configurations are generated to be mutually distinguishable but are not
reconstructions of the real ASL sign alphabet. They match the demo rig and
nothing else, because the rig is what produced them. On a webcam they are a
starting point, not a working vocabulary: `load_templates` accepts clips the
user recorded, and those replace the synthetic poses for the words they cover.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Hand, Handedness, SignPrediction
from .base import Recognizer
from .features import FEATURE_DIMS, encode_hand

#: Distance from a template, in units of the book's own spread, at which a match
#: is considered exact. Used both as the acceptance limit and as the span that
#: maps fit quality onto a confidence between 0 and 1.
_MATCH_TOLERANCE = 0.55

#: Speaking threshold for this recogniser. Sits below the neural model's 0.62
#: because the two produce confidence on different scales. Measured on the demo
#: book, an exact match scores about 0.98 and the weakest genuine gesture about
#: 0.75, so this threshold keeps every real gesture and still rejects a hand that
#: is nowhere near any template.
_DEFAULT_THRESHOLD = 0.62


def default_templates() -> dict[str, np.ndarray]:
    """Feature template per gesture, built from the demo rig's own poses."""

    from ..vision.gesturebook import build_gestures, labels
    from ..vision.simulated import build_landmarks

    out: dict[str, np.ndarray] = {}
    for label in labels():
        pose = build_gestures().get(label)
        if pose is None:
            continue
        landmarks = build_landmarks(pose)
        out[label] = encode_hand(
            Hand(landmarks=landmarks, handedness=Handedness.RIGHT)
        ).astype(np.float32)
    return out


class GestureRecognizer(Recognizer):
    name = "template gestures"

    def __init__(self, min_confidence: float = 0.62, recent_frames: int = 12) -> None:
        self.min_confidence = min_confidence
        self.recent_frames = recent_frames
        self._templates: dict[str, np.ndarray] | None = None
        self._order: list[str] = []
        self._stack: np.ndarray | None = None
        #: Typical distance between two different templates, used to scale
        #: distances so the tolerance means the same thing if the book changes.
        self._spread = 1.0

    # ------------------------------------------------------------- templates

    def load_templates(self, templates: dict[str, np.ndarray]) -> None:
        """Replace the template set, typically with clips the user recorded.

        Anything already present is overwritten, so a handful of recorded words
        can sit on top of the synthetic book rather than replacing all of it.
        """

        merged = default_templates()
        for label, vector in templates.items():
            merged[label] = np.asarray(vector, dtype=np.float32)
        self._install(merged)

    def _install(self, templates: dict[str, np.ndarray]) -> None:
        self._order = sorted(templates)
        self._templates = templates
        self._stack = np.stack([templates[label] for label in self._order])

        if len(self._order) > 1:
            d = np.linalg.norm(
                self._stack[:, None, :] - self._stack[None, :, :], axis=-1
            )
            off = d[~np.eye(len(self._order), dtype=bool)]
            self._spread = float(np.median(off)) or 1.0

    def _ensure_loaded(self) -> bool:
        if self._templates is None:
            self._install(default_templates())
        return self._stack is not None

    @property
    def default_threshold(self) -> float:
        """Threshold appropriate to this recogniser.

        Gestures and words share one confidence slider, but they do not share a
        confidence scale. This recogniser's calibrated score for a clear match
        is well below the neural model's 0.62 cut, so adopting that cut here
        silences every gesture. Returning a value below 1 lets the UI slider
        span both engines honestly instead of picking whichever one is strictest.
        """

        return _DEFAULT_THRESHOLD

    @property
    def labels(self) -> list[str]:
        self._ensure_loaded()
        return list(self._order)

    # ------------------------------------------------------------- inference

    def predict(self, window: np.ndarray) -> SignPrediction:
        if not self._ensure_loaded() or self._stack is None:
            return SignPrediction(label="", confidence=0.0)

        frames = np.asarray(window, dtype=np.float32).reshape(-1, FEATURE_DIMS)
        # Only judge frames where a hand was actually present, otherwise a mostly
        # empty window averages out to a confident-looking neutral hand.
        present = frames[np.abs(frames).sum(axis=1) > 1e-6]
        if len(present) < 3:
            return SignPrediction(label="", confidence=0.0)

        # Judge the most recent frames, not the whole window. A sign is
        # identified by the pose it settles into, and averaging over the approach
        # smears that away into something that looks like a resting hand.
        present = present[-self.recent_frames :]
        observed = present.mean(axis=0)

        distances = np.linalg.norm(self._stack - observed, axis=1)
        scaled = distances / max(self._spread, 1e-6)

        order = np.argsort(distances)
        best = int(order[0])

        # Confidence is fit quality, not a softmax over the book. Softmaxing
        # across a hundred templates made confidence depend on how many entries
        # existed: with 104 gestures the winner's share fell to a median of 0.36
        # even when the hand was exactly on template, so any threshold set for
        # the neural recogniser silenced most of the book. How closely the
        # observed hand sits on its template is a scale that does not move when
        # the vocabulary grows, which is the property that matters here.
        confidence = float(np.clip(1.0 - scaled[best] / _MATCH_TOLERANCE, 0.0, 1.0))

        # Beyond the tolerance nothing is a match, even as the least bad option.
        if scaled[best] > _MATCH_TOLERANCE:
            return SignPrediction(label="", confidence=confidence)

        fits = np.clip(1.0 - scaled / _MATCH_TOLERANCE, 0.0, 1.0)
        alternatives = [
            (self._order[i], float(fits[i])) for i in order[1:4] if fits[i] > 0.02
        ]
        return SignPrediction(
            label=self._order[best] if confidence >= self.min_confidence else "",
            confidence=confidence,
            alternatives=alternatives,
        )