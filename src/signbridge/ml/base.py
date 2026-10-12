"""Recognizer interface and the vocabulary of spoken words.

Every recognizer turns a window of feature frames into one :class:`SignPrediction`.
The pipeline never imports a concrete model, which is what lets the same UI run
against a trained network, MediaPipe's canned gestures, or nothing at all.
"""

from __future__ import annotations

import abc
from typing import ClassVar

import numpy as np

from ..core.types import SignPrediction


class Recognizer(abc.ABC):
    """Scores a (window_frames, feature_dims) array of landmarks."""

    name: str = "abstract"

    @abc.abstractmethod
    def predict(self, window: np.ndarray) -> SignPrediction:
        """Classify one temporal window."""

    def reset(self) -> None:  # noqa: B027
        """Drop any per-stream state. Optional: most recognisers hold none."""

    @property
    def labels(self) -> list[str]:
        return []

    @property
    def ready(self) -> bool:
        return True

    def status(self) -> str:
        return self.name


class NullRecognizer(Recognizer):
    """The app still runs (camera, HUD, speech) when no model is installed."""

    name = "no model loaded"

    def predict(self, window: np.ndarray) -> SignPrediction:
        return SignPrediction(label="", confidence=0.0)


# Labels the pipeline treats specially rather than speaking.
IGNORED_LABELS = frozenset({"", "none", "unknown", "silence", "no_gesture"})


class Glossary:
    """Maps classifier labels to the words that actually get spoken.

    Labels come from whatever dataset trained the model, which is often not what
    you want a screen reader to say. This is the one place to fix that, and it is
    also where multi word signs get their phrasing.
    """

    #: label -> spoken text. Anything absent is title cased from the label.
    #: Annotated ClassVar because this is shared configuration, not instance
    #: state: subclasses and tests both read it.
    OVERRIDES: ClassVar[dict[str, str]] = {
        # The built in rule based vocabulary. These map onto ordinary words,
        # because "open palm" is nobody's idea of a greeting.
        "open_palm": "hello",
        "thumbs_up": "good",
        "i_love_you": "I love you",
        "pointing_up": "up",
        "victory": "peace",
        "closed_fist": "stop",
        # Common dataset labels that need phrasing rather than underscores.
        "thank_you": "thank you",
        "good_morning": "good morning",
        "nice_to_meet_you": "nice to meet you",
        "asl": "ASL",
    }

    @classmethod
    def say(cls, label: str) -> str:
        """The phrase a confirmed sign should produce."""

        key = label.strip().lower()
        if key in IGNORED_LABELS:
            return ""
        if key in cls.OVERRIDES:
            return cls.OVERRIDES[key]
        return key.replace("_", " ")

    @classmethod
    def is_spoken(cls, label: str) -> bool:
        return cls.say(label) != ""

    @classmethod
    def display(cls, label: str) -> str:
        """What to show on screen.

        Dataset labels are lowercase and sometimes multi-word ("thank you"), and
        the live sign readout is the largest text in the UI, so it gets sentence
        case rather than whatever the dataset happened to use.
        """

        spoken = cls.say(label)
        if spoken:
            return spoken if spoken.isupper() else spoken[:1].upper() + spoken[1:]
        return label.replace("_", " ").upper()
