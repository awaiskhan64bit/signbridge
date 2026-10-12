"""Temporal decoding: from noisy per frame guesses to confirmed signs and phrases.

Two problems get solved here.

Stability. A classifier re-decodes the same held sign on every window, so a naive
"read the top result" loop would say HELLO forty times a second. A sign is only
confirmed once it has won `votes_required` consecutive windows, and then it latches
until the hand moves somewhere else, so one physical sign produces one event.

Timing. People do not announce word boundaries, so signs accumulate into a phrase
that is spoken when the signer pauses, or when the phrase grows past a limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..ml.base import Glossary
from ..ml.features import FEATURE_DIMS
from .types import SignPrediction, SpeechChunk


class Event(str, Enum):
    SIGN = "sign"
    PHRASE = "phrase"


@dataclass(slots=True)
class DecodedEvent:
    kind: Event
    label: str
    text: str
    confidence: float
    phrase: str = ""
    signs: list[str] = field(default_factory=list)


class SignDecoder:
    """Stateful, single consumer. Feed it predictions in arrival order."""

    def __init__(
        self,
        *,
        votes_required: int = 3,
        signs_per_phrase: int = 0,
        phrase_pause_ms: float = 900.0,
        max_phrase_signs: int = 6,
    ) -> None:
        self.votes_required = max(1, votes_required)
        self.signs_per_phrase = signs_per_phrase
        self.phrase_pause_ms = phrase_pause_ms
        self.max_phrase_signs = max_phrase_signs

        self._candidate = ""
        self._votes = 0
        self._latched = ""  # the sign already confirmed, waiting for the hand to leave
        self._candidate_confidence = 0.0
        self._candidate_alternatives: list[tuple[str, float]] = []

        self._phrase: list[str] = []
        self._phrase_text: list[str] = []
        self._last_sign_at_ms = -1e9
        self._started_at_ms = 0.0

    # ------------------------------------------------------------- accessors

    @property
    def pending_label(self) -> str:
        return self._candidate

    @property
    def pending_confidence(self) -> float:
        return self._candidate_confidence

    @property
    def pending_alternatives(self) -> list[tuple[str, float]]:
        return self._candidate_alternatives

    @property
    def phrase_signs(self) -> list[str]:
        return list(self._phrase_text)

    @property
    def phrase(self) -> str:
        return " ".join(self._phrase_text)

    def reset(self) -> None:
        self._candidate = ""
        self._votes = 0
        self._latched = ""
        self._candidate_confidence = 0.0
        self._candidate_alternatives = []
        self._phrase.clear()
        self._phrase_text.clear()
        self._last_sign_at_ms = -1e9

    # ------------------------------------------------------------------ feed

    def push(self, prediction: SignPrediction, now_ms: float) -> list[DecodedEvent]:
        """Feed one window's prediction. Returns whatever became finalised."""

        events: list[DecodedEvent] = []
        label = prediction.label if Glossary.is_spoken(prediction.label) else ""

        if label != self._candidate:
            self._candidate = label
            self._votes = 1 if label else 0
            self._candidate_confidence = prediction.confidence
            self._candidate_alternatives = prediction.alternatives
        else:
            self._votes += 1
            self._candidate_confidence = prediction.confidence
            self._candidate_alternatives = prediction.alternatives

        if label and label != self._latched and self._votes >= self.votes_required:
            self._latched = label
            events.append(
                DecodedEvent(
                    kind=Event.SIGN,
                    label=label,
                    text=Glossary.say(label),
                    confidence=prediction.confidence,
                )
            )
            self._append_to_phrase(label, now_ms, events)

        # Unlatch as soon as the hand moves off the sign, so the same word can be
        # signed twice in a row provided there is a movement between them.
        if self._latched and label != self._latched:
            self._latched = ""

        flushed = self._flush_if_paused(now_ms)
        events.extend(flushed)
        return events

    def flush(self, now_ms: float) -> list[DecodedEvent]:
        """Force the current phrase out, for example when the user presses stop."""

        events = self._flush_if_paused(now_ms, force=True)
        self._latched = ""
        return events

    # -------------------------------------------------------------- internals

    def _append_to_phrase(
        self, label: str, now_ms: float, events: list[DecodedEvent]
    ) -> None:
        text = Glossary.say(label)
        if not text:
            return
        if not self._phrase_text:
            self._started_at_ms = now_ms
        self._phrase.append(label)
        self._phrase_text.append(text)
        self._last_sign_at_ms = now_ms

        limit = self.signs_per_phrase or self.max_phrase_signs
        if len(self._phrase_text) >= limit:
            events.extend(self._emit(now_ms))

    def _flush_if_paused(self, now_ms: float, force: bool = False) -> list[DecodedEvent]:
        if not self._phrase_text:
            return []
        if force or (now_ms - self._last_sign_at_ms) >= self.phrase_pause_ms:
            return self._emit(now_ms)
        return []

    def _emit(self, now_ms: float) -> list[DecodedEvent]:
        chunk = SpeechChunk(
            text=self.phrase,
            signs=list(self._phrase),
            started_at_ms=self._started_at_ms,
        )
        event = DecodedEvent(
            kind=Event.PHRASE,
            label=self._phrase[-1] if self._phrase else "",
            text=chunk.text,
            confidence=self._candidate_confidence,
            phrase=chunk.text,
            signs=chunk.signs,
        )
        self._phrase.clear()
        self._phrase_text.clear()
        return [event]


def window_has_hand(window: np.ndarray) -> bool:
    """True when a window still contains a real hand in its most recent frames."""

    if window is None or len(window) == 0:
        return False
    recent = window.reshape(-1, FEATURE_DIMS)[-8:]
    return bool(np.abs(recent).sum() > 1e-6)
