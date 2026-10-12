"""Tests for the temporal decoder.

The decoder is what stops the app repeating itself, so these tests are about
counting: how many times does one physical sign produce an event?
"""

from __future__ import annotations

from signbridge.core.decoder import Event, SignDecoder
from signbridge.core.types import SignPrediction


def pred(label: str, confidence: float = 0.9) -> SignPrediction:
    return SignPrediction(label=label, confidence=confidence)


def signs(decoder: SignDecoder, label: str, times: int, start: float = 0.0) -> list:
    out = []
    for i in range(times):
        out.extend(decoder.push(pred(label), start + i * 33.0))
    return out


def test_low_confidence_is_not_a_sign():
    """The recognizer returns an empty label below threshold; that must not
    become a confirmed sign."""

    decoder = SignDecoder(votes_required=2)
    assert signs(decoder, "", 10) == []


def test_a_held_sign_fires_exactly_once():
    """The whole point of latching: forty windows of the same sign is one event,
    not forty events."""

    decoder = SignDecoder(votes_required=3)
    events = signs(decoder, "open_palm", 40)
    assert len([e for e in events if e.kind is Event.SIGN]) == 1


def test_sign_needs_enough_agreeing_windows():
    decoder = SignDecoder(votes_required=3)
    events = signs(decoder, "victory", 2)
    assert not [e for e in events if e.kind is Event.SIGN]


def test_moving_away_then_back_repeats_the_sign():
    """Signing the same word twice in a row has to work, so a sign must unlatch
    when the hand leaves and relatch when it comes back."""

    decoder = SignDecoder(votes_required=2)
    events = signs(decoder, "victory", 5)
    events += signs(decoder, "", 5, start=200.0)
    events += signs(decoder, "victory", 5, start=400.0)

    confirmed = [e for e in events if e.kind is Event.SIGN]
    assert len(confirmed) == 2


def test_phrase_flushes_after_a_pause():
    """People do not announce word boundaries, so a pause ends the phrase."""

    decoder = SignDecoder(votes_required=2, phrase_pause_ms=500.0)
    events = signs(decoder, "open_palm", 4)
    events += signs(decoder, "thumbs_up", 4, start=200.0)

    # Nothing yet: the two signs are still close together.
    assert not [e for e in events if e.kind is Event.PHRASE]

    # Push well past the pause and the phrase closes itself.
    events += decoder.push(pred("thumbs_up"), 2000.0)
    phrases = [e for e in events if e.kind is Event.PHRASE]
    assert len(phrases) == 1
    assert phrases[0].text == "hello good"


def test_flush_forces_the_phrase_out():
    decoder = SignDecoder(votes_required=2, phrase_pause_ms=100_000.0)
    signs(decoder, "open_palm", 4)
    phrases = [e for e in decoder.flush(1000.0) if e.kind is Event.PHRASE]
    assert len(phrases) == 1


def test_phrase_splits_at_the_sign_limit():
    """A long phrase would drag on forever, so it is cut at a fixed length."""

    decoder = SignDecoder(votes_required=1, signs_per_phrase=3)
    events = signs(decoder, "open_palm", 1)
    events += signs(decoder, "thumbs_up", 1, start=100.0)
    events += signs(decoder, "i_love_you", 1, start=200.0)

    phrases = [e for e in events if e.kind is Event.PHRASE]
    assert len(phrases) == 1
    assert phrases[0].signs == ["open_palm", "thumbs_up", "i_love_you"]
    assert phrases[0].text == "hello good I love you"


def test_labels_map_to_spoken_words():
    """Classifier labels are not sentences. The glossary is what makes the app
    say "hello" rather than "open palm"."""

    decoder = SignDecoder(votes_required=1)
    events = signs(decoder, "i_love_you", 2)
    sign = next(e for e in events if e.kind is Event.SIGN)
    assert sign.text == "I love you"
