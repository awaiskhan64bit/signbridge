"""Tests for the gesture book and the template recogniser.

These pin down the behaviour the app depends on: the rig animates exactly the
gestures the recogniser can name, all six clear the confidence threshold, a hand
that is nowhere near any template stays silent, and a clip the user recorded can
replace a synthetic template.

The vocabulary is six, and the tests assert that number so it cannot quietly grow
back. A hundred-entry version was built and measured: it read perfectly on the
demo rig, which is the only place those templates are valid, and reached a real
hand no closer than before. Growing the table inflated the demo without adding
any reach into live input.
"""

from __future__ import annotations

import numpy as np

from signbridge.core.types import Hand, Handedness
from signbridge.ml.features import FEATURE_DIMS, encode_hand
from signbridge.ml.gestures import GestureRecognizer, default_templates
from signbridge.vision.gesturebook import (
    EXPECTED_GESTURES,
    build_gestures,
    label_count,
    labels,
)
from signbridge.vision.simulated import PHRASEBOOK, build_landmarks

WINDOW = 64


def _window_for(pose, hold: int = 12) -> np.ndarray:
    """A window shaped the way the live pipeline delivers one: zeros, then a
    held pose for the last few frames."""

    rest = np.zeros(FEATURE_DIMS, dtype=np.float32)
    frame = encode_hand(
        Hand(landmarks=build_landmarks(pose), handedness=Handedness.RIGHT)
    )
    return np.concatenate(
        [np.stack([rest] * (WINDOW - hold)), np.stack([frame] * hold)]
    )


def test_the_book_is_six_gestures():
    assert label_count() == EXPECTED_GESTURES == 6
    assert len(set(labels())) == 6, "duplicate gesture names"


def test_the_six_original_gestures_are_the_ones_that_remain():
    assert set(labels()) == {
        "open_palm",
        "closed_fist",
        "pointing_up",
        "thumbs_up",
        "victory",
        "i_love_you",
    }


def test_every_gesture_has_a_pose_and_a_template():
    poses = build_gestures()
    templates = default_templates()
    assert len(poses) == EXPECTED_GESTURES
    assert set(poses) == set(templates)


def test_the_rig_animates_exactly_what_the_recogniser_knows():
    """Rig and recogniser read one table, so this can only fail if they drift."""

    recogniser = GestureRecognizer(min_confidence=0.0)
    assert {label for label, _ in PHRASEBOOK} == set(recogniser.labels)
    assert len(PHRASEBOOK) == EXPECTED_GESTURES


def test_recogniser_names_every_gesture_the_rig_draws():
    recogniser = GestureRecognizer(min_confidence=0.0)
    wrong = []
    for label, poses in PHRASEBOOK:
        prediction = recogniser.predict(_window_for(poses[-1]))
        if prediction.label != label:
            wrong.append((label, prediction.label or "silent"))

    assert not wrong, f"mislabelled: {wrong}"


def test_every_gesture_speaks_at_the_threshold_the_app_ships():
    """The bug this guards: gestures went silent because their confidence scale
    did not match the neural model's.

    Confidence used to be a softmax over the whole book, so the winner's share
    fell as entries were added, and a threshold set for words silenced every
    gesture. Fit quality is scale-free, so the whole book must clear the cut.
    """

    from signbridge.ml.gestures import _DEFAULT_THRESHOLD

    recogniser = GestureRecognizer(min_confidence=_DEFAULT_THRESHOLD)
    silent = [
        label for label, poses in PHRASEBOOK if not recogniser.predict(_window_for(poses[-1])).label
    ]

    assert not silent, f"silent at the shipped threshold: {silent}"


def test_confidence_degrades_as_the_hand_drifts_off_template():
    """A confidence dial pinned at 1.0 is decoration, not information."""

    recogniser = GestureRecognizer(min_confidence=0.0)
    rng = np.random.default_rng(3)

    means = []
    for noise in (0.0, 0.05, 0.12):
        confs = []
        for _, poses in PHRASEBOOK:
            rest = np.zeros(FEATURE_DIMS, dtype=np.float32)
            frame = encode_hand(
                Hand(landmarks=build_landmarks(poses[-1]), handedness=Handedness.RIGHT)
            )
            frame = frame * (1.0 + rng.normal(0.0, noise, size=frame.shape))
            window = np.concatenate(
                [np.stack([rest] * (WINDOW - 12)), np.stack([frame] * 12)]
            )
            confs.append(recogniser.predict(window).confidence)
        means.append(float(np.mean(confs)))

    assert means[0] > 0.95, f"an exact match should be near certain, got {means[0]}"
    assert means[0] > means[1], f"confidence must fall with drift: {means}"


def test_a_hand_nowhere_near_a_template_stays_silent():
    """A real webcam hand sits far outside these synthetic templates.

    It must be rejected rather than snapped to the nearest one, which is what
    would otherwise produce confident nonsense from every frame of live input.
    """

    recogniser = GestureRecognizer(min_confidence=0.0)
    rng = np.random.default_rng(1)
    junk = rng.normal(0.0, 25.0, size=(WINDOW, FEATURE_DIMS)).astype(np.float32)
    assert recogniser.predict(junk).label == ""


def test_an_absent_hand_stays_silent():
    """All-zero frames must not average into a confident-looking pose."""

    recogniser = GestureRecognizer(min_confidence=0.0)
    silent = recogniser.predict(np.zeros((WINDOW, FEATURE_DIMS), dtype=np.float32))
    assert silent.label == ""


def test_a_recorded_clip_can_replace_a_synthetic_template():
    """The path from demo gestures to a working camera vocabulary.

    Whatever the user records has to win over the generated pose for the words it
    covers, which is what makes teaching the app possible.
    """

    recogniser = GestureRecognizer(min_confidence=0.0)
    before = set(recogniser.labels)

    recogniser.load_templates({"open_palm": np.full(FEATURE_DIMS, 7.0, dtype=np.float32)})

    assert set(before) == set(recogniser.labels), "template set must not lose entries"
    assert recogniser._templates["open_palm"].tolist() == [7.0] * FEATURE_DIMS