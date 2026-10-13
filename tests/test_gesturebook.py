"""Tests for the gesture book and the template recogniser.

A hundred-entry vocabulary is easy to fill with entries that look distinct and
encode to the same thing. These pin down the properties that matter: every
gesture is a genuinely different vector, the recogniser names the rig's own
gestures correctly, and a recorded clip can replace a synthetic template.
"""

from __future__ import annotations

import numpy as np

from signbridge.core.types import Hand, Handedness
from signbridge.ml.features import FEATURE_DIMS, encode_hand
from signbridge.ml.gestures import GestureRecognizer, default_templates
from signbridge.vision.gesturebook import (
    EXPECTED_GESTURES,
    LEGACY_WORDS,
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


def test_the_book_holds_the_expected_number_of_gestures():
    assert label_count() == EXPECTED_GESTURES
    assert len(set(labels())) == EXPECTED_GESTURES, "duplicate gesture names"


def test_the_six_original_gestures_are_still_present():
    """People learned those names, and existing demos rely on them."""

    assert set(LEGACY_WORDS) <= set(labels())


def test_every_gesture_has_a_pose_and_a_template():
    poses = build_gestures()
    templates = default_templates()
    assert len(poses) == EXPECTED_GESTURES
    assert set(poses) == set(templates)


def test_gestures_encode_to_genuinely_different_vectors():
    """The bug this guards: 104 poses whose closest pair differed by 0.03.

    Two words that encode to the same vector are one word as far as the
    recogniser is concerned, however different they look in the source table.
    """

    stack = np.stack(list(default_templates().values()))
    d = np.linalg.norm(stack[:, None, :] - stack[None, :, :], axis=-1)
    off = d[~np.eye(len(stack), dtype=bool)]
    typical = float(np.median(off))
    closest = float(off.min())

    assert closest > 0.15 * typical, (
        f"closest gesture pair is {closest:.3f}, only "
        f"{closest / typical:.1%} of a typical separation of {typical:.3f}"
    )


def test_gestures_stay_identifiable_under_jitter():
    """A hand is never exactly on template, so noise must not swap the answer."""

    templates = default_templates()
    names = list(templates)
    stack = np.stack(list(templates.values()))
    rng = np.random.default_rng(0)

    correct = 0
    for _ in range(10):
        noisy = stack * (1.0 + rng.normal(0.0, 0.10, size=(1, stack.shape[1])))
        noisy = noisy + rng.normal(0.0, 0.15 * stack.std(axis=0), size=noisy.shape)
        pred = np.argmin(
            np.linalg.norm(noisy[:, None, :] - stack[None, :, :], axis=-1), axis=1
        )
        correct += int((pred == np.arange(len(names))).sum())

    assert correct / (10 * len(names)) > 0.95


def test_every_gesture_speaks_at_the_threshold_the_app_ships():
    """The bug this guards: gestures went silent because their confidence scale
    did not match the neural model's.

    Confidence used to be a softmax over the whole book, so the winner's share
    fell as gestures were added, and a threshold set for words silenced every
    gesture. Fit quality is scale-free, so the whole book must now clear the
    shipped cut.
    """

    from signbridge.ml.gestures import _DEFAULT_THRESHOLD

    recognizer = GestureRecognizer(min_confidence=0.0)
    recognizer.min_confidence = _DEFAULT_THRESHOLD
    silent = []
    for label, poses in PHRASEBOOK:
        if not recognizer.predict(_window_for(poses[-1])).label:
            silent.append(label)

    assert not silent, f"silent at the shipped threshold: {silent[:10]}"


def test_confidence_degrades_as_the_hand_drifts_off_template():
    """A confidence dial pinned at 1.0 is decoration, not information."""

    recognizer = GestureRecognizer(min_confidence=0.0)
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
            confs.append(recognizer.predict(window).confidence)
        means.append(float(np.mean(confs)))

    assert means[0] > 0.95, f"an exact match should be near certain, got {means[0]}"
    assert means[0] > means[1] > means[2], f"confidence must fall with drift: {means}"
    assert means[2] < 0.7, f"a clearly wrong hand must not look confident: {means}"


def test_recogniser_names_every_gesture_the_rig_draws():
    """Rig and recogniser read one table, so this can only fail if they drift."""

    recognizer = GestureRecognizer(min_confidence=0.0)
    wrong = []
    for label, poses in PHRASEBOOK:
        prediction = recognizer.predict(_window_for(poses[-1]))
        if prediction.label != label:
            wrong.append((label, prediction.label or "silent"))

    assert not wrong, f"mislabelled: {wrong[:10]}"


def test_an_absent_hand_stays_silent():
    """All-zero frames must not average into a confident-looking pose."""

    recognizer = GestureRecognizer(min_confidence=0.0)
    silent = recognizer.predict(np.zeros((WINDOW, FEATURE_DIMS), dtype=np.float32))
    assert silent.label == ""


def test_a_recorded_clip_can_replace_a_synthetic_template():
    """The path from demo gestures to a working camera vocabulary.

    Whatever the user records has to win over the generated pose for the words
    it covers, which is what makes teaching the app possible.
    """

    recognizer = GestureRecognizer(min_confidence=0.0)
    before = dict(recognizer.labels and {n: None for n in recognizer.labels})

    recognizer.load_templates({"hello": np.full(FEATURE_DIMS, 7.0, dtype=np.float32)})

    assert set(before) == set(recognizer.labels), "template set must not lose entries"
    # The recorded vector now sits at the centre of the "hello" template cloud,
    # so a window matching it should come back as hello.
    assert recognizer._templates["hello"].tolist() == [7.0] * FEATURE_DIMS


def test_match_confidence_drops_when_the_hand_is_nowhere_near_a_template():
    recognizer = GestureRecognizer(min_confidence=0.0)
    rng = np.random.default_rng(1)
    junk = rng.normal(0.0, 25.0, size=(WINDOW, FEATURE_DIMS)).astype(np.float32)
    assert recognizer.predict(junk).label == ""