"""Tests for the feature encoder.

These are the tests that matter most in the project. If normalisation is wrong,
training data and live input disagree and accuracy collapses, but nothing crashes
and no error is logged. The invariants below are what stop that happening
silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.core.types import LANDMARK_COUNT, Hand, Handedness
from signbridge.ml.features import (
    COORD_DIMS,
    FEATURE_DIMS,
    FRAME_START,
    encode_frame,
    encode_hand,
    encode_sequence,
    frame_features,
    normalise,
)


def make_hand(curl: float = 0.0, scale: float = 1.0, handedness=Handedness.RIGHT) -> np.ndarray:
    """A synthetic hand. `curl` folds the fingers so tests can vary the pose."""

    pts = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]
    mcp_positions = [(5, 8, 0.34), (9, 12, 0.41), (13, 16, 0.48), (17, 20, 0.54)]
    for i, (mcp, _tip, x) in enumerate(mcp_positions):
        drift = curl * 0.06 * i
        pts[mcp] = [x, 0.62, 0.0]
        pts[mcp + 1] = [x, 0.5, 0.0]
        pts[mcp + 2] = [x + drift, 0.42, -curl * 0.15]
        pts[mcp + 3] = [x + drift * 1.6, 0.34 + curl * 0.04, -curl * 0.3]
    pts[1] = [0.27, 0.74, 0.0]
    pts[2] = [0.24, 0.66, 0.0]
    pts[3] = [0.23 - curl * 0.03, 0.6, 0.0]
    pts[4] = [0.22 - curl * 0.05, 0.54, 0.0]

    pts[:, :2] = (pts[:, :2] - [0.5, 0.6]) * scale + [0.5, 0.6]
    return pts


def test_feature_shape_and_finiteness():
    features = encode_hand(Hand(landmarks=make_hand()))
    assert features.shape == (FEATURE_DIMS,)
    assert np.isfinite(features).all()


def test_rejects_wrong_landmark_count():
    with pytest.raises(ValueError):
        normalise(np.zeros((10, 3), dtype=np.float32))


@pytest.mark.parametrize("curl", [0.0, 0.4, 1.0])
def test_canonical_block_is_invariant_to_placement(curl):
    """The canonical half of the encoding describes hand shape and nothing else.

    Moving, scaling or rolling the hand in frame must not disturb it. This is
    what lets the network recognise a shape across camera setups.
    """

    base = make_hand(curl=curl)

    rng = np.random.default_rng(7)
    angle = rng.uniform(-0.6, 0.6)
    c, s = np.cos(angle), np.sin(angle)
    moved = base.copy()
    moved[:, :2] = base[:, :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)
    moved *= 2.3
    moved[:, 0] += -0.33
    moved[:, 1] += 0.21

    a = encode_hand(Hand(landmarks=base))
    b = encode_hand(Hand(landmarks=moved))
    assert np.abs(a[:COORD_DIMS] - b[:COORD_DIMS]).max() < 1e-3


@pytest.mark.parametrize("curl", [0.0, 0.4, 1.0])
def test_frame_block_tracks_placement_and_rotation(curl):
    """The frame half must move with the hand, or the model loses the sign.

    Sign language is not defined by hand shape alone: a flat hand at the
    forehead is HELLO and the same flat hand at the chest is a different sign.
    An encoder that made the whole vector placement invariant, as an earlier
    version did, deleted that distinction before training could use it, and a
    linear probe over WLASL scored 11.2% against 20.9% for raw landmarks.
    """

    base = make_hand(curl=curl)

    shifted = base.copy()
    shifted[:, 0] += 0.25
    shifted[:, 1] -= 0.15

    a = encode_hand(Hand(landmarks=base))[FRAME_START:]
    b = encode_hand(Hand(landmarks=shifted))[FRAME_START:]
    assert not np.allclose(a, b), "hand position was discarded"

    turned = base.copy()
    angle = np.pi / 2
    c, s = np.cos(angle), np.sin(angle)
    turned[:, :2] = base[:, :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)

    a = encode_hand(Hand(landmarks=base))[FRAME_START:]
    b = encode_hand(Hand(landmarks=turned))[FRAME_START:]
    assert not np.allclose(a[2:], b[2:]), "hand orientation was discarded"


def test_frame_block_mirrors_a_left_hand_consistently():
    """Folding a left hand has to happen before the directions are read.

    Reading the directions first and mirroring afterwards flips their x while
    leaving the canonical block alone, so a left hand and a right hand encode to
    different vectors and the model sees two encodings for one physical sign.
    """

    base = make_hand(curl=0.5)
    left = base.copy()
    left[:, 0] = 1.0 - left[:, 0]

    mirrored = frame_features(left, Handedness.LEFT)
    assert np.allclose(mirrored, frame_features(base, Handedness.RIGHT), atol=1e-3)

    # And it has to agree with the encoder that consumes it, not just with itself.
    assert np.allclose(
        encode_hand(Hand(landmarks=left, handedness=Handedness.LEFT)),
        encode_hand(Hand(landmarks=base, handedness=Handedness.RIGHT)),
        atol=1e-3,
    )


def test_frame_block_direction_columns_are_unit_length():
    """The four direction columns are (cos, sin) pairs and must stay normalised.

    Augmentation rotates them, and a pair that drifts off the unit circle would
    quietly start encoding something other than direction.
    """

    frame = encode_hand(Hand(landmarks=make_hand(curl=0.3)))[FRAME_START:]
    wrist_x, wrist_y = frame[0], frame[1]
    assert 0.0 <= wrist_x <= 1.0
    assert 0.0 <= wrist_y <= 1.0

    for pair in ((2, 3), (4, 5)):
        cos, sin = frame[pair[0]], frame[pair[1]]
        assert abs(float(np.hypot(cos, sin)) - 1.0) < 1e-3


def test_left_and_right_hands_encode_identically():
    """MediaPipe reports handedness, so the two can be folded into one space."""

    base = make_hand(curl=0.5)
    left = base.copy()
    left[:, 0] = 1.0 - left[:, 0]

    right_features = encode_hand(Hand(landmarks=base, handedness=Handedness.RIGHT))
    left_features = encode_hand(Hand(landmarks=left, handedness=Handedness.LEFT))
    assert np.abs(right_features - left_features).max() < 1e-3


def test_curl_rises_monotonically():
    """A straighter finger must read as less curled, or the derived features are
    inverted and the model learns backwards."""

    previous = -1.0
    for curl in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = float(encode_hand(Hand(landmarks=make_hand(curl=curl)))[COORD_DIMS + 1])
        assert value > previous
        previous = value


def test_absent_hand_encodes_as_zeros():
    """Training pads missing hands with zeros, so inference must do the same."""

    assert not encode_frame([]).any()
    assert encode_frame([]).shape == (FEATURE_DIMS,)


def test_sequence_window_right_aligns():
    """A window longer than the track keeps the newest frames and drops the
    oldest, because a sign is identified by how it settles, not how it started."""

    frames = [np.full(FEATURE_DIMS, i + 1, dtype=np.float32) for i in range(10)]
    window = encode_sequence(frames, 8)

    assert window.shape == (8, FEATURE_DIMS)
    assert window[0][0] == 3.0  # frame index 2, the oldest that still fits
    assert window[-1][0] == 10.0  # newest frame last


def test_sequence_shorter_than_window_is_left_padded():
    frames = [np.ones(FEATURE_DIMS, dtype=np.float32)]
    window = encode_sequence(frames, 8)
    assert window[:7].sum() == 0.0
    assert window[7].all()
