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
    encode_frame,
    encode_hand,
    encode_sequence,
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
def test_invariant_to_translation_scale_and_rotation(curl):
    """Where the hand is, how big it is and how the camera is rolled must not
    change the encoding. SignBridge has to work from a fixed webcam at a desk.

    All three axes are transformed together. Scaling x and y but not z would be
    a non-uniform squash, not a change of camera distance, and MediaPipe's depth
    channel does scale with x and y.
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
    assert np.abs(a - b).max() < 1e-3


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
