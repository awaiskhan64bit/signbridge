"""A brief sign must fill the window, not share it with the idle time before it.

The window is right aligned, so frames from before the hand appeared land as zeros
on the left. The corpus has a 10th percentile of 34 frames and a median of 59,
while the live window is a fixed 48, so a quick sign arriving into a window that
is already partly idle was being diluted by frames that carry no movement at all.

Training clips are trimmed to the sign, so trimming the idle head off live input
is what makes the two distributions match.
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.core.types import Hand, Handedness
from signbridge.ml.features import (
    FEATURE_DIMS,
    active_span,
    encode_frame,
    encode_sequence,
)


def _frame(fill: float = 1.0) -> np.ndarray:
    return np.full(FEATURE_DIMS, fill, dtype=np.float32)


def _hand() -> Hand:
    return Hand(
        landmarks=np.random.default_rng(0).normal(0.5, 0.05, (21, 3)).astype(np.float32),
        handedness=Handedness.RIGHT,
    )


class TestActiveSpan:
    def test_no_idle_frames_to_drop(self):
        assert active_span([_frame(), _frame()]) == 0

    def test_counts_the_idle_frames_before_the_hand(self):
        assert active_span([np.zeros(FEATURE_DIMS, np.float32)] * 7 + [_frame()]) == 7

    def test_only_the_leading_run_counts(self):
        frames = [_frame(), _frame(), np.zeros(FEATURE_DIMS, np.float32), _frame()]
        assert active_span(frames) == 0, "a gap mid-sign is not idle time before it"

    def test_an_empty_window_is_entirely_idle(self):
        assert active_span([np.zeros(FEATURE_DIMS, np.float32)] * 5) == 5

    def test_a_window_with_nothing_in_it_does_not_crash(self):
        assert active_span([]) == 0


class TestBriefSigns:
    def test_a_brief_sign_is_right_aligned_after_the_idle_head_is_dropped(self):
        idle = [np.zeros(FEATURE_DIMS, dtype=np.float32)] * 33
        brief = [_frame() for _ in range(15)]

        window = encode_sequence(idle + brief, 48)

        assert window.shape == (48, FEATURE_DIMS)
        # The 15 real frames now sit at the right edge, and the leading zeros are
        # padding again rather than 33 frames of idle time that used to be inside
        # the model's field of view.
        assert np.any(window[-15:] != 0.0)
        assert np.all(window[: 48 - 15] == 0.0)

    def test_the_idle_head_does_not_displace_the_sign(self):
        brief = [_frame() for _ in range(15)]
        idle = [np.zeros(FEATURE_DIMS, dtype=np.float32)] * 33

        # The point of the trim: a sign preceded by idle time produces the same
        # window as the sign on its own, so how long the user waited before
        # starting no longer changes what the model sees.
        np.testing.assert_allclose(
            encode_sequence(idle + brief, 48), encode_sequence(brief, 48)
        )

    def test_a_full_window_is_left_alone(self):
        frames = [_frame() for _ in range(48)]
        assert active_span(frames) == 0
        window = encode_sequence(frames, 48)
        assert np.all(window != 0.0)

    def test_a_window_longer_than_the_sign_keeps_the_newest_frames(self):
        frames = [_frame() for _ in range(80)]
        window = encode_sequence(frames, 48)
        assert window.shape == (48, FEATURE_DIMS)
        assert np.all(window != 0.0), "the oldest frames should have fallen off"

    def test_trailing_zeros_are_kept_because_they_mark_the_end_of_the_sign(self):
        frames = [_frame()] * 10 + [np.zeros(FEATURE_DIMS, dtype=np.float32)] * 5
        window = encode_sequence(frames, 48)
        assert np.any(window[-5:] == 0.0), "the hand settling is part of the sign"


class TestShape:
    def test_an_empty_sequence_still_returns_a_full_window(self):
        assert encode_sequence([], 48).shape == (48, FEATURE_DIMS)

    def test_width_comes_from_the_data_not_the_constant(self):
        # The body aware tracks are wider than the hand only encoding. Padding a
        # wide track into a narrow window raises a broadcast error that reads as
        # corrupt data rather than as a wrong constant.
        wide = [np.full(120, 1.0, dtype=np.float32) for _ in range(5)]
        window = encode_sequence(wide, 48)
        assert window.shape == (48, 120)

    def test_encode_frame_is_all_zeros_with_no_hands(self):
        np.testing.assert_array_equal(encode_frame([]), np.zeros(FEATURE_DIMS))

    def test_encode_frame_matches_a_real_hand(self):
        assert np.any(encode_frame([_hand()]) != 0.0)


@pytest.mark.parametrize("idle", [0, 5, 20, 40])
def test_the_window_always_holds_the_whole_sign(idle: int):
    sign = [_frame() for _ in range(12)]
    window = encode_sequence([np.zeros(FEATURE_DIMS, dtype=np.float32)] * idle + sign, 48)
    assert np.all(window[-12:] != 0.0), f"{idle} idle frames displaced part of the sign"