"""Tests for skeleton-space augmentation.

Each test pins one transform with a stub generator so the transform under test is
the only thing acting. Stubbing matters: a stub that lets every random draw fire
also enables frame dropout, which zeroes the track and makes an unrelated
assertion fail with a confusing message.
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.ml import skeleton as sk
from signbridge.ml.augment import MIN_FRAMES, augment, mirror
from signbridge.ml.holistic import L_SHOULDER, NOSE, R_SHOULDER

#: Reflect x only; y and z are unchanged by a left-right reflection.
REFLECT_X = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


class Stub:
    """Answers only what a transform asks for.

    `yes` lists the 1-based positions of the `random()` calls that should return
    0.0 (transform on). Everything else declines, so exactly one transform acts.
    `angle` is returned by `uniform` when the low bound is negative, which is how
    the rotation angle is requested; a positive low bound is the time-crop
    fraction, which returns 1.0 so nothing is cropped.
    """

    def __init__(self, yes: int = (), angle: float = 0.3):
        self.yes = set(yes)
        self.angle = angle
        self.calls = 0

    def random(self, *shape):
        self.calls += 1
        return 0.0 if self.calls in self.yes else 1.0

    def uniform(self, low, high):
        # Three different draws are made, and the bounds tell them apart:
        # the rotation angle is uniform(-0.25, 0.25), the crop fraction is
        # uniform(0.0, 0.45), and the scale factor is uniform(0.9, 1.1).
        # Returning 0.0 for the crop keeps every frame; returning 1.0 for the
        # scale leaves it unchanged.
        if low < 0:
            return self.angle
        return 1.0 if high > 1.0 else 0.0

    def integers(self, low, high):
        return 0

    def normal(self, *shape, **kw):
        return 0.0


def _track(frames: int = 32, batch: int = 1) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=(batch, frames, sk.JOINTS, 3)).astype(np.float16)


class TestRotation:
    def test_applied_exactly_once(self):
        # The regression: x was written and then read back for the y update, so
        # the coordinates were rotated twice. Comparing against one explicit
        # matrix product is the only way to catch that, because a double rotation
        # still produces plausible-looking coordinates.
        ang = 0.3
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        rot = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        track = _track()
        got = augment(track.copy(), Stub(yes=(1,), angle=ang), amount=1.0)
        np.testing.assert_allclose(
            got.astype(np.float32), track.astype(np.float32) @ rot.T, atol=1e-2
        )

    def test_preserves_joint_lengths(self):
        track = _track()
        got = augment(track.copy(), Stub(yes=(1,)), amount=1.0).astype(np.float32)
        np.testing.assert_allclose(
            np.linalg.norm(got, axis=-1),
            np.linalg.norm(track.astype(np.float32), axis=-1),
            atol=1e-2,
        )


class TestMirror:
    def test_is_an_involution(self):
        track = _track()
        np.testing.assert_allclose(mirror(mirror(track)), track, atol=1e-3)

    def test_exchanges_hands(self):
        track = _track()
        got = mirror(track).astype(np.float32)
        np.testing.assert_allclose(
            got[:, :, :21], track[:, :, 21:42].astype(np.float32) * REFLECT_X, atol=1e-2
        )

    def test_exchanges_body_sides(self):
        track = _track()
        got = mirror(track).astype(np.float32)
        np.testing.assert_allclose(
            got[:, :, sk.POSE_START + L_SHOULDER],
            track[:, :, sk.POSE_START + R_SHOULDER].astype(np.float32) * REFLECT_X,
            atol=1e-2,
        )

    def test_reflects_but_does_not_exchange_the_nose(self):
        # The nose is the midline, so it reflects and stays put.
        track = _track()
        got = mirror(track).astype(np.float32)
        np.testing.assert_allclose(
            got[:, :, sk.POSE_START + NOSE],
            track[:, :, sk.POSE_START + NOSE].astype(np.float32) * REFLECT_X,
            atol=1e-2,
        )

    def test_does_not_mutate_its_input(self):
        track = _track()
        before = track.copy()
        mirror(track)
        np.testing.assert_array_equal(track, before)


class TestAugment:
    def test_rejects_a_track_that_is_not_batched(self):
        with pytest.raises(ValueError, match="expected"):
            augment(np.zeros((32, sk.JOINTS, 3), dtype=np.float32), Stub())

    def test_shape_is_preserved(self):
        track = _track()
        for seed in range(5):
            got = augment(track.copy(), np.random.default_rng(seed), amount=1.0)
            assert got.shape == track.shape

    def test_zero_amount_is_a_no_op(self):
        track = _track()
        got = augment(track.copy(), np.random.default_rng(0), amount=0.0)
        np.testing.assert_array_equal(got, track)

    def test_output_is_finite(self):
        track = _track()
        for seed in range(5):
            got = augment(track.copy(), np.random.default_rng(seed), amount=1.0)
            assert np.isfinite(got).all()

    def test_short_tracks_pass_through(self):
        track = _track(frames=MIN_FRAMES - 1)
        np.testing.assert_array_equal(
            augment(track.copy(), np.random.default_rng(0), amount=1.0), track
        )

    def test_clips_in_a_batch_are_augmented_independently(self):
        # Crop, rotation and dropout all draw per clip. If they were shared, one
        # unlucky clip would be zeroed whenever any other clip was, which quietly
        # deletes a fraction of the training set every step.
        batch = _track(batch=8)
        got = augment(batch.copy(), np.random.default_rng(0), amount=1.0)
        alive = (np.abs(got.astype(np.float32)).sum(axis=(1, 2, 3)) > 0)
        assert alive.any(), "every clip was dropped at once"