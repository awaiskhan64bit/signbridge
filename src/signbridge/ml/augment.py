"""Skeleton-space augmentation for training.

These transforms live in the joint frame rather than on encoded features,
because the graph needs real coordinates to convolve over, and because a feature
vector cannot be mirrored or rotated meaningfully.

Every transform here corresponds to something a different signer or camera would
genuinely do. That is the test a transform has to pass: not "does it change the
numbers" but "does it produce a pose a signer could actually be in". A transform
that only satisfies the first produces a model that is confidently wrong.

`amount` scales how often each transform fires, so a training sweep can dial
augmentation down without editing this module.
"""

from __future__ import annotations

import numpy as np

from .skeleton import (
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    POSE_START,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)

#: Left and right pose joints exchanged under mirroring, MediaPipe indices.
_PAIRS_A = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP]
_PAIRS_B = [R_SHOULDER, L_SHOULDER, R_ELBOW, L_ELBOW, R_WRIST, L_WRIST, R_HIP, L_HIP]

#: A track shorter than this cannot be meaningfully cropped.
MIN_FRAMES = 8


def augment(
    joints: np.ndarray, rng: np.random.Generator, amount: float = 1.0
) -> np.ndarray:
    """Vary a batch of body-relative joint tracks.

    `joints` is (N, T, V, 3) and the result has the same shape. `amount` of zero
    returns the input untouched, which is what a no-augmentation ablation wants.
    """

    if joints.ndim != 4:
        raise ValueError(f"expected (N, T, V, 3), got shape {joints.shape}")
    t = joints.shape[1]
    if t < MIN_FRAMES or amount <= 0:
        return joints

    out = joints.astype(np.float32)

    # Time crop, then resample back to the same length, so the sign is replayed
    # faster or slower rather than merely truncated. Truncation would change
    # which part of the sign is present, which changes the label.
    keep = max(MIN_FRAMES, int(t * (1.0 - amount * rng.uniform(0.0, 0.45))))
    start = int(rng.integers(0, t - keep + 1))
    idx = np.clip(
        np.round(np.linspace(start, start + keep - 1, t)).astype(int), 0, t - 1
    )
    out = out[:, idx]

    # In-plane rotation, standing in for camera roll and a tilted signer.
    #
    # x is copied first. Assigning straight into out[..., 0] and then reading
    # out[..., 0] back for the y update rotates by the angle twice, because the
    # second line sees the value the first line just wrote.
    if rng.random() < 0.7 * amount:
        ang = rng.uniform(-0.25, 0.25)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        x = out[..., 0].copy()
        out[..., 0] = x * cos_a - out[..., 1] * sin_a
        out[..., 1] = x * sin_a + out[..., 1] * cos_a

    # Per-joint jitter plus a small global scale, for tracking noise and body
    # proportion. Jitter is independent per joint because MediaPipe's error is
    # not correlated across the skeleton.
    scale = 1.0 + amount * (rng.uniform(0.9, 1.1) - 1.0)
    out = out * scale + rng.normal(0.0, 0.02 * amount, size=out.shape)

    if rng.random() < 0.5 * amount:
        out = mirror(out)

    # Frame dropout: the tracker losing the hands for a moment. Zeroing the whole
    # clip is the same convention the missing-hand case already uses.
    if rng.random() < 0.35 * amount:
        dead = rng.random(out.shape[0]) < 0.05 * amount
        out[dead] = 0.0

    return out.astype(np.float16)


def mirror(joints: np.ndarray) -> np.ndarray:
    """Reflect the skeleton and exchange its left and right halves.

    Reflection alone would leave the acting hand on the same side of the image
    wearing the other hand's anatomy. Exchanging alone would move it without
    turning it over. Both together are what a different signer performing the
    same sign actually looks like.

    `joints` is (N, T, V, 3) and the result has the same shape.
    """

    out = joints.copy()
    out[..., 0] *= -1.0

    hands = out[:, :, :POSE_START].copy()
    hands[:, :, :21] = out[:, :, 21:42]
    hands[:, :, 21:42] = out[:, :, :21]
    out[:, :, :POSE_START] = hands

    body = out[:, :, POSE_START:].copy()
    body[:, :, _PAIRS_A] = body[:, :, _PAIRS_B]
    out[:, :, POSE_START:] = body
    return out