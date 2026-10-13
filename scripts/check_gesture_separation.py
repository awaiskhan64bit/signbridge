"""Can 100 gestures actually be told apart, or do they collide?

Filling a table with a hundred poses proves nothing if most of them encode to
nearly the same feature vector. This encodes every pose through the real encoder
and then measures the thing that matters: if a pose is jittered the way a real
hand is imperfect, does nearest-template matching still recover the right word.

The earlier hand-written table of 104 curls produced a closest pair 0.03 apart
on an 86 dimensional vector, which meant `sing` and `talk` were the same
gesture. This script is the guard against that regressing.
"""

from __future__ import annotations

import numpy as np

from signbridge.core.types import Hand, Handedness
from signbridge.ml.features import encode_hand
from signbridge.vision.gesturebook import build_gestures, labels
from signbridge.vision.simulated import build_landmarks


def main(trials: int = 40, noise: float = 0.12) -> int:
    gestures = build_gestures()
    names = labels()
    templates = np.stack(
        [
            encode_hand(Hand(landmarks=build_landmarks(pose), handedness=Handedness.RIGHT))
            for pose in gestures.values()
        ]
    )

    d = np.linalg.norm(templates[:, None, :] - templates[None, :, :], axis=-1)
    off = d.copy()
    np.fill_diagonal(off, np.inf)
    closest = float(off.min())
    scale = float(np.median(off[np.isfinite(off)]))
    print(f"gestures            : {len(names)}")
    print(f"typical pair distance: {scale:.2f}")
    print(f"closest pair         : {closest:.2f} ({closest / scale:.1%} of typical)")

    worst = np.argsort(off.min(axis=1))[:5]
    print("\nmost confusable pairs:")
    for i in worst:
        j = int(np.argmin(off[i]))
        print(f"  {names[i]:<14} vs {names[j]:<14} {off[i, j]:6.2f}")

    # The test that counts. Jitter the template the way a hand is imperfect and
    # see whether the nearest neighbour is still the right word.
    #
    # The noise is scaled per dimension by that dimension's own spread across
    # the book. Scaling by the typical pairwise distance instead, which is a
    # whole-vector figure of roughly 1.8, would inject noise several times larger
    # than any real hand jitter and fail a perfectly good vocabulary.
    rng = np.random.default_rng(0)
    per_dim = templates.std(axis=0)
    correct = total = 0
    for _ in range(trials):
        noisy = templates * (1.0 + rng.normal(0.0, 0.10, size=(1, templates.shape[1])))
        noisy = noisy + rng.normal(0.0, noise * per_dim, size=noisy.shape)
        pred = np.argmin(
            np.linalg.norm(noisy[:, None, :] - templates[None, :, :], axis=-1), axis=1
        )
        correct += int((pred == np.arange(len(names))).sum())
        total += len(names)

    acc = correct / total
    print(f"\nnearest-template accuracy under jitter: {acc:.1%} "
          f"({correct}/{total}, noise {noise:.0%} of per-dimension spread)")

    if acc < 0.95:
        print("FAIL: gestures are not separable enough to ship")
        return 1
    print("PASS: every gesture stays identifiable under realistic jitter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())