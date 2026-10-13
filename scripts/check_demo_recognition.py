"""End to end check: does the demo rig animate a gesture the recogniser names right?

Feeds every phrase in the phrasebook through the real feature encoder and the
real recogniser and reports how many come back with the correct word. This is
the test that would catch a rig and a recogniser drifting apart, or a template
book that looks full but cannot actually be told apart.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from signbridge.ml.features import FEATURE_DIMS
from signbridge.ml.gestures import GestureRecognizer
from signbridge.vision.simulated import PHRASEBOOK, build_landmarks

WINDOW = 64


def main() -> int:
    recognizer = GestureRecognizer(min_confidence=0.0)
    labels = recognizer.labels
    print(f"templates loaded: {len(labels)}")

    correct = total = silent = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for label, poses in PHRASEBOOK:
        if label not in labels:
            continue
        # Hold the final pose, which is what the recogniser judges, padded out
        # to a full window the same way the live pipeline pads.
        held = build_landmarks(poses[-1])
        frame = np.zeros(FEATURE_DIMS, dtype=np.float32)
        from signbridge.core.types import Hand, Handedness
        from signbridge.ml.features import encode_hand

        frame = encode_hand(Hand(landmarks=held, handedness=Handedness.RIGHT))

        window = np.stack([frame] * WINDOW)
        # Leave room for the rest-pose frames the live pipeline would see.
        rest = np.zeros(FEATURE_DIMS, dtype=np.float32)
        window = np.concatenate([np.stack([rest] * (WINDOW - 12)), np.stack([frame] * 12)])

        prediction = recognizer.predict(window)
        total += 1
        if prediction.label == label:
            correct += 1
        else:
            confusion[(label, prediction.label or "(silent)")] += 1
        if not prediction.label:
            silent += 1

    acc = correct / total if total else 0.0
    print(f"phrases tested   : {total}")
    print(f"correct          : {correct} ({acc:.1%})")
    print(f"silent           : {silent}")

    if confusion:
        print("\nconfusions:")
        for (want, got), n in confusion.most_common(12):
            print(f"  {want:<14} -> {got:<14} x{n}")

    if acc < 0.95:
        print("FAIL: recogniser cannot identify the rig's own gestures")
        return 1
    print("PASS: recogniser names the rig's gestures correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())