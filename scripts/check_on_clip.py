"""Measure how often the shipped model is right on a real clip, at the shipped cut.

Top-1 over a held-out split is the research number. What a user actually
experiences is "how often does this thing say something, and is it right". This
plays a clip through the real capture path and reports both.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

from signbridge.config import Settings
from signbridge.core.types import Hand
from signbridge.ml.features import FEATURE_DIMS, encode_hand
from signbridge.ml.neural import NeuralRecognizer
from signbridge.vision.hands import HandTracker

MODEL = Path("assets/models/sign_classifier.onnx")
WINDOW = 64
MIN_FRAMES = 8


def main() -> int:
    # A clip is passed in rather than searched for. This used to glob the
    # author's own Downloads folder for a personal video file, which put a
    # username and a private filename into the repository.
    if len(sys.argv) < 2:
        print("usage: check_on_clip.py <video file>")
        print("  Runs the shipped model over a recorded clip and reports what it")
        print("  would have said on every frame.")
        return 2
    video = sys.argv[1]
    if not Path(video).exists():
        print(f"no such file: {video}")
        return 1
    print(f"clip        : {Path(video).name}")

    rec = NeuralRecognizer.load(MODEL, min_confidence=0.0)
    print(f"model       : {rec.name}, {len(rec.labels)} words")
    print(f"calibrated  : {rec.default_threshold:.2f}")
    print(f"effective   : {rec.effective_threshold:.2f}")
    print()

    tracker = HandTracker(Settings.load().models_dir / "hand_landmarker.task")
    if not tracker.available:
        print(f"hand tracker unavailable: {tracker.error}")
        return 1
    cap = cv2.VideoCapture(video)
    track: list[list[float]] = []
    spoken: Counter[str] = Counter()
    peak = ("", 0.0)
    frames = hands_seen = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        found = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.0)
        frames += 1
        if found:
            hands_seen += 1
            track.append(
                encode_hand(
                    Hand(landmarks=found[0].landmarks, handedness=found[0].handedness)
                )
            )
        else:
            track.append([0.0] * FEATURE_DIMS)

        if len(track) >= MIN_FRAMES:
            window = track[-WINDOW:]
            pad = [[0.0] * FEATURE_DIMS] * max(0, WINDOW - len(window))
            result = rec.predict(pad + window)
            if result.confidence > peak[1]:
                peak = (result.label, result.confidence)
            if result.label:
                spoken[result.label] += 1

    cap.release()
    tracker.close()

    print(f"frames          : {frames}  (hand visible in {hands_seen})")
    print(f"peak confidence : {peak[1]:.3f}  -> {peak[0] or 'below every cut'}")
    print(f"speaks at       : {rec.effective_threshold:.2f}")
    print(f"frames spoken   : {sum(spoken.values())}")
    if spoken:
        print("it would have said:")
        for label, count in spoken.most_common(10):
            print(f"  {label:24s} x{count}")
    else:
        print("it would have said nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())