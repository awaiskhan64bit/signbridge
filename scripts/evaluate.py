"""Evaluate the trained classifier and produce the numbers in docs/results.md.

Training prints accuracy. This prints the things that actually decide whether
the app is usable: how often it is right *when it chooses to speak*, and what
that costs in coverage. Run it after training, or on any model in assets/models.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --thresholds 0.1 0.3 0.5 0.7 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signbridge.ml.dataset import WLASLIndex, resize_track  # noqa: E402
from signbridge.ml.features import FEATURE_DIMS  # noqa: E402
from signbridge.ml.neural import NeuralRecognizer  # noqa: E402

DATA = ROOT / "data"
MODEL = ROOT / "assets" / "models" / "sign_classifier.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    args = parser.parse_args()

    if not MODEL.exists():
        print("No trained model. Run scripts/train.py first.", file=sys.stderr)
        return 1

    meta = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))
    labels = meta["labels"]
    window = meta["window"]
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)

    rec = NeuralRecognizer.load(MODEL, min_confidence=0.0)
    print(f"model      : {meta['name']}")
    print(f"classes    : {len(labels)}")
    print(f"parameters : {meta['metrics'].get('params', 0):,}")
    print(f"window     : {window} frames of {FEATURE_DIMS} features")

    index = WLASLIndex(DATA / "wlasl-index.json")
    known = set(labels)
    samples = [s for s in index.split("test") if s.gloss in known]
    print(f"test clips : {len(samples)} (of {len(index.split('test'))} in the split)")

    from signbridge.ml.dataset import encode_clip

    tracks = [encode_clip(s.path) for s in samples]
    tracks = [(t - mean) / std for t in tracks]
    y = np.array([labels.index(s.gloss) for s in samples])

    probs = np.zeros((len(tracks), len(labels)), dtype=np.float32)
    for start in range(0, len(tracks), 256):
        chunk = np.stack([resize_track(t, window) for t in tracks[start : start + 256]])
        logits = rec._session.run(None, {rec._input_name: chunk.astype(np.float32)})[0]
        probs[start : start + len(chunk)] = _softmax(logits)

    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top5 = order[:, :5]
    confidence = probs.max(axis=1)

    print()
    print(f"top-1 accuracy : {(top1 == y).mean():.4f}")
    print(f"top-5 accuracy : {(top5 == y[:, None]).any(axis=1).mean():.4f}")

    print("\nprecision when it chooses to speak")
    print(f"{'threshold':>10} {'coverage':>9} {'precision':>10}")
    thresholds = args.thresholds or [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        keep = confidence >= t
        if not keep.any():
            print(f"{t:>10.2f} {'0%':>9} {'-':>10}")
            continue
        precision = float((top1[keep] == y[keep]).mean())
        print(f"{t:>10.2f} {keep.mean():>8.1%} {precision:>9.1%}")

    # Accuracy by how many training clips the word had. This is where the long
    # tail shows up, and it is the honest picture of what the model can do.
    train_counts = Counter(s.gloss for s in index.split("train"))
    buckets = [(1, 3, "1 to 3 clips"), (4, 8, "4 to 8 clips"), (9, 10**6, "9 or more")]
    print("\naccuracy by how much data the word had")
    print(f"{'clips':>14} {'words':>6} {'top-1':>8}")
    for lo, hi, name in buckets:
        words = {g for g, c in train_counts.items() if lo <= c <= hi and g in known}
        if not words:
            continue
        mask = np.array([s.gloss in words for s in samples])
        print(f"{name:>14} {len(words):>6} {(top1[mask] == y[mask]).mean():>7.1%}")

    print("\nworst confusions (true -> predicted, most frequent)")
    confusions: Counter = Counter()
    for t, p in zip(y, top1, strict=True):
        if t != p:
            confusions[(labels[t], labels[p])] += 1
    for (truth, guess), count in confusions.most_common(8):
        print(f"  {truth:>22} -> {guess:<22} {count}")

    return 0


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    sys.exit(main())
