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

from signbridge.ml.calibration import calibrate, curve, precision_at  # noqa: E402
from signbridge.ml.dataset import WLASLIndex, resize_track  # noqa: E402
from signbridge.ml.features import FEATURE_DIMS  # noqa: E402
from signbridge.ml.neural import NeuralRecognizer  # noqa: E402

DATA = ROOT / "data"
MODEL = ROOT / "assets" / "models" / "sign_classifier.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument(
        "--write-threshold",
        action="store_true",
        help="recompute the shipped confidence cut from the validation split "
        "and update the model metadata",
    )
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

    from signbridge.ml.dataset import encode_clip

    cache: dict = {}

    def probability_matrix(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        samples = [s for s in index.split(split) if s.gloss in known]
        tracks = []
        for s in samples:
            if s.video_id not in cache:
                try:
                    cache[s.video_id] = encode_clip(s.path)
                except Exception:
                    cache[s.video_id] = np.zeros((1, FEATURE_DIMS), dtype=np.float32)
            tracks.append((cache[s.video_id] - mean) / std)

        out = np.zeros((len(tracks), len(labels)), dtype=np.float32)
        for start in range(0, len(tracks), 256):
            chunk = np.stack(
                [resize_track(t, window) for t in tracks[start : start + 256]]
            ).astype(np.float32)
            logits = rec._session.run(None, {rec._input_name: chunk})[0]
            out[start : start + len(chunk)] = _softmax(logits)
        y = np.array([labels.index(s.gloss) for s in samples])
        return out, y, [s.gloss for s in samples]

    print(f"test clips : {len([s for s in index.split('test') if s.gloss in known])}")

    if args.write_threshold:
        val_probs, y_val, _ = probability_matrix("val")
        fitted = calibrate(val_probs, y_val)
        test_probs, y_test, _ = probability_matrix("test")
        cov, prec = precision_at(test_probs, y_test, fitted.threshold)
        meta["threshold"] = float(fitted.threshold)
        meta.setdefault("metrics", {})
        meta["metrics"].update(
            accept_accuracy=float(fitted.precision),
            accept_coverage=float(fitted.coverage),
            accept_accuracy_test=float(prec),
            accept_coverage_test=float(cov),
        )
        MODEL.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nwrote threshold {fitted.threshold:.4f} to {MODEL.with_suffix('.json').name}")
        print(f"  val  {fitted.describe()}")
        print(f"  test {cov:.1%} coverage at {prec:.1%} precision")

    probs, y, glosses = probability_matrix("test")
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top5 = order[:, :5]

    print()
    print(f"top-1 accuracy : {(top1 == y).mean():.4f}")
    print(f"top-5 accuracy : {(top5 == y[:, None]).any(axis=1).mean():.4f}")

    print("\nprecision when it chooses to speak")
    print(f"{'threshold':>10} {'coverage':>9} {'precision':>10}")
    for t, cov, prec in curve(probs, y, args.thresholds):
        print(f"{t:>10.2f} {cov:>8.1%} {prec:>9.1%}")

    # Accuracy by how many training clips the word had. This is where the long
    # tail shows up, and it is the honest picture of what the model can do.
    train_counts = Counter(gloss for gloss in (s.gloss for s in index.split("train")))
    buckets = [(1, 3, "1 to 3 clips"), (4, 8, "4 to 8 clips"), (9, 10**6, "9 or more")]
    print("\naccuracy by how much data the word had")
    print(f"{'clips':>14} {'words':>6} {'top-1':>8}")
    for lo, hi, name in buckets:
        words = {g for g, c in train_counts.items() if lo <= c <= hi and g in known}
        if not words:
            continue
        mask = np.array([g in words for g in glosses])
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
