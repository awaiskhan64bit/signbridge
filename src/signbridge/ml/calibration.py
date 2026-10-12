"""Confidence calibration.

A word level classifier with six training clips per word will be wrong often.
Accuracy hides that, because most of its errors are on clips it would never have
spoken anyway. What a user actually experiences is:

    of the signs the app decided were confident enough to say, how many were right

That is a precision and coverage trade, and this module finds the useful point
on it. It lives outside the training script so the threshold can be recomputed
from an existing model without retraining, which is what
`scripts/evaluate.py --write-threshold` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Coverage at which the coverage penalty stops biting. Below this, precision is
#: discounted in proportion to how little of the input is being accepted, so
#: "correct on the two easiest clips" cannot beat "correct on a third of them".
COVERAGE_TARGET = 0.35


@dataclass(frozen=True, slots=True)
class Calibration:
    threshold: float
    precision: float
    coverage: float
    score: float

    def describe(self) -> str:
        return (
            f"threshold {self.threshold:.2f}: accepts {self.coverage:.1%} of clips "
            f"at {self.precision:.1%} precision"
        )


def _sweep(probs: np.ndarray, labels: np.ndarray) -> Calibration:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    order = np.argsort(-confidence)

    best = Calibration(0.5, 0.0, 0.0, -1.0)
    running = 0.0
    total = len(order)

    for k in range(1, total + 1):
        running += correct[order[k - 1]]
        coverage = k / total
        precision = running / k
        score = precision * min(1.0, coverage / COVERAGE_TARGET)
        if score > best.score:
            best = Calibration(
                threshold=float(confidence[order[k - 1]]),
                precision=float(precision),
                coverage=float(coverage),
                score=float(score),
            )
    return best


def calibrate(probs: np.ndarray, labels: np.ndarray) -> Calibration:
    """Find the confidence cut to ship.

    `probs` must be the same distribution the app will see, so this has to be
    fitted on held out clips, never on the training set.
    """

    if len(probs) == 0:
        raise ValueError("cannot calibrate on an empty set")
    return _sweep(np.asarray(probs, dtype=np.float64), np.asarray(labels))


def precision_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Coverage and precision at a fixed cut, for reporting a chosen threshold."""

    confidence = probs.max(axis=1)
    keep = confidence >= threshold
    if not keep.any():
        return 0.0, 0.0
    return float(keep.mean()), float((probs.argmax(axis=1)[keep] == labels[keep]).mean())


def curve(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[tuple[float, float, float]]:
    """(threshold, coverage, precision) at a set of cuts, for a report table."""

    out = []
    for t in thresholds or [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
        coverage, precision = precision_at(probs, labels, t)
        out.append((t, coverage, precision))
    return out
