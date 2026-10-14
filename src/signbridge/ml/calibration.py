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

#: Precision a cut must reach before it is allowed to speak at all.
#:
#: Without a floor the sweep maximises precision times coverage, and coverage is
#: always positive, so the best it can do is pick the least-bad cut rather than
#: stay silent. On a 500 word model scoring 2.7% top-1 that produced a cut of
#: 0.245 at 4% precision: the app announced a guess four times in five, which
#: for speech is far worse than saying nothing. A cut that cannot beat this is
#: not a cut, so the answer becomes 1.0 and the app stays quiet.
MIN_PRECISION = 0.30

#: A cut is only trusted if at least this many clips reach it.
#:
#: Precision measured on a handful of clips is noise. The measured curve for the
#: shipped model runs 19.6% at 0.20, 28.6% at 0.50, 63.6% at 0.90 and then falls
#: back to 42.9% at 0.95, purely because two clips sit above 0.95 and one of them
#: is wrong. Without a floor on sample count the sweep picked 0.98 off that
#: wobble, which both spoke almost nothing and spoke it wrongly.
MIN_SAMPLES = 0.03


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
    """Pick the cut that maximises precision times coverage, with two floors.

    The precision floor stops a hopeless model from speaking at all. The sample
    floor stops a cut from being chosen on the strength of a couple of clips,
    which is how a noisy tail at 0.95 outranks the real knee at 0.50.

    Among the cuts that clear both floors, the best one is taken rather than the
    strictest, because precision and coverage trade against each other: the
    highest precision on this data is always the cut that says almost nothing.
    """

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    order = np.argsort(-confidence)

    total = len(order)
    min_count = max(3, round(MIN_SAMPLES * total))
    best = Calibration(1.0, 0.0, 0.0, -1.0)
    running = 0.0

    for k in range(1, total + 1):
        running += correct[order[k - 1]]
        coverage = k / total
        precision = running / k
        if precision < MIN_PRECISION:
            # Deliberately not a break. A looser cut averages over every clip above
            # it plus the ones between, so precision at a looser cut is a running
            # mean that dips and recovers rather than falling monotonically.
            # Stopping at the first dip therefore discarded every cut below it,
            # which is where the real operating point lives: on the shipped model
            # the sweep gave up at the very first clip and answered "stay silent".
            continue
        if k < min_count:
            # Too few clips to trust the precision estimate yet.
            continue
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
