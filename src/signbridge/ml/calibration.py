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

#: The share of clips a cut must accept for the app to be usable at all.
#:
#: A recognizer that never fires is not a recognizer. Calibrating purely for
#: precision on a weak model walks the threshold off the top of the confidence
#: range and past the confidence the model actually gives a correct answer: the
#: shipped model peaks at 0.58 on a real "hello" clip, with the right word on
#: top, and a cut of 0.75 left it silent on every frame of it. Requiring some
#: real coverage forces the cut to sit somewhere the model can actually speak,
#: and the precision that costs is then reported rather than hidden.
#: The share of clips a cut must accept for the app to be usable at all.
#:
#: A recognizer that never fires is not a recognizer. Calibrating purely for
#: precision on a weak model walks the threshold off the top of the confidence
#: range and past the confidence the model actually gives a correct answer: the
#: shipped model peaks at 0.58 on a real "hello" clip, with the right word on
#: top, and a cut of 0.75 left it silent on every frame of it. Requiring some
#: real coverage forces the cut to sit somewhere the model can actually speak,
#: and the precision that costs is then reported rather than hidden.
MIN_COVERAGE = 0.25


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


def _sweep(probs: np.ndarray, labels: np.ndarray) -> tuple[Calibration, Calibration | None]:
    """Return the strict cut and the usability fallback.

    The strict cut maximises precision times coverage among cuts that clear both
    the precision floor and the sample floor. The sample floor stops a cut being
    chosen on the strength of two clips, which is how a noisy tail at 0.95
    outranks the real knee at 0.50.

    The fallback is the loosest cut that still accepts MIN_COVERAGE of clips. It
    is returned separately rather than being blended in, because whether it is
    acceptable depends on how far above chance the model is, and that is a
    question about the number of classes rather than about the sweep.
    """

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    order = np.argsort(-confidence)

    total = len(order)
    min_count = max(3, round(MIN_SAMPLES * total))
    best = Calibration(1.0, 0.0, 0.0, -1.0)
    usable: Calibration | None = None
    running = 0.0

    for k in range(1, total + 1):
        running += correct[order[k - 1]]
        coverage = k / total
        precision = running / k
        cut = float(confidence[order[k - 1]])

        if usable is None and coverage >= MIN_COVERAGE:
            usable = Calibration(cut, precision, coverage, 0.0)

        # Deliberately not a break. A looser cut averages over every clip above it
        # plus the ones between, so precision at a looser cut is a running mean
        # that dips and recovers rather than falling monotonically. Stopping at the
        # first dip discarded every cut below it, which is where the real operating
        # point lives: on the shipped model the sweep gave up at the very first clip
        # and answered "stay silent".
        if precision < MIN_PRECISION or k < min_count:
            continue
        if coverage < MIN_COVERAGE:
            # Usability is a requirement, not a tie break. A cut that clears the
            # precision floor while speaking on a sixth of clips still never fires
            # on a real sign: the shipped model peaks at 0.58 confidence and a cut
            # of 0.75 is silent on every frame of it. Precision is what the cut has
            # to clear to be allowed to speak, not a licence to speak on almost
            # nothing.
            continue

        score = precision * min(1.0, coverage / COVERAGE_TARGET)
        if score > best.score:
            best = Calibration(cut, precision, coverage, score)

    return best, usable


def calibrate(probs: np.ndarray, labels: np.ndarray) -> Calibration:
    """Find the confidence cut to ship.

    `probs` must be the same distribution the app will see, so this has to be
    fitted on held out clips, never on the training set.

    Two rules can conflict. The precision floor says stay quiet rather than
    guess. The coverage floor says a cut nothing can reach is not a decision at
    all. Precision wins when the model is genuinely hopeless and coverage wins
    when it is merely weak, which is what the chance comparison is for: on a 200
    word problem 2% precision is ten times better than guessing, while on a five
    word problem it would be worse than guessing.
    """

    if len(probs) == 0:
        raise ValueError("cannot calibrate on an empty set")
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)

    strict, usable = _sweep(probs, labels)
    if strict.score >= 0.0:
        return strict
    if usable is None:
        return strict

    chance = 1.0 / max(2, probs.shape[1])
    if usable.precision < max(MIN_PRECISION / 2.0, chance * 2.0):
        return strict
    return usable


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
