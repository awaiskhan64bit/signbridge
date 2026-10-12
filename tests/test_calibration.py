"""Tests for confidence calibration.

The shipped default threshold comes out of this, and a bug here does not crash
anything: it just ships a threshold that makes the app either mute or
overconfident. So it gets tested properly.
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.ml.calibration import COVERAGE_TARGET, calibrate, curve, precision_at


def separable(n: int = 2000, n_classes: int = 10, seed: int = 0):
    """A set where confident predictions are reliably correct and the rest are
    noise. The shape a well behaved classifier's output should have."""

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, n)
    probs = rng.dirichlet(np.ones(n_classes) * 0.2, n)
    probs[np.arange(n), labels] += rng.random(n) * 0.8
    return probs, labels


def test_prefers_broad_coverage_over_a_lucky_few():
    """The original bug compared a score against a stored accuracy, so it latched
    onto the first tiny coverage point it found and stopped. With mostly wrong
    predictions, accepting only the single luckiest clip must not win."""

    probs, labels = separable()
    fitted = calibrate(probs, labels)

    assert fitted.coverage > 0.05, "settled on a vanishingly small coverage"
    assert fitted.threshold < 1.0, "threshold of 1.0 means it accepts nothing"
    assert fitted.precision >= (probs.argmax(axis=1) == labels).mean()


def test_prefers_broad_coverage_when_precision_is_equal():
    """When every cut has the same precision, the widest one that still clears
    the coverage target wins. Ties must not resolve to the smallest cut."""

    rng = np.random.default_rng(3)
    labels = rng.integers(0, 4, 500)
    probs = rng.random((500, 4)) * 0.2
    probs[np.arange(500), labels] += 0.6  # everything is correct and confident

    fitted = calibrate(probs, labels)
    assert fitted.precision == pytest.approx(1.0)
    assert fitted.coverage >= COVERAGE_TARGET


def test_precision_at_matches_a_direct_count():
    probs, labels = separable(n=500, seed=5)
    threshold = 0.5
    coverage, precision = precision_at(probs, labels, threshold)

    confidence = probs.max(axis=1)
    keep = confidence >= threshold
    assert coverage == pytest.approx(keep.mean())
    assert precision == pytest.approx((probs.argmax(axis=1)[keep] == labels[keep]).mean())


def test_precision_at_reports_zero_when_nothing_passes():
    probs, labels = separable(n=100, seed=7)
    assert precision_at(probs, labels, 1.5) == (0.0, 0.0)


def test_empty_input_is_refused():
    with pytest.raises(ValueError):
        calibrate(np.zeros((0, 3)), np.zeros(0, dtype=int))


def test_curve_is_ordered_and_monotone():
    probs, labels = separable(seed=9)
    points = curve(probs, labels, [0.1, 0.3, 0.5, 0.7, 0.9])

    assert [p[0] for p in points] == [0.1, 0.3, 0.5, 0.7, 0.9]
    coverages = [p[1] for p in points]
    # A higher threshold can only ever accept less.
    assert coverages == sorted(coverages, reverse=True)
    assert COVERAGE_TARGET > 0


def test_random_guessing_has_no_usable_threshold():
    """With no signal, every cut should be about as bad as the rest. If
    calibration reports a confident threshold here, it is fitting noise."""

    rng = np.random.default_rng(11)
    n, n_classes = 3000, 50
    probs = rng.dirichlet(np.ones(n_classes), n)
    labels = rng.integers(0, n_classes, n)

    fitted = calibrate(probs, labels)
    overall = float((probs.argmax(axis=1) == labels).mean())
    assert fitted.precision < overall + 0.05
