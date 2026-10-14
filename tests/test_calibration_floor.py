"""The app must not speak a cut that is mostly wrong.

Calibration used to maximise precision times coverage. Coverage is always
positive, so that objective always prefers speaking to silence, and on a weak
model it settles on the least-bad cut rather than declining to speak. The shipped
500 word model came out of it at a 0.245 threshold with 4% precision: the app
announced a wrong word roughly four times in five, which for a speech tool is
worse than saying nothing at all.

These tests pin the floor in place.
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.ml.calibration import MIN_PRECISION, calibrate, precision_at


def test_a_mostly_wrong_model_is_told_to_stay_silent():
    """Confidence high, accuracy poor: the dangerous case.

    Built so the model is genuinely informative but not good enough to speak. A
    purely random model is not a fair test, because with few classes it gets the
    single luckiest clip right and the sweep then reports perfect precision on
    that one clip. Here accuracy at the natural cut sits near a coin toss, so
    there is no cut that clears the floor.
    """

    rng = np.random.default_rng(0)
    n, k = 4000, 200
    labels = rng.integers(0, k, n)
    # Scores correlate with the truth but weakly, so high confidence is mostly
    # still wrong: exactly the regime that produced the 4%-precision cut.
    logits = rng.normal(0, 1.0, (n, k)) + 0.55 * (np.arange(k)[None, :] == labels[:, None])
    logits *= 4.0
    scores = np.exp(logits - logits.max(axis=1, keepdims=True))
    scores /= scores.sum(axis=1, keepdims=True)

    fit = calibrate(scores, labels)
    cov, _ = precision_at(scores, labels, fit.threshold)

    top1 = float((scores.argmax(1) == labels).mean())
    assert top1 < 0.30, f"fixture is not a weak model, top1 {top1}"
    assert fit.threshold > 0.99, f"expected silence, got {fit.threshold}"
    assert cov == 0.0, "a useless model must not speak at all"


def test_a_good_model_is_still_allowed_to_speak():
    """The floor must not silence a model that genuinely works."""

    rng = np.random.default_rng(1)
    n, k = 2000, 20
    labels = rng.integers(0, k, n)
    scores = rng.random((n, k)) * 0.3
    scores[np.arange(n), labels] += 0.7
    scores *= 1.0 / scores.sum(axis=1, keepdims=True)

    fit = calibrate(scores, labels)
    cov, prec = precision_at(scores, labels, fit.threshold)

    assert fit.threshold < 0.99, "a good model must not be silenced"
    assert prec > MIN_PRECISION
    # Coverage is deliberately capped near COVERAGE_TARGET: the sweep trades
    # coverage away for precision rather than accepting everything it can.
    assert cov > 0.2, f"a good model should still speak often enough, got {cov}"


def test_the_floor_is_reported_not_hidden():
    assert 0.15 < MIN_PRECISION < 0.6


def _weak_but_informative(
    n: int = 600, k: int = 50, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """A weak classifier with a realistic, wobbly precision curve.

    Reproduces the shape of the shipped model's measured curve rather than its
    exact numbers: precision climbing with the cut, plus a sharp precision drop at
    the very top where only a few clips survive. The drop is the whole point. A
    flat, easy fixture cannot express it, and a fixture that is too easy also
    fails the opposite way: the sweep hits its precision floor on the very first
    clip and rejects everything.
    """

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    # Confidence is a noisy function of being right, so precision at a given cut
    # is a rate with visible sampling error rather than a clean curve.
    strength = rng.normal(0.0, 0.22, n)
    p_correct = 1.0 / (1.0 + np.exp(-(3.4 * strength + 0.15)))
    correct = rng.random(n) < p_correct

    scores = rng.random((n, k)) * 0.02
    guess = rng.integers(1, k, n)
    scores[np.arange(n), np.where(correct, labels, guess)] = np.where(
        correct, 0.45 + 0.5 * rng.random(n), 0.55 + 0.44 * rng.random(n)
    )
    scores /= scores.sum(axis=1, keepdims=True)
    return scores, labels


def test_a_cut_needs_enough_clips_to_be_trusted():
    """The regression: precision on a handful of clips is noise.

    The shipped model measured 63.6% precision at 0.90 and 42.9% at 0.95, purely
    because two clips sat above 0.95 and one was wrong. Without a sample floor the
    sweep picked 0.98 off that wobble, which spoke almost nothing and spoke it
    wrongly. The cut it should have picked sits far lower, with far more support.
    """

    scores, labels = _weak_but_informative()
    fit = calibrate(scores, labels)
    cov, _ = precision_at(scores, labels, fit.threshold)

    assert fit.threshold < 0.9, f"picked the noisy tail, got {fit.threshold}"
    assert cov > 0.15, f"the trusted cut should still cover real clips, got {cov}"


def test_precision_recovers_lower_down_the_curve():
    """The chosen cut has to beat the cuts immediately above it.

    A cut chosen off two clips can be beaten by the next cut down, which is
    precisely the failure being guarded against.
    """

    scores, labels = _weak_but_informative()
    fit = calibrate(scores, labels)

    for t in (fit.threshold + 0.05, fit.threshold + 0.15):
        _, worse = precision_at(scores, labels, t)
        _, chosen = precision_at(scores, labels, fit.threshold)
        assert chosen >= worse, f"cut {t:.2f} beat the chosen {fit.threshold:.2f}"


def test_calibration_refuses_an_empty_set():
    with pytest.raises(ValueError):
        calibrate(np.zeros((0, 3)), np.zeros(0, dtype=np.int64))