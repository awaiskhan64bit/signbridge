"""The confidence slider has to be able to loosen the cut, not only tighten it.

This reverses an earlier decision and the reason is worth recording. The slider
was clamped from below by the model's calibrated threshold, on the reasoning that a
looser setting is not a trade but simply ignoring the calibration. In practice
the calibration landed on a noisy tail: the shipped model measured 63.6% precision
at 0.90, then 42.9% at 0.95, and the sweep chose 0.98. Clamped there, the app was
stuck at a cut that spoke almost nothing, and the user could not loosen it even
when the correct sign was visibly on top.

What is kept is the part that was genuinely protecting the user: the calibrated
cut is still reported as the default, and every prediction still carries the
confidence of what it decided to say, so a loosened setting shows its own
low confidence rather than claiming certainty.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from signbridge.ml.features import FEATURE_DIMS
from signbridge.ml.neural import NeuralRecognizer

META = {
    "name": "WLASL-200",
    "labels": ["hello", "thank_you"],
    "window": 8,
    "features": FEATURE_DIMS,
    "threshold": 0.62,
    "mean": [0.0] * FEATURE_DIMS,
    "std": [1.0] * FEATURE_DIMS,
}


@pytest.fixture
def recogniser(tmp_path):
    """A recogniser whose graph is a constant, so confidence is predictable."""

    import onnxruntime as ort
    import torch

    onnx_path = tmp_path / "sign_classifier.onnx"

    class Constant(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, x):
            # Zero out a real element of the input so the exporter keeps it as a
            # graph input rather than constant folding the whole tensor away.
            head = x[:, 0, 0:1] * 0.0
            return torch.cat([head + self.value, head + 0.0], dim=1)

    torch.onnx.export(
        Constant(0.30),
        torch.zeros(1, META["window"], FEATURE_DIMS),
        str(onnx_path),
        input_names=["window"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    assert ort.get_available_providers(), "onnxruntime unavailable"
    (tmp_path / "sign_classifier.json").write_text(json.dumps(META), encoding="utf-8")
    return onnx_path


def _window() -> np.ndarray:
    return np.zeros((META["window"], FEATURE_DIMS), dtype=np.float32)


def test_the_slider_can_loosen_below_the_calibrated_cut(recogniser):
    """The bug: correct on top, but the calibrated 0.62 silenced it anyway."""

    rec = NeuralRecognizer.load(recogniser, min_confidence=0.20)
    prediction = rec.predict(_window())

    # The graph emits logits of (0.30, 0.0), so softmax puts 0.574 on the winner.
    assert prediction.confidence == pytest.approx(0.574, abs=0.01)
    assert prediction.label == "hello", "a loosened slider must be able to speak"


def test_the_slider_can_still_tighten(recogniser):
    rec = NeuralRecognizer.load(recogniser, min_confidence=0.95)
    assert rec.predict(_window()).label == "", "a raised bar must still be able to silence"


def test_the_calibrated_cut_is_still_reported_as_the_default(recogniser):
    """Loosening is a choice the user makes, not the state the app starts in."""

    rec = NeuralRecognizer.load(recogniser, min_confidence=0.05)
    assert rec.default_threshold == pytest.approx(0.62)


def test_a_loosened_setting_still_reports_its_confidence(recogniser):
    # The protection against a loose slider is not the floor, it is that the
    # number shown next to the answer is the real one. A wrong answer under a
    # loosened slider has to look wrong.
    rec = NeuralRecognizer.load(recogniser, min_confidence=0.20)
    prediction = rec.predict(_window())
    assert prediction.confidence < 0.62
    assert prediction.display


def test_the_cut_is_clamped_to_a_usable_range(recogniser):
    # Out of range values would otherwise make the app permanently mute or
    # permanently talk, neither of which is recoverable from the UI.
    rec = NeuralRecognizer.load(recogniser, min_confidence=-5.0)
    assert 0.0 <= rec.effective_threshold <= 1.0
    rec.min_confidence = 99.0
    assert 0.0 <= rec.effective_threshold <= 1.0