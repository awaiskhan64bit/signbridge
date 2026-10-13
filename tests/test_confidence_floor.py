"""The app must not speak below the confidence the model was calibrated to.

A 31% accurate model thresholded at 0.20 speaks confidently and wrongly, which
is the worst outcome for a speech app: worse than silence, because the user
cannot tell a guess from a fact. Calibration picked 0.62, and the confidence
slider was free to ignore it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from signbridge.ml.features import FEATURE_DIMS
from signbridge.ml.neural import NeuralRecognizer

META = {
    "name": "WLASL-500",
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


def test_low_confidence_is_suppressed_even_when_the_slider_is_low(recogniser):
    """The exact bug: slider at 0.20, model well under its bar, previously spoken."""

    rec = NeuralRecognizer.load(recogniser, min_confidence=0.20)
    prediction = rec.predict(_window())

    # The graph emits logits of (0.30, 0.0), so softmax puts 0.57 on the winner.
    # That is below the calibrated 0.62, so it must stay silent.
    assert prediction.confidence == pytest.approx(0.574, abs=0.01)
    assert prediction.label == "", "spoke a guess despite sitting under the calibrated bar"


def test_speaks_once_the_confidence_clears_the_calibrated_bar(recogniser):
    rec = NeuralRecognizer.load(recogniser, min_confidence=0.20)
    # Same graph, but the caller raises the bar above the model, so it is allowed.
    rec.min_confidence = 0.10
    rec._threshold = 0.0
    assert rec.predict(_window()).label == "hello"


def test_slider_can_only_tighten_not_loosen(recogniser):
    rec = NeuralRecognizer.load(recogniser, min_confidence=0.05)

    assert rec.default_threshold == pytest.approx(0.62)
    assert rec.effective_threshold == pytest.approx(0.62)

    rec.min_confidence = 0.90
    assert rec.effective_threshold == pytest.approx(0.90), "slider must be able to tighten"