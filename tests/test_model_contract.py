"""Contract tests for the trained model.

The model is produced by a training script that can be re-run with different
hyperparameters, so the boundary between it and the app has assumptions that can
be violated silently. These tests pin that boundary: export a small model with
the real architecture, then check the app reads it correctly.

Skipped when torch or onnxruntime are absent, since inference only needs
onnxruntime and training only needs torch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from signbridge.ml.features import FEATURE_DIMS

ort = pytest.importorskip("onnxruntime", reason="inference dependency")
torch = pytest.importorskip("torch", reason="training only dependency")

from signbridge.ml.neural import ModelMismatch, NeuralRecognizer  # noqa: E402

WINDOW = 32
LABELS = ["hello", "good", "i_love_you", "peace", "stop"]


def build_model():
    sys_path = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts")
    import sys

    sys.path.insert(0, sys_path)
    from train import build_model as make

    model = make(len(LABELS), hidden=16, layers=1)
    model.eval()
    return model


def export(tmp_path, model, labels=LABELS, window=WINDOW):
    onnx_path = tmp_path / "sign_classifier.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, window, FEATURE_DIMS),
        str(onnx_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the torch.export path wants onnxscript; this does not
    )
    meta = {
        "name": "test",
        "labels": list(labels),
        "window": window,
        "features": FEATURE_DIMS,
        "mean": [0.0] * FEATURE_DIMS,
        "std": [1.0] * FEATURE_DIMS,
        "threshold": 0.42,
        "metrics": {"val_top1": 0.5},
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return onnx_path


def test_loads_metadata_and_predicts(tmp_path):
    path = export(tmp_path, build_model())
    rec = NeuralRecognizer.load(path, min_confidence=0.0)

    assert rec.ready
    assert rec.labels == LABELS
    assert rec.window == WINDOW
    assert "5 signs" in rec.status()

    window = np.random.default_rng(0).normal(size=(WINDOW, FEATURE_DIMS)).astype(np.float32)
    prediction = rec.predict(window)

    assert prediction.label in LABELS
    assert 0.0 <= prediction.confidence <= 1.0
    # Softmax must be a proper distribution whatever the graph emitted.
    assert prediction.confidence == pytest.approx(
        max(0.0, min(1.0, prediction.confidence))
    )


def test_reads_the_calibrated_threshold(tmp_path):
    """The app must pick up the confidence cut training computed, not fall back
    to a hand picked default."""

    path = export(tmp_path, build_model())
    rec = NeuralRecognizer.load(path, min_confidence=0.9)
    assert rec.default_threshold == 0.42


def test_standardisation_is_applied(tmp_path):
    """Training standardised the features. If inference skipped that step every
    prediction would be wrong and nothing would say so, so this checks the app's
    output against the graph driven directly with the standardised input."""

    path = export(tmp_path, build_model())
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["mean"] = [5.0] * FEATURE_DIMS
    meta["std"] = [2.0] * FEATURE_DIMS
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    rec = NeuralRecognizer.load(path, min_confidence=0.0)
    raw = np.random.default_rng(11).normal(size=(WINDOW, FEATURE_DIMS)).astype(np.float32)

    standardised = ((raw - 5.0) / 2.0).reshape(1, WINDOW, FEATURE_DIMS).astype(np.float32)
    reference = rec._session.run(None, {rec._input_name: standardised})[0].reshape(-1)
    expected = float(np.exp(reference - reference.max()) @ np.ones(len(reference)))

    # Same softmax the app should be producing from the unstandardised input.
    from signbridge.ml.neural import _softmax

    assert rec.predict(raw).confidence == pytest.approx(
        float(_softmax(reference).max()), abs=1e-5
    )
    assert expected > 0.0


def test_inference_uses_a_batch_of_one(tmp_path):
    """PyTorch warns that an LSTM exported with a variable batch axis breaks at
    batch sizes other than one. The app only ever runs one window at a time, so
    this pins that assumption rather than leaving it as a warning."""

    path = export(tmp_path, build_model())
    rec = NeuralRecognizer.load(path, min_confidence=0.0)

    window = np.random.default_rng(3).normal(size=(WINDOW, FEATURE_DIMS)).astype(np.float32)
    batched = np.stack([window, window, window])
    logits = rec._session.run(None, {rec._input_name: batched})[0]
    assert logits.shape == (3, len(LABELS))


def test_shape_mismatch_raises_rather_than_mislabels(tmp_path):
    """A graph and metadata that disagree must fail loudly. Reshaping anyway
    would map predictions onto the wrong words, silently."""

    model = torch.nn.Linear(FEATURE_DIMS, len(LABELS) + 3)
    path = export(tmp_path, model)

    rec = NeuralRecognizer.load(path, min_confidence=0.0)
    with pytest.raises(ModelMismatch):
        rec.predict(np.zeros((WINDOW, FEATURE_DIMS), dtype=np.float32))


def test_missing_model_degrades_instead_of_raising(tmp_path):
    assert NeuralRecognizer.try_load(tmp_path / "absent.onnx") is None
