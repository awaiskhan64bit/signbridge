"""ONNX wrapper around the trained sign classifier.

Inference deliberately avoids PyTorch. The training script needs it, but the app
only needs to run a small bidirectional LSTM or TCN, and onnxruntime does that
from a few megabytes of weights with no framework in the way.

The bundle on disk is two files that sit next to each other:

    sign_classifier.onnx      the graph
    sign_classifier.json      labels, window length, feature dims, metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.types import SignPrediction
from .base import Recognizer


class ModelNotFound(FileNotFoundError):
    pass


class NeuralRecognizer(Recognizer):
    """Runs the trained classifier over a window of encoded landmarks."""

    name = "sign classifier"

    def __init__(self, model_path: str | Path, min_confidence: float = 0.62) -> None:
        self.model_path = Path(model_path)
        self.meta_path = self.model_path.with_suffix(".json")
        self.min_confidence = min_confidence

        self._session = None
        self._labels: list[str] = []
        self._input_name = ""
        self._window = 48
        self._features = 80
        self._metrics: dict = {}
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._threshold = 0.0

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def load(cls, model_path: str | Path, min_confidence: float = 0.62) -> NeuralRecognizer:
        rec = cls(model_path, min_confidence)
        rec._open()
        return rec

    def _open(self) -> None:
        if not self.model_path.exists():
            raise ModelNotFound(f"No model at {self.model_path}")
        if not self.meta_path.exists():
            raise ModelNotFound(f"No metadata at {self.meta_path}")

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._labels = list(meta["labels"])
        self._window = int(meta.get("window", 48))
        self._features = int(meta.get("features", 80))
        self._metrics = meta.get("metrics", {})
        self.name = meta.get("name", "sign classifier")

        # Training standardised the features, so inference has to as well, and
        # it carries the confidence cut that maximised accepted accuracy.
        mean = meta.get("mean")
        std = meta.get("std")
        if mean and std:
            self._mean = np.asarray(mean, dtype=np.float32)
            self._std = np.asarray(std, dtype=np.float32)
        self._threshold = float(meta.get("threshold") or 0.0)

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    @classmethod
    def try_load(cls, model_path: str | Path, min_confidence: float = 0.62):
        """Load without raising, so a missing model degrades instead of crashing."""

        try:
            return cls.load(model_path, min_confidence)
        except Exception:
            return None

    @property
    def ready(self) -> bool:
        return self._session is not None

    @property
    def default_threshold(self) -> float:
        """The confidence cut that maximised accuracy on accepted clips.

        Falls back to the caller's preference when the model predates it.
        """

        return self._threshold or self.min_confidence

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def window(self) -> int:
        return self._window

    def status(self) -> str:
        if not self.ready:
            return "classifier not loaded"
        acc = self._metrics.get("val_top1")
        suffix = f", {acc:.1%} val" if isinstance(acc, (int, float)) else ""
        return f"{self.name} ({len(self._labels)} signs{suffix})"

    # ------------------------------------------------------------- inference

    def predict(self, window: np.ndarray) -> SignPrediction:
        if self._session is None or not self._labels:
            return SignPrediction(label="", confidence=0.0)

        batch = window.reshape(1, self._window, self._features).astype(np.float32)
        if self._mean is not None and self._std is not None:
            batch = (batch - self._mean) / self._std
        batch = np.ascontiguousarray(batch)

        logits = self._session.run(None, {self._input_name: batch})[0]
        scores = np.asarray(logits, dtype=np.float32).reshape(-1)

        # The graph may already emit probabilities, so only softmax when it does not.
        already_normalised = scores.min() >= -1e-6 and abs(float(scores.sum()) - 1.0) < 1e-2
        if not already_normalised:
            scores = _softmax(scores)

        order = np.argsort(scores)[::-1]
        best = int(order[0])
        confidence = float(scores[best])
        alternatives = [
            (self._labels[i], float(scores[i])) for i in order[1:4] if scores[i] > 0.03
        ]
        return SignPrediction(
            label=self._labels[best] if confidence >= self.min_confidence else "",
            confidence=confidence,
            alternatives=alternatives,
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / (exp.sum() + 1e-9)
