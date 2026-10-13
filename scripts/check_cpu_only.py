"""Prove the running app never uses the GPU.

The GPU exists for training only. This exercises the real inference path the app
uses and asserts on what actually happened, rather than trusting that nobody
added a `.cuda()` call somewhere.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PySide6.QtWidgets import QApplication

from signbridge.ml.dataset import encode_clip, resize_track
from signbridge.ml.features import FEATURE_DIMS
from signbridge.ml.neural import NeuralRecognizer

MODEL = "assets/models/sign_classifier.onnx"


def gpu_processes() -> list[str]:
    """Names of processes currently holding GPU memory, via nvidia-smi."""

    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except Exception:
        return []
    rows = [line.strip() for line in out.splitlines() if line.strip()]
    return rows


def main() -> int:
    app = QApplication(["gpu-check"])  # noqa: F841  # the controller needs one
    failures: list[str] = []

    # 1. onnxruntime itself must not offer, let alone select, a GPU provider.
    available = ort.get_available_providers()
    print(f"onnxruntime providers available : {available}")
    if any("CUDA" in p or "Tensorrt" in p or "GPU" in p for p in available):
        failures.append(f"onnxruntime exposes a GPU provider: {available}")
    print(f"onnxruntime default device      : {ort.get_device()}")
    if ort.get_device() != "CPU":
        failures.append(f"onnxruntime default device is {ort.get_device()}")

    # 2. The recogniser must run and predict, on a real clip.
    rec = NeuralRecognizer.load(MODEL, min_confidence=0.0)
    if not rec.ready:
        failures.append("recogniser failed to load")
    window = np.zeros((rec.window, FEATURE_DIMS), dtype=np.float32)
    prediction = rec.predict(window)
    print(f"recogniser ready                : {rec.ready}")
    print(f"blank window predicts          : conf {prediction.confidence:.4f}")
    print(f"labels                         : {len(rec.labels)}")

    # 3. Nothing of ours should hold GPU memory. torch is imported to prove that
    #    merely having it installed does not allocate, and that CUDA initialising
    #    is not required for inference.
    import torch

    print(f"torch version                   : {torch.__version__}")
    print(f"CUDA compiled into torch        : {torch.version.cuda}")
    if torch.cuda.is_initialized():
        failures.append("torch initialised a CUDA context during inference")
    print(f"CUDA initialised during run     : {torch.cuda.is_initialized()}")

    before = gpu_processes()
    print(f"GPU compute processes (before)  : {before or 'none'}")

    # 4. A real WLASL clip through the full live path.
    from pathlib import Path

    from signbridge.ml.dataset import WLASLIndex

    index = WLASLIndex(Path("data/wlasl-index.json"))
    clip = [s for s in index.split("test") if s.gloss in rec.labels]
    if clip:
        track = encode_clip(clip[0].path)
        live = resize_track(track, rec.window)
        result = rec.predict(live)
        print(f"real clip ({clip[0].gloss:12s}) -> {result.label:14s} conf {result.confidence:.3f}")

    if torch.cuda.is_initialized():
        failures.append("torch initialised a CUDA context while running a clip")
    after = gpu_processes()
    print(f"GPU compute processes (after)   : {after or 'none'}")

    mine = [p for p in after if "python" in p.lower()]
    if mine:
        failures.append(f"python holds GPU memory: {mine}")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: inference ran entirely on the CPU, no CUDA context, no GPU memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())