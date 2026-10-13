"""Isolate why the pose corpus trains so badly.

The run scored 3.5% top-1, below what a deliberately broken encoder managed, so
the cause is more likely a defect than a weak signal. The clean way to tell is to
hold everything fixed and vary one thing: train the same model, same splits, same
corpus, with the pose columns zeroed out.

If the hand-only slice of this corpus scores normally, the corpus and the splits
are fine and the pose columns are the problem. If it scores badly too, the fault
is in the pipeline rather than in the features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("scripts")))
sys.path.insert(0, str(Path("src")))
from train_pose import main as train_main

FEATURE_DIMS = 86


def main() -> int:
    from signbridge.ml import holistic_dataset as H

    tracks, _labels, _splits, _ = H.load()

    print("zeroing pose columns to isolate them from the hand block\n")

    stripped = []
    for t in tracks:
        out = t.copy()
        out[:, FEATURE_DIMS:] = 0.0
        stripped.append(out)

    # Hand the stripped corpus to the trainer through the module it reads.
    H.CACHE = Path("data/mutemotion/tracks_handonly.npz")
    np.savez(H.CACHE, tracks=np.array(stripped, dtype=object), allow_pickle=True)

    sys.argv = ["train_pose.py", "--epochs", "25", "--patience", "8",
                "--device", "cuda", "--vocab", "500"]
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())