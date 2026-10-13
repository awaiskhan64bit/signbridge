"""Encode the MuteMotion corpus into feature tracks and cache them.

The archive is a single 8.9GB npz that has to be read sequentially, and the
encoder is the slow part, so encoding once and caching is what makes repeated
training runs affordable. On this machine the whole corpus encodes in a few
minutes and the cache is a few hundred megabytes, against 8.9GB to re-read.

A track is the hand features this project already used, with the body features
appended. Keeping the hand block intact means the gesture recogniser and the
existing model contract are untouched; the model simply sees more columns.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..core.types import Hand, Handedness
from .features import FEATURE_DIMS, encode_hand
from .holistic import MuteMotion, pose_features, pose_velocity

ROOT = Path("data/mutemotion")
CACHE = ROOT / "tracks.npz"
INDEX = ROOT / "index.json"

#: Hand features plus body features and their frame-to-frame change.
TRACK_DIMS = FEATURE_DIMS + 17 + 17

_EPS = 1e-6


def encode_one(mm: MuteMotion, i: int) -> np.ndarray:
    """One clip into (frames, TRACK_DIMS)."""

    clip = mm.clip(i)
    pose = pose_features(clip)
    vel = pose_velocity(pose)

    # Body features are only meaningful relative to a visible body, so they are
    # dropped on frames with no pose rather than left as a confident zero.
    pose[pose_velocity(pose)[:, 0] == 0] = 0.0

    tracks = np.zeros((len(clip), TRACK_DIMS), dtype=np.float32)

    # The archive stores both hands; MediaPipe ordering puts them at 0:21 and
    # 21:42. The dominant hand is chosen once over the whole clip, matching the
    # hand-only loader, because switching mid-sequence destroys the temporal
    # structure the model reads movement from.
    left = clip[:, 0:21, :]
    right = clip[:, 21:42, :]
    left_ok = np.abs(left).sum(axis=(1, 2)) > _EPS
    right_ok = np.abs(right).sum(axis=(1, 2)) > _EPS
    if right_ok.sum() >= left_ok.sum():
        hand, present = right, right_ok
    else:
        hand, present = left, left_ok

    for f in np.flatnonzero(present):
        tracks[f, :FEATURE_DIMS] = encode_hand(
            Hand(landmarks=hand[f], handedness=Handedness.RIGHT)
        )
    tracks[:, FEATURE_DIMS:] = np.concatenate([pose, vel], axis=1)
    return tracks


def build(workers: int = 8) -> None:
    mm = MuteMotion()
    n = len(mm)
    print(f"encoding {n} clips", flush=True)

    tracks: list[np.ndarray] = [None] * n  # type: ignore[list-item]

    def one(i: int) -> tuple[int, np.ndarray]:
        return i, encode_one(mm, i)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, track in pool.map(one, range(n), chunksize=64):
            tracks[i] = track
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{n}", flush=True)

    # Variable lengths, so the cache stores objects rather than one array.
    np.savez(CACHE, tracks=np.array(tracks, dtype=object), allow_pickle=True)
    INDEX.write_text(
        json.dumps(
            {
                "labels": mm.labels(),
                "splits": mm.splits(),
                "video_ids": mm.video_ids(),
                "dims": TRACK_DIMS,
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote {CACHE.name} and index.json, {TRACK_DIMS} dims per frame")


def load() -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    """Return (tracks, labels, splits, video_ids)."""

    data = np.load(CACHE, allow_pickle=True)
    meta = json.loads(INDEX.read_text(encoding="utf-8"))
    return (
        [t for t in data["tracks"]],
        meta["labels"],
        meta["splits"],
        meta["video_ids"],
    )


if __name__ == "__main__":
    build()