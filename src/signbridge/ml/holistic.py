"""Whole-body landmarks from the MuteMotion archive, and what to feed the model.

Why this exists. The hand-only corpus this project started with gave 21
landmarks per hand and nothing else, so the only reference a sign had was its own
hand. Sign language is not defined that way: a flat hand at the forehead and the
same flat hand at the chest are different signs, and telling them apart needs the
body. The archive carries MediaPipe Holistic output with all 553 joints, which
includes the 33 pose landmarks that supply that missing reference.

Joint layout, as stored. Probed rather than assumed, because getting it wrong
produces a model that trains cleanly and recognises nothing:

    [  0: 21)  left hand
    [ 21: 42)  right hand
    [ 42: 75)  pose, 33 joints, MediaPipe ordering so index 0 is the nose
    [ 75: 543) face, 468 joints
    [543: 553) ten trailing joints of unknown purpose

Only pose and hands are used. The face block is 468 of the 553 dimensions and
carries expression detail that WLASL never labels, so feeding it in mostly adds
noise the model has to learn to ignore, on top of a corpus far too small to
support it.

Pose indices used here, from MediaPipe's own ordering.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("data/mutemotion")

LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
POSE = slice(42, 75)
FACE = slice(75, 543)

JOINTS = 553

# MediaPipe pose indices, relative to the start of POSE.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

#: Dims added on top of the hand-only feature vector.
#: Body frame: 3 (shoulder centre) + 3 (shoulder axis, as a unit vector)
#: Arm: 2 (elbow and wrist in body frame) + 1 (arm extension)
#: Height: 4 (hands relative to nose and shoulders, in body-frame units)
#: Motion: 4 (wrist velocity in body frame, per hand)
POSE_DIMS = 17


class MuteMotion:
    """Reader for landmarks_V3.npz joined to WLASL labels.

    The archive's keys are positional, 0 to 21082, not WLASL video ids. Verified
    rather than trusted: at 570 sampled positions the gloss and frame range in
    `WLASL_parsed_data.json` agree exactly with the official `WLASL_v0.3.json`.
    Joining on video id instead silently matches only 3,309 of 21,083, which
    looks like missing data rather than a wrong join.
    """

    def __init__(self, root: Path = ROOT) -> None:
        self.npz = np.load(root / "landmarks_V3.npz", allow_pickle=True)
        meta = json.loads((root / "WLASL_parsed_data.json").read_text(encoding="utf-8"))
        if len(meta) != len(self.npz.files):
            raise ValueError(
                f"metadata has {len(meta)} entries but the archive has "
                f"{len(self.npz.files)} clips, so they cannot be joined positionally"
            )
        self.meta = meta

    def __len__(self) -> int:
        return len(self.meta)

    def labels(self) -> list[str]:
        return [e["gloss"] for e in self.meta]

    def splits(self) -> list[str]:
        return [e["split"] for e in self.meta]

    def video_ids(self) -> list[str]:
        return [
            m.group(1) if (m := re.search(r"(\d+)\.mp4$", e["video_path"])) else ""
            for e in self.meta
        ]

    def clip(self, i: int) -> np.ndarray:
        """(frames, 553, 3) landmarks for one clip, as float32."""

        return np.asarray(self.npz[str(i)], dtype=np.float32)


def _pose(landmarks: np.ndarray) -> np.ndarray:
    """(frames, 33, 3) pose block."""

    return landmarks[:, POSE, :]


def _present(joints: np.ndarray) -> np.ndarray:
    return np.abs(joints).sum(axis=-1) > 1e-6


def body_frame(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shoulder centre, unit shoulder axis, and a validity mask, per frame.

    The shoulder axis is what makes hand height meaningful: "above the head" is a
    statement about a hand's position relative to the shoulders, not about where
    it sits in the picture. When the shoulders cannot be found the frame is
    flagged invalid and the caller leaves the body features at zero, which is the
    same convention used for a missing hand.
    """

    n = len(pose)
    centre = np.zeros((n, 3), dtype=np.float32)
    axis = np.zeros((n, 3), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)

    left = pose[:, L_SHOULDER, :]
    right = pose[:, R_SHOULDER, :]
    have = _present(left) & _present(right)

    for i in np.flatnonzero(have):
        mid = (left[i] + right[i]) * 0.5
        # Down the body is where the hips are, which fixes the vertical.
        hips = (pose[i, L_HIP] + pose[i, R_HIP]) * 0.5
        if not _present(hips[None, :])[0]:
            continue
        down = hips - mid
        norm = float(np.linalg.norm(down))
        if norm < 1e-6:
            continue
        centre[i] = mid
        axis[i] = down / norm
        ok[i] = True

    return centre, axis, ok


def pose_features(landmarks: np.ndarray) -> np.ndarray:
    """(frames, POSE_DIMS) body reference features.

    Everything here is expressed relative to the signer's own shoulders, so a
    person sitting close to the camera and one sitting far away produce the same
    numbers. That is the property the hand-only corpus could not offer at all.
    """

    pose = _pose(landmarks)
    n = len(pose)
    out = np.zeros((n, POSE_DIMS), dtype=np.float32)

    centre, down, ok = body_frame(pose)
    if not ok.any():
        return out

    scale = np.zeros(n, dtype=np.float32)
    shoulder_w = np.zeros(n, dtype=np.float32)
    valid = np.flatnonzero(ok)
    for i in valid:
        w = float(np.linalg.norm(pose[i, L_SHOULDER] - pose[i, R_SHOULDER]))
        if w > 1e-4:
            shoulder_w[i] = w
            scale[i] = w
    scale[~ok] = 1.0

    def _side(wrist_idx: int) -> np.ndarray:
        """Wrist position relative to the shoulder centre, in shoulder widths."""

        wrist = pose[:, wrist_idx, :]
        rel = (wrist - centre) / scale[:, None]
        return np.where(_present(wrist)[:, None], rel, 0.0).astype(np.float32)

    right_rel = _side(R_WRIST)
    left_rel = _side(L_WRIST)

    nose = pose[:, NOSE, :]
    nose_rel = np.where(
        _present(nose)[:, None], (nose - centre) / scale[:, None], 0.0
    ).astype(np.float32)

    idx = valid
    out[idx, 0:3] = centre[idx]
    out[idx, 3:6] = down[idx]
    out[idx, 6:9] = right_rel[idx]
    out[idx, 9:12] = left_rel[idx]
    # Extension is how far the wrist sits from the shoulder centre, in shoulder
    # widths. A hand at rest sits near 1, a fully extended arm beyond 2.
    out[idx, 12] = np.linalg.norm(right_rel[idx], axis=-1)
    out[idx, 13] = np.linalg.norm(left_rel[idx], axis=-1)
    # Height above the shoulder line, measured along the body's own down axis.
    out[idx, 14] = np.einsum("ij,ij->i", right_rel[idx], down[idx])
    out[idx, 15] = np.einsum("ij,ij->i", left_rel[idx], down[idx])
    out[idx, 16] = np.einsum("ij,ij->i", nose_rel[idx], down[idx])
    return out


def pose_velocity(features: np.ndarray) -> np.ndarray:
    """Frame-to-frame change of the body features, per hand.

    Sign is movement. The static features say where the hand is, this says where
    it is going, and many WLASL words differ only in the second.
    """

    out = np.zeros_like(features)
    if len(features) < 2:
        return out
    out[1:] = features[1:] - features[:-1]
    return out