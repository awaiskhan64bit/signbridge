"""Skeleton graph representation for sign recognition.

Why a graph and not a vector. Every published result that beats a flat
sequence model on WLASL convolves over the skeleton itself. Knowing that a wrist
joins to an elbow which joins to a shoulder is precisely what separates one sign
from another, and flattening the joints into a feature vector throws that
structure away before the network sees it. The reference architecture for this
is a temporal graph convolution network with multi-head attention, which reaches
over 85% on WLASL using 55 pose keypoints; this module supplies the same kind of
input from a different landmark source.

Joint layout, 75 joints per frame:

    [ 0:21)  left hand, MediaPipe order
    [21:42)  right hand, MediaPipe order
    [42:75)  pose, MediaPipe order, index 0 is the nose

The face landmarks in the corpus are deliberately excluded. They are 468 of the
553 stored joints, they carry expression detail that WLASL never labels, and
including them would quadruple the input width for signal the task does not use.

Everything is expressed relative to the signer's own shoulders. A person close
to the camera and one far away then produce the same numbers, and "the hand is
above the head" becomes a statement about the body rather than about the frame.
That is the normalisation WLASL signers do not need and a laptop webcam does,
and skipping it is what makes a model trained here fail on live input.

Bones are encoded alongside coordinates. Two joints move together when they are
joined, and the bone vector makes that explicit where raw coordinates only imply
it. Four streams (joint, joint motion, bone, bone motion) are what the reference
work aggregates, and they are cheap.
"""

from __future__ import annotations

import numpy as np

from .holistic import (
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    LEFT_HAND,
    NOSE,
    POSE,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    RIGHT_HAND,
)

#: Total joints carried per frame.
JOINTS = 75
#: pose joints start here, after the two hands.
POSE_START = 42

#: Hand skeleton in MediaPipe order, wrist outward then finger by finger.
_HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

#: Pose bones, MediaPipe indices, relative to POSE_START.
_POSE_BONES = (
    (L_SHOULDER, R_SHOULDER),          # shoulder line, the scale reference
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),     # left arm
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),     # right arm
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),     # torso
    (L_HIP, R_HIP),                     # hips, fixes the body frame
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),       # head, so the nose has a direction
    # MediaPipe's coarse hand proxies. They are the only pose joints with no edge
    # elsewhere in the tree, so without these six nodes sit in their own component
    # and never exchange information with anything.
    (L_WRIST, 17), (L_WRIST, 18), (L_WRIST, 19),
    (R_WRIST, 20), (R_WRIST, 21), (R_WRIST, 22),
)

#: The rest of MediaPipe's pose tree: face detail, and the legs below the hips.
#:
#: None of these joints is read as a feature anywhere, but every node in the graph
#: has to be reachable or the adjacency has a disconnected component and those
#: nodes contribute nothing while still costing channels, parameters and a share
#: of the attention. Connecting them to their parent in the pose tree keeps the
#: graph connected without promoting them to something the model relies on.
_POSE_TREE = (
    (0, 1), (0, 4), (1, 2), (2, 3), (2, 6), (3, 5),      # face
    (0, 7), (0, 8),                                        # ears
    (10, 0), (9, 0),                                       # mouth, anchored to the nose
    (L_HIP, 25), (R_HIP, 26), (25, 27), (26, 28),          # upper legs
    (27, 29), (27, 31), (28, 30), (28, 32),                # shins and feet
)


def _edges() -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in _HAND_BONES:
        out.append((a, b))
        out.append((21 + a, 21 + b))
    for a, b in _POSE_BONES:
        out.append((POSE_START + a, POSE_START + b))
    for a, b in _POSE_TREE:
        out.append((POSE_START + a, POSE_START + b))

    # Attach each hand to its own arm.
    #
    # Without these the two hand subtrees are separate connected components from
    # the pose, and a graph convolution can never move information between them.
    # The pose would then contribute nothing to the hands, which defeats the
    # reason pose was added at all: "flat hand at the forehead" and the same flat
    # hand at the chest differ only by where the hand is relative to the body.
    #
    # Two attachment points per hand, elbow and wrist, because one edge is a
    # single point of failure if that landmark is misdetected in a frame.
    for hand_wrist, elbow, wrist in ((0, L_ELBOW, L_WRIST), (21, R_ELBOW, R_WRIST)):
        out.append((hand_wrist, POSE_START + elbow))
        out.append((hand_wrist, POSE_START + wrist))
    return out


EDGES = _edges()

#: One parent per joint, or -1 for a root. This is what the bone stream is built
#: from: one bone vector per joint rather than one node per bone.
#:
#: Giving each joint its own bone, instead of giving each bone its own node, keeps
#: all four streams on the same 75-node axis. The alternative looks natural and is
#: wrong: there are 77 bones and 75 joints, so bone nodes cannot be padded onto
#: the joint axis without either dropping two bones or overflowing the adjacency,
#: and a graph whose node count disagrees with its tensor is a shape error waiting
#: to happen. A tree gives exactly one parent per joint, so the two agree by
#: construction.
_HAND_PARENT = (
    (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)
)
#: Parent per MediaPipe pose joint, in MediaPipe's own indexing:
#: 0 nose, 1-6 eyes, 7-8 ears, 9-10 mouth, 11-16 arms, 17-22 hand proxies,
#: 23-24 hips, 25-28 knees and ankles, 29-32 heels and feet.
_POSE_PARENT = (
    -1, 0, 1, 2, 0, 4, 5, 0, 0, 0, 9,        # 0-10 head
    NOSE, NOSE,                              # 11-12 shoulders
    L_SHOULDER, R_SHOULDER,                   # 13-14 elbows
    L_ELBOW, R_ELBOW,                         # 15-16 wrists
    L_WRIST, L_WRIST, L_WRIST, R_WRIST, R_WRIST, R_WRIST,   # 17-22 hand proxies
    L_SHOULDER, R_SHOULDER,                   # 23-24 hips
    L_HIP, R_HIP,                             # 25-26 knees
    L_HIP, R_HIP,                             # 27-28 ankles
    25, 26, 27, 28,                           # 29-32 heels and feet
)


def _shift(block: tuple[int, ...], offset: int) -> list[int]:
    """Offset every parent, but leave a root at -1.

    Offsetting -1 would turn a root into an unrelated joint, which silently
    invents a parent bone on a joint that has none.
    """

    return [p + offset if p >= 0 else -1 for p in block]


PARENT = np.asarray(
    list(_HAND_PARENT)
    + _shift(_HAND_PARENT, 21)
    + _shift(_POSE_PARENT, POSE_START),
    dtype=np.int64,
)

# A miscounted tuple here silently truncates the skeleton, so it is checked once
# at import rather than discovered as a shape error three modules later.
assert len(_HAND_PARENT) == 21, "each hand block has 21 joints"
assert len(_POSE_PARENT) == 33, "MediaPipe pose has 33 joints"
assert PARENT.shape == (JOINTS,), f"parent array is {PARENT.shape}, expected ({JOINTS},)"

#: True where a joint has no parent, so its bone vector is defined as zero.
_IS_ROOT = PARENT < 0


#: Source and target index arrays for every bone, resolved once at import.
_BONE_A = np.asarray([i for i, _ in EDGES])
_BONE_B = np.asarray([j for _, j in EDGES])


def adjacency() -> np.ndarray:
    """Symmetric normalised adjacency with self-loops, shape (JOINTS, JOINTS).

    Normalising by the degree keeps the aggregation from amplifying as the graph
    gets denser. Self-loops guarantee a non-zero degree, so the inverse square
    root needs no masked power, which would leave its output uninitialised where
    the mask is false.
    """

    a = np.zeros((JOINTS, JOINTS), dtype=np.float32)
    for i, j in EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    a += np.eye(JOINTS, dtype=np.float32)
    deg = a.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    return (a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


def _present(points: np.ndarray) -> np.ndarray:
    return np.abs(points).sum(axis=-1) > 1e-6


def _used(clip: np.ndarray) -> np.ndarray:
    """Trim a 553-joint corpus frame to the 75 this model uses.

    Live capture assembles its 75 joints directly, so a track that is already the
    right width is passed through rather than being re-sliced. The face block is
    dropped here rather than downstream so nothing later has to remember to skip
    it.
    """

    if clip.shape[1] == JOINTS:
        return clip
    return np.concatenate(
        [clip[:, LEFT_HAND, :], clip[:, RIGHT_HAND, :], clip[:, POSE, :]], axis=1
    )


def assemble(
    left_hand: np.ndarray | None, right_hand: np.ndarray | None, pose: np.ndarray | None
) -> np.ndarray:
    """One body frame as (JOINTS, 3), with zeros where nothing was detected.

    Slots are by handedness, not by detection order, because that is how the
    corpus is laid out: the left hand always occupies [0, 21) and the right hand
    always occupies [21, 42). Putting whichever hand happened to be detected
    first into slot zero would swap the two roughly half the time, and the model
    would read a right-handed signer as a left-handed one on half of all frames.
    That failure is invisible in aggregate accuracy and obvious to a user.

    Handedness is MediaPipe's anatomical label for the signer, which is the same
    convention the corpus uses. Detecting from a horizontally flipped image flips
    the coordinates but not the anatomical label, which is why hands and pose
    must be extracted from the same image in the same pass.
    """

    out = np.zeros((JOINTS, 3), dtype=np.float32)
    if left_hand is not None:
        arr = np.asarray(left_hand, dtype=np.float32)
        if arr.shape == (21, 3):
            out[0:21] = arr
    if right_hand is not None:
        arr = np.asarray(right_hand, dtype=np.float32)
        if arr.shape == (21, 3):
            out[21:42] = arr
    if pose is not None:
        arr = np.asarray(pose, dtype=np.float32)
        if arr.shape == (33, 3):
            out[POSE_START:] = arr
    return out


def normalise_body(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body-relative (T, 75, 3), and a per-frame valid mask.

    Translation, scale and the viewer's distance all cancel: coordinates are
    expressed in shoulder widths along axes built from the shoulders and hips.
    Frames with no detectable body keep the raw coordinates and are masked, so a
    dropped pose is never read as a hand in a plausible position.
    """

    clip = _used(clip)
    joints = clip.shape[0]
    pose = clip[:, POSE_START:, :]
    out = clip.copy()
    valid = np.zeros(joints, dtype=bool)

    left_sh = pose[:, L_SHOULDER, :]
    right_sh = pose[:, R_SHOULDER, :]
    left_hip = pose[:, L_HIP, :]
    right_hip = pose[:, R_HIP, :]

    have = (
        _present(left_sh) & _present(right_sh) & _present(left_hip) & _present(right_hip)
    )
    idx = np.flatnonzero(have)
    if len(idx) == 0:
        return out, valid

    centre = (left_sh[idx] + right_sh[idx]) * 0.5
    hip = (left_hip[idx] + right_hip[idx]) * 0.5
    width = np.linalg.norm(left_sh[idx] - right_sh[idx], axis=-1)
    scale = np.where(width > 1e-4, width, 1.0)

    # Down the body fixes vertical, and across the shoulders fixes horizontal.
    # Together they are a body frame that does not rotate with the camera roll.
    down = hip - centre
    down = down / np.maximum(np.linalg.norm(down, axis=-1, keepdims=True), 1e-6)
    across = left_sh[idx] - right_sh[idx]
    across = across / np.maximum(np.linalg.norm(across, axis=-1, keepdims=True), 1e-6)
    fwd = np.cross(down, across, axis=-1)

    basis = np.stack([across, down, fwd], axis=1)  # (n, 3, 3)
    rel = (clip[idx] - centre[:, None, :]) / scale[:, None, None]
    out[idx] = np.einsum("nij,nkj->nki", basis, rel)
    valid[idx] = True
    return out, valid


def build_input(clip: np.ndarray, window: int = 64) -> np.ndarray:
    """One clip into (window, JOINTS, 3), resampled and body-relative."""

    joints, valid = normalise_body(clip)
    # Resample in time so every clip is the same length.
    src = np.linspace(0, len(joints) - 1, window)
    idx = np.clip(np.round(src).astype(int), 0, len(joints) - 1)
    out = np.zeros((window, JOINTS, 3), dtype=np.float32)
    seen = valid[idx]
    out[seen] = joints[idx][seen]
    return out


#: Source and target index arrays for every bone, resolved once at import.
_BONE_A = np.asarray([i for i, _ in EDGES])
_BONE_B = np.asarray([j for _, j in EDGES])


def bone_stream(joints: np.ndarray) -> np.ndarray:
    """One bone vector per joint: `joints[..., i] - joints[..., parent(i)]`.

    Roots have no parent and get zero, which is the same convention already used
    for a joint that was not detected. The result has the same shape as the
    input, so all four streams share one node axis and nothing needs padding.
    """

    out = np.zeros_like(joints)
    child = np.flatnonzero(~_IS_ROOT)
    out[..., child, :] = joints[..., child, :] - joints[..., PARENT[child], :]
    return out


def motion(stream: np.ndarray, axis: int = -3) -> np.ndarray:
    """Frame-to-frame difference along the time axis, zeroed on the first frame.

    The axis defaults to -3 rather than 0 because a batch of tracks has its frame
    axis there, and differencing along 0 would instead subtract one clip from the
    next, which trains silently and learns nothing.
    """

    head = np.take(stream, np.array([0]), axis=axis)
    return np.diff(stream, axis=axis, prepend=head)


def four_streams(clip: np.ndarray, window: int = 64) -> dict[str, np.ndarray]:
    """The four inputs the reference work aggregates.

    Joints say where everything is, motion says how it moved, bones say which
    parts move together, and bone motion says how that coordination changes. All
    four cost little and together they carry most of what separates two signs
    performed by different people.
    """

    j = build_input(clip, window)
    b = bone_stream(j)
    return {
        "j": j,
        "jm": motion(j),
        "b": b,
        "bm": motion(b),
    }