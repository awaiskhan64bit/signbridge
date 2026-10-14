"""Feature encoding for the sign classifier.

A raw frame of 21 MediaPipe landmarks is a noisy model input: it depends on
where the signer stands and how far away they are. Some of that variation has
to come out, but not all of it, and drawing that line in the wrong place is the
single most expensive mistake available in this codebase.

An earlier version normalised each hand all the way to a canonical frame:
translate the wrist to the origin, rotate so the palm points a fixed way, scale
by palm length. That gave a clean rotation, scale and translation invariant
descriptor of hand shape, and it scored 11.2% on a linear probe over WLASL.
Feeding the same probe raw landmarks with their frame position intact scored
20.9%. The invariance was not making the features robust, it was deleting the
label.

The reason is that American Sign Language is not defined by hand shape alone. A
flat hand at the forehead travelling outwards is HELLO. The same flat hand at the
chest travelling inwards is a different sign entirely. Once the encoder has
rotated, scaled and translated the hand, those two signs produce identical
features, and the distinguishing information is gone before the model ever sees
it. No amount of extra training or a bigger network recovers that.

So the encoder does both jobs at once. It keeps a canonical block, which is
stable under camera placement and describes hand shape, plus a frame block
carrying what the canonical block threw away:

    canonical   63   landmark coordinates, invariant to placement
    derived     17   curl, reach, knuckle spread, palm normal
    frame        6   hand position in frame, plus two direction vectors

The frame block is small on purpose. Two reference directions, wrist to middle
knuckle and wrist to index knuckle, pin down the hand's in-plane rotation, and
the wrist position says where in the signing space the hand is doing it. That
is the minimum needed to separate HELLO from GOOD, and it stays robust to camera
angle because the model can weigh it against the rest of the vector.

Apparent hand size is deliberately absent. WLASL is tightly cropped phone
footage where the hand fills the frame, while a laptop webcam shows a whole
upper body, so anything encoding apparent size would be a train-versus-live
mismatch rather than a signal.

This corpus carries no depth channel, so z is zero throughout. That rules out a
frame-space palm normal, which would collapse onto the z axis here and differ
from live input. The two direction vectors are used instead because they are
well defined in 2D and 3D alike.
"""

from __future__ import annotations

import numpy as np

from ..core.types import LANDMARK_COUNT, Hand, Handedness

# Landmark indices, named once so the maths below reads like anatomy.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (mcp, pip, dip, tip) per finger, thumb included.
FINGER_CHAINS = (
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)

FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

COORD_DIMS = LANDMARK_COUNT * 3
# per finger: curl + reach, then palm normal (3), knuckle spread (4)
DERIVED_DIMS = len(FINGER_CHAINS) * 2 + 3 + 4
#: Offset of the frame block. Appending it keeps every earlier offset valid,
#: so the gesture rules in gestures.py keep indexing the same curl and reach
#: columns without knowing this block exists.
FRAME_START = COORD_DIMS + DERIVED_DIMS

#: wrist x, wrist y, then (cos, sin) for the middle-knuckle and index-knuckle
#: directions. Six numbers that carry placement and in-plane rotation.
FRAME_DIMS = 6

FEATURE_DIMS = FRAME_START + FRAME_DIMS

_EPS = 1e-6


def normalise(landmarks: np.ndarray, handedness: Handedness = Handedness.UNKNOWN) -> np.ndarray:
    """Return (21, 3) position, rotation, scale and handedness invariant landmarks.

    This is the canonical half of the encoding. It answers "what shape is this
    hand" and nothing about where it is, which is why `frame_features` exists
    alongside it rather than instead of it.
    """

    pts = np.asarray(landmarks, dtype=np.float32).copy()
    if pts.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"expected {(LANDMARK_COUNT, 3)} landmarks, got {pts.shape}")

    # A left hand is mirrored into right-hand image space. Mirroring alone is not
    # enough, because it also flips which side of the palm each knuckle sits on,
    # so the frame below is anchored on the palm rather than on a single point.
    if handedness is Handedness.LEFT:
        pts[:, 0] = 1.0 - pts[:, 0]

    pts -= pts[WRIST]

    # Anchor the frame on the palm: "up" runs wrist to middle knuckle, and the
    # palm normal decides which perpendicular counts as "right", which is what
    # makes a left hand and a right hand encode to the same vector.
    up = pts[MIDDLE_MCP][:2] - pts[WRIST][:2]
    up_len = float(np.linalg.norm(up))
    if up_len < _EPS:
        up = np.array([0.0, -1.0], dtype=np.float32)
        up_len = 1.0
    up = up / up_len
    perp = np.array([-up[1], up[0]], dtype=np.float32)

    if _palm_normal(pts)[2] < 0:
        perp = -perp

    rotation = np.array(
        [[perp[0], perp[1], 0.0], [up[0], up[1], 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pts = pts @ rotation.T

    # Scale by the palm length, which is stable across distance.
    scale = up_len + _EPS
    return pts / scale


def frame_features(
    landmarks: np.ndarray, handedness: Handedness = Handedness.UNKNOWN
) -> np.ndarray:
    """Return the placement and orientation the canonical form discards.

    Wrist position in normalised image coordinates, followed by the unit
    direction from the wrist to the middle and index knuckles. The directions
    are stored as (cos, sin) rather than as an angle so the encoding has no
    wraparound discontinuity at plus or minus pi for the model to trip over.
    """

    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"expected {(LANDMARK_COUNT, 3)} landmarks, got {pts.shape}")

    # Fold a left hand into right-hand image space first, exactly as `normalise`
    # does. Doing it after computing the directions would leave them mirrored,
    # so the two hands would disagree on which way the fingers point and stop
    # encoding identically.
    pts = pts.copy()
    if handedness is Handedness.LEFT:
        pts[:, 0] = 1.0 - pts[:, 0]

    x = float(pts[WRIST][0])
    y = float(pts[WRIST][1])

    middle = _unit2(pts[MIDDLE_MCP][:2] - pts[WRIST][:2], default=(0.0, -1.0))
    index = _unit2(pts[INDEX_MCP][:2] - pts[WRIST][:2], default=(0.0, -1.0))

    return np.asarray([x, y, middle[0], middle[1], index[0], index[1]], dtype=np.float32)


def _unit2(v: np.ndarray, default: tuple[float, float]) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return default
    return float(v[0] / n), float(v[1] / n)


def _turning(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Bend at joint q, in radians. A straight chain gives 0, a folded one gives pi."""

    v1 = p - q
    v2 = r - q
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.pi - np.arccos(np.clip(cos, -1.0, 1.0)))


def _curl(pts: np.ndarray, chain: tuple[int, ...]) -> tuple[float, float]:
    """Curl and reach for one finger.

    curl averages the bend at each joint and maps straight to 0, fully folded to
    1. reach is the tip distance over the palm scale, which separates "curled
    toward the palm" from "pointing away from the camera".
    """

    turns = [
        _turning(pts[chain[i]], pts[chain[i + 1]], pts[chain[i + 2]])
        for i in range(len(chain) - 2)
    ]
    curl = float(np.clip(float(np.mean(turns)) / np.pi, 0.0, 1.0)) if turns else 0.0
    reach = float(np.linalg.norm(pts[chain[-1]])) / max(
        float(np.linalg.norm(pts[MIDDLE_MCP][:2])), _EPS
    )
    return curl, reach


def _palm_normal(pts: np.ndarray) -> np.ndarray:
    """Unit vector of the palm plane, built from the wrist and two knuckles."""

    v1 = pts[INDEX_MCP] - pts[WRIST]
    v2 = pts[PINKY_MCP] - pts[WRIST]
    n = np.cross(v1, v2)
    norm = float(np.linalg.norm(n))
    if norm < _EPS:
        return np.zeros(3, dtype=np.float32)
    return (n / norm).astype(np.float32)


def encode_hand(hand: Hand) -> np.ndarray:
    """One frame of landmarks becomes one fixed-length feature vector."""

    pts = normalise(hand.landmarks, hand.handedness)

    curls: list[float] = []
    reaches: list[float] = []
    for chain in FINGER_CHAINS:
        curl, reach = _curl(pts, chain)
        curls.append(curl)
        reaches.append(reach)

    normal = _palm_normal(pts)

    # Adjacent fingertip gaps tell apart a flat spread hand from a relaxed one.
    spread = [
        float(np.linalg.norm(pts[FINGER_TIPS[i]][:2] - pts[FINGER_TIPS[i + 1]][:2]))
        for i in range(4)
    ]

    features = np.concatenate(
        [
            pts.reshape(-1),
            np.asarray(curls, dtype=np.float32),
            np.asarray(reaches, dtype=np.float32),
            normal,
            np.asarray(spread, dtype=np.float32),
            frame_features(hand.landmarks, hand.handedness),
        ]
    ).astype(np.float32)

    if features.shape != (FEATURE_DIMS,):  # pragma: no cover - guards refactors
        raise RuntimeError(f"feature size drifted: {features.shape} != {(FEATURE_DIMS,)}")
    return features


def encode_frame(hands: list[Hand]) -> np.ndarray:
    """Encode a frame using the most prominent hand, all zeros when there is none.

    The model is trained with missing frames left as zeros, so this keeps the
    distribution identical between training data and live input.
    """

    if not hands:
        return np.zeros(FEATURE_DIMS, dtype=np.float32)
    return encode_hand(hands[0])


def active_span(features: list[np.ndarray]) -> int:
    """How many leading frames to drop before the sign started.

    The window is right aligned, so idle frames before the hand appears end up as
    zeros on the left. For a brief sign that is most of the window: the corpus runs
    a median of 59 frames with a 10th percentile of 34, and against a 48 frame
    window a quick 15 frame sign would be two thirds zeros, diluting exactly the
    frames that carry the movement.

    Training clips are trimmed to the sign, so trimming here is what makes live
    input look like what the model was trained on rather than like a padded clip.

    Only leading frames are dropped. Trailing zeros are the hand settling or
    leaving, which is part of how a sign ends and which the decoder relies on.
    """

    drop = 0
    for frame in features:
        if np.any(frame):
            break
        drop += 1
    return drop


def encode_sequence(features: list[np.ndarray], length: int) -> np.ndarray:
    """Right-align a variable length sequence into a fixed (length, width) window.

    Right alignment matters: a sign ends when the hand settles, so the most recent
    frames are the meaningful ones and the oldest frames fall off the left.

    Idle frames ahead of the sign are dropped first, so a brief sign fills the
    window instead of sharing it with the period before the hand appeared.

    The width is taken from the data rather than from `FEATURE_DIMS`. The body
    aware tracks append 34 pose columns to the 86 hand columns, and padding a
    120 wide track into an 86 wide window raises a broadcast error that looks
    like corrupt data rather than a wrong constant.
    """

    if not features:
        return np.zeros((length, FEATURE_DIMS), dtype=np.float32)
    idle = active_span(features)
    if idle:
        features = features[idle:]
    tail = np.asarray(features[-length:], dtype=np.float32)
    out = np.zeros((length, tail.shape[1]), dtype=np.float32)
    out[length - len(tail) :] = tail
    return out