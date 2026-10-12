"""Feature encoding for the sign classifier.

A raw frame of 21 MediaPipe landmarks is a terrible model input: it depends on
where the signer stands, how far away they are, and how the camera is angled.
Everything here exists to strip that variation out so the network sees shape.

Normalisation, in order:

1. translate the wrist to the origin
2. rotate in the image plane so the index knuckle points along a fixed axis
3. scale so the wrist to middle-knuckle distance is 1
4. mirror the left hand so both hands encode identically

On top of the 63 normalised coordinates we add derived quantities that are cheap
for the network to learn but awkward to work out from coordinates: joint curl
per finger, palm orientation, knuckle spread and reach.
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
EXTRA_DIMS = len(FINGER_CHAINS) * 2 + 3 + 4
FEATURE_DIMS = COORD_DIMS + EXTRA_DIMS

_EPS = 1e-6


def normalise(landmarks: np.ndarray, handedness: Handedness = Handedness.UNKNOWN) -> np.ndarray:
    """Return (21, 3) position, rotation, scale and handedness invariant landmarks."""

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


def encode_sequence(features: list[np.ndarray], length: int) -> np.ndarray:
    """Right-align a variable length sequence into a (length, FEATURE_DIMS) window.

    Right alignment matters: a sign ends when the hand settles, so the most recent
    frames are the meaningful ones and the oldest frames fall off the left.
    """

    out = np.zeros((length, FEATURE_DIMS), dtype=np.float32)
    if not features:
        return out
    tail = features[-length:]
    out[length - len(tail) :] = np.asarray(tail, dtype=np.float32)
    return out
