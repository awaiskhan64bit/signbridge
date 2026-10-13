"""Synthetic signer.

The app has to be demonstrable on a machine with no camera plugged in, so this
module builds anatomically plausible 21-point hand rigs from a handful of pose
parameters and animates them through a small phrasebook. It plugs into the same
capture interface as a real camera, which means the whole pipeline (features,
model, phrase assembly, speech) runs for real in demo mode.

    SimulatedSigner -> frames with attached hands -> pipeline -> TTS
"""

from __future__ import annotations

import math

import numpy as np

from ..core.types import LANDMARK_COUNT, Hand, Handedness
from . import gesturebook
from .camera import VideoSource

# Draw colours in BGR, matching the live overlay so demo mode looks identical.
_BONE = (96, 214, 255)
_JOINT = (255, 255, 255)
_TIP = (255, 240, 168)

# Canonical palm frame, in units of "wrist to middle knuckle". +x runs toward the
# pinky, +y runs from the knuckles down to the wrist, so fingers extend along -y
# and end up pointing up the image the way a real signer's hand does.
_PALM = {
    0: (0.00, 0.00),
    1: (-0.34, -0.16),  # thumb CMC
    5: (-0.30, -1.00),  # index MCP
    9: (0.00, -1.06),  # middle MCP
    13: (0.28, -1.02),  # ring MCP
    17: (0.52, -0.90),  # pinky MCP
}

# (proximal, middle, distal) bone lengths per finger.
_BONES = {
    5: (0.44, 0.27, 0.21),
    9: (0.48, 0.30, 0.22),
    13: (0.44, 0.28, 0.21),
    17: (0.34, 0.21, 0.18),
}
_FINGER_MCPS = (5, 9, 13, 17)
# Resting splay per finger, radians.
_SPLAY = (-0.16, -0.02, 0.14, 0.32)
# Maximum bend per joint, radians. Tuned so curl=1 reads as a closed fist and
# curl=0 reads as a fully extended finger, which is what the gesture thresholds
# in ml/gestures.py were calibrated against.
_MAX_BEND = (1.80, 2.00, 1.40)


def _rot(a: float) -> tuple[float, float, float, float]:
    c, s = math.cos(a), math.sin(a)
    return c, s, -s, c


class Pose:
    """One hand configuration. Every field is roughly in 0..1 unless noted."""

    __slots__ = (
        "curls",
        "scale",
        "spread",
        "thumb_extend",
        "thumb_opposition",
        "wrist_tilt",
        "x",
        "y",
        "z",
    )

    def __init__(
        self,
        curls: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
        thumb_opposition: float = 0.0,
        thumb_extend: float = 0.5,
        spread: float = 0.25,
        wrist_tilt: float = 0.0,
        scale: float = 0.24,
        x: float = 0.5,
        y: float = 0.68,
        z: float = 0.0,
    ) -> None:
        self.curls = curls
        self.thumb_opposition = thumb_opposition
        self.thumb_extend = thumb_extend
        self.spread = spread
        self.wrist_tilt = wrist_tilt
        self.scale = scale
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def lerp(a: Pose, b: Pose, t: float) -> Pose:
        t = max(0.0, min(1.0, t))
        e = t * t * (3.0 - 2.0 * t)  # smoothstep, so motion eases in and out
        mix = lambda p, q: p + (q - p) * e  # noqa: E731
        return Pose(
            curls=tuple(
                mix(a.curls[i], b.curls[i]) for i in range(len(a.curls))
            ),
            thumb_opposition=mix(a.thumb_opposition, b.thumb_opposition),
            thumb_extend=mix(a.thumb_extend, b.thumb_extend),
            spread=mix(a.spread, b.spread),
            wrist_tilt=mix(a.wrist_tilt, b.wrist_tilt),
            scale=mix(a.scale, b.scale),
            x=mix(a.x, b.x),
            y=mix(a.y, b.y),
            z=mix(a.z, b.z),
        )


def build_landmarks(pose: Pose) -> np.ndarray:
    """Turn a pose into 21 landmarks in normalised image coordinates."""

    pts = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)

    # Curl each finger as a 3-link chain bending in the palm plane, with a touch
    # of out of plane motion on the distal joint so it does not look flat.
    for f, mcp in enumerate(_FINGER_MCPS):
        l1, l2, l3 = _BONES[mcp]
        curl = float(np.clip(pose.curls[f + 1], 0.0, 1.0))
        base = _PALM[mcp]
        # The bend accumulates: each joint rotates the direction the next bone
        # leaves at, which is what makes a curl read as a curl.
        angle = -math.pi / 2 + _SPLAY[f] * (0.4 + pose.spread)
        pts[mcp] = (*base, 0.0)  # the knuckle itself is an anchor, not a bone end

        for seg, length in enumerate((l1, l2, l3)):
            angle += curl * _MAX_BEND[seg] * (0.35 if seg == 0 else 1.0)
            nx = base[0] + math.cos(angle) * length
            ny = base[1] + math.sin(angle) * length
            # The MCP landmark is the anchor, so each new joint is one past it:
            # pip = mcp+1, dip = mcp+2, tip = mcp+3.
            idx = mcp + seg + 1
            depth = curl * 0.22 * seg  # fingers fold toward the palm as they curl
            pts[idx] = (nx, ny, -depth)
            base = (nx, ny)

    # Thumb: a short chain swung out to the side and rotated across the palm.
    tcurl = float(np.clip(pose.curls[0], 0.0, 1.0))
    opp = float(np.clip(pose.thumb_opposition, 0.0, 1.0))
    ext = float(np.clip(pose.thumb_extend, 0.0, 1.0))
    base = _PALM[1]
    angle = -math.pi / 2 - 0.85 * (1.0 - opp * 0.8)
    # The thumb folds less the more it is extended, so a thumbs up is a straight
    # thumb regardless of how curled the rest of the pose claims to be.
    fold = tcurl * (1.0 - ext)
    for seg, length in enumerate((0.30, 0.28, 0.22)):
        angle += fold * _MAX_BEND[seg] * 0.7 * (1.0 - opp * 0.5)
        nx = base[0] + math.cos(angle) * length
        ny = base[1] + math.sin(angle) * length
        pts[2 + seg] = (nx, ny, -opp * 0.25 * seg)
        base = (nx, ny)
    pts[1] = (*_PALM[1], 0.0)
    pts[0] = (*_PALM[0], 0.0)

    # Place the palm: scale, in-plane roll, then translate into the frame.
    # z has to be scaled too, otherwise the depth cue ends up several times
    # larger than the hand itself and dominates the feature vector.
    pts *= pose.scale
    c, s, ns, nc = _rot(pose.wrist_tilt)
    rot = np.array([[c, s, 0.0], [ns, nc, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pts = pts @ rot.T

    pts[:, 0] += pose.x
    pts[:, 1] += pose.y
    pts[:, 2] += pose.z
    # Only the image plane needs clamping. Clipping z would flatten the depth
    # cue the feature encoder uses to tell a curled finger from a pointing one.
    pts[:, :2] = np.clip(pts[:, :2], 0.002, 0.998)
    return pts.astype(np.float32)


# --------------------------------------------------------------------- poses

# The gesture book lives in vision/gesturebook.py so the rig and the recogniser
# cannot drift apart: both read the same table, so a pose cannot be drawn unless
# the recogniser also has a template for it.
GESTURES: dict[str, Pose] = gesturebook.build_gestures()

#: A loose, non-signing hand shape, used between gestures so the decoder sees a
#: real transition instead of one pose snapping straight into the next.
_REST = Pose(curls=(0.5, 0.6, 0.6, 0.6, 0.55), thumb_opposition=0.5, thumb_extend=0.2, spread=0.2)


def _phrase(label: str) -> list[Pose]:
    """Approach then hold, which is what a real sign looks like in time.

    The list must end on the target pose: the signer holds whatever is last here,
    and the gap between phrases is what returns the hand to rest.
    """

    target = GESTURES[label]
    approach = Pose.lerp(_REST, target, 0.45)
    return [approach, target, target, target]


#: Every gesture in the book, cycled endlessly. The demo used to cycle six; the
#: full book is what exercises a hundred-word recogniser end to end.
PHRASEBOOK: list[tuple[str, list[Pose]]] = [
    (label, _phrase(label)) for label in gesturebook.labels() if label in GESTURES
]


class SimulatedSigner(VideoSource):
    """A capture backend that draws its own frames and hands."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        phrasebook: list[tuple[str, list[Pose]]] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.phrasebook = phrasebook or PHRASEBOOK
        self.hands: list[Hand] = []
        self.current_label: str = ""
        self.frame_index = 0

        self._pose_frames = 8
        self._hold_frames = 44
        self._gap_frames = 8
        per_sign = self._pose_frames + self._hold_frames + self._gap_frames
        self._total = len(self.phrasebook) * per_sign
        self._background = self._make_background()

    # ------------------------------------------------------------- lifecycle

    def open(self) -> None:
        return None

    def release(self) -> None:
        return None

    # ------------------------------------------------------------- rendering

    def _make_background(self) -> np.ndarray:
        """A plain studio backdrop. A faint grid gives the rig's motion something
        to read against, which a flat fill does not."""

        h, w = self.height, self.width
        top = np.array([18, 20, 34], dtype=np.float32)
        bottom = np.array([8, 9, 16], dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        img = top[None, :] * (1.0 - ramp) + bottom[None, :] * ramp
        img = np.repeat(img[:, None, :], w, axis=1)

        step = 64
        img[::step, :, :] = np.minimum(img[::step, :, :] + 7.0, 255.0)
        img[:, ::step, :] = np.minimum(img[:, ::step, :] + 7.0, 255.0)
        return img.astype(np.uint8)

    # ----------------------------------------------------------------- cycle

    def _segment(self, index: int) -> tuple[str, Pose]:
        """Map a frame index to the label and pose that should be shown."""

        step = self._pose_frames + self._hold_frames + self._gap_frames
        which = (index // step) % len(self.phrasebook)
        label, poses = self.phrasebook[which]
        local = index % step
        if local >= self._pose_frames + self._hold_frames:
            return label, _REST  # gap: hand rests between signs

        if local < self._pose_frames:
            # Ease through the sign's key poses.
            t = local / max(1, self._pose_frames - 1)
            scaled = t * (len(poses) - 1)
            i = int(scaled)
            frac = scaled - i
            a = poses[i]
            b = poses[min(i + 1, len(poses) - 1)]
            return label, Pose.lerp(a, b, frac)
        return label, poses[-1]

    def read(self) -> np.ndarray | None:
        label, pose = self._segment(self.frame_index)
        self.frame_index += 1
        self.current_label = label

        landmarks = build_landmarks(pose)
        self.hands = [Hand(landmarks=landmarks, handedness=Handedness.RIGHT, score=0.97)]
        # Only the backdrop is drawn here. The skeleton is left to the shared
        # overlay in vision.overlay, so demo mode renders exactly like a camera,
        # and no text is burned in because the preview can be mirrored.
        return self._background.copy()
