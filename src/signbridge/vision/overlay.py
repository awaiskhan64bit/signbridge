"""Draws the landmark overlay onto captured frames.

Kept separate from the pipeline so the drawing style can be changed without
touching any logic, and so the demo mode can reuse the exact same rendering as a
real camera.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Hand

# BGR. Amber for the skeleton, warm white for the joints, cyan for the tips.
_BONE = (255, 214, 96)
_JOINT = (236, 236, 236)
_TIP = (96, 214, 255)
_DIM = (150, 140, 110)

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
FINGER_TIPS = (4, 8, 12, 16, 20)


def draw_hands(frame: np.ndarray, hands: list[Hand], *, mirror: bool = False) -> np.ndarray:
    """Return a copy of `frame` with every detected hand drawn on it."""

    if not hands:
        return frame
    out = frame
    h, w = out.shape[:2]

    for hand in hands:
        pts = hand.landmarks[:, :2].copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        # In a mirrored preview the x axis is flipped, so the skeleton must be too.
        if mirror:
            pts[:, 0] = w - pts[:, 0]
        p = np.round(pts).astype(np.int32)

        for a, b in CONNECTIONS:
            cv2.line(out, tuple(p[a]), tuple(p[b]), _BONE, 2, cv2.LINE_AA)
        for i, point in enumerate(p):
            radius = 5 if i in FINGER_TIPS else 3
            colour = _TIP if i in FINGER_TIPS else _JOINT
            cv2.circle(out, tuple(point), radius, colour, -1, cv2.LINE_AA)

        fingers = hand.fingers_up
        wrist = tuple(p[0])
        # One digit per finger, thumb first. Small and low contrast: it is a
        # debugging aid, not part of the presentation.
        label = "".join("1" if f else "0" for f in fingers)
        cv2.putText(
            out,
            f"{hand.handedness.value[:1].upper()} {label}",
            (wrist[0] - 18, wrist[1] + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            _DIM,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_hud(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """Small status text in the top left, for the stats the UI also shows."""

    for i, (text, colour) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, 30 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            1,
            cv2.LINE_AA,
        )
    return frame
