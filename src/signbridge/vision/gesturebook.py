"""The six-gesture book shared by the demo rig and the gesture recogniser.

Both sides read this table, which is what keeps them honest. The rig animates a
pose and the recogniser matches the features it produces, so a gesture cannot be
declared recognisable unless it is actually produced.

Six is the number, and it is a deliberate limit rather than a shortfall. These
templates are synthetic, so they match the demo rig and nothing else: on a
webcam a real hand sits about 2.5 units away from the nearest template, five
times the acceptance tolerance, and widening the tolerance to reach it would make
every gesture match something and report confident nonsense.

A hundred-entry version of this book was built and measured, and it read well on
the rig while being no closer to a real hand. Growing the table inflated the demo
vocabulary without adding any reach into live input, and it crowded out the
vocabulary that matters: the trained word model.

Getting gestures onto a camera is a recording job, not a rule-writing one. The
in-app recorder writes clips through `Collector`, and `load_templates` accepts
them, so a handful of recorded signs replace these synthetic poses for the words
they cover. That is the only route that works on live input, and it starts at
whatever number of words the user actually records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..vision.simulated import Pose

#: Per-gesture presentation, generated from ml.gestures.GESTURES.
#:
#: The finger patterns and thumb state live in one table in ml.gestures, because
#: that is what the recogniser scores against. This module only supplies what a
#: pose needs on top of that: how open the hand sits, how far the thumb reaches
#: across the palm when it is tucked, and where in frame the hand rests.
#:
#: The old six had these written out by hand and drifted from the recogniser's
#: view of them more than once. Deriving both from one table means a gesture
#: cannot exist on one side only.
_EXTENDED = 0.0
_FOLDED = 0.88
#: Thumb curl is held neutral. What separates a leading thumb from a tucked one
#: is reach, which comes from thumb_extend and thumb_opposition, not from curl.
_THUMB_CURL = 0.5
#: label -> (spread, wrist_tilt, x, y)
_PLACEMENT: dict[str, tuple[float, float, float, float]] = {
    "open_palm": (0.85, 0.0, 0.50, 0.68),
    "four": (0.75, 0.0, 0.50, 0.66),
    "closed_fist": (0.30, 0.0, 0.50, 0.34),
    "thumbs_up": (0.80, 0.35, 0.50, 0.32),
    "pointing_up": (0.85, -0.15, 0.50, 0.40),
    "finger_gun": (0.60, -0.10, 0.50, 0.42),
    "victory": (0.85, 0.0, 0.50, 0.38),
    "horns": (0.60, 0.0, 0.50, 0.40),
    "three": (0.80, 0.0, 0.50, 0.44),
    "six": (0.70, 0.0, 0.50, 0.46),
    "pinky_up": (0.80, 0.20, 0.50, 0.30),
    "shaka": (0.85, 0.15, 0.50, 0.32),
    "i_love_you": (0.85, 0.10, 0.50, 0.36),
    "hang_ten": (0.75, 0.05, 0.50, 0.38),
    "rock_on": (0.85, 0.05, 0.50, 0.34),
}


def build_gestures() -> dict[str, Pose]:
    """Turn the gesture table into Pose objects.

    Imported lazily to avoid a cycle: simulated.py imports this module to know
    what to draw, and ml.gestures is imported by this module's callers.
    """

    from ..ml.gestures import GESTURES
    from .simulated import Pose

    out: dict[str, Pose] = {}
    for label, pattern, thumb_out in GESTURES:
        spread, wrist_tilt, x, y = _PLACEMENT[label]
        # Pose.curls is (thumb, index, middle, ring, pinky): simulated.py reads
        # finger f from curls[f + 1] and the thumb from curls[0]. Getting that
        # order wrong leaves the pinky pinned at one value in every gesture, which
        # quietly merges any two signs that differ only in the pinky.
        curls = [_THUMB_CURL] + [_FOLDED if flag else _EXTENDED for flag in pattern]
        out[label] = Pose(
            curls=tuple(curls),
            # A tucked thumb folds across the palm, a leading thumb extends away
            # from it. These two values are what separate the gesture pairs that
            # share a finger pattern, such as a fist and a thumbs up.
            thumb_extend=0.95 if thumb_out else 0.10,
            thumb_opposition=0.10 if thumb_out else 0.85,
            spread=spread,
            wrist_tilt=wrist_tilt,
            x=x,
            y=y,
        )
    return out


def labels() -> list[str]:
    from ..ml.gestures import GESTURES

    return [label for label, _, _ in GESTURES]


def label_count() -> int:
    from ..ml.gestures import EXPECTED_GESTURES

    return EXPECTED_GESTURES


#: Re-exported so callers can assert on the size without importing both modules.
#: Assigned rather than defined, so there is exactly one number.
def __getattr__(name: str):
    if name == "EXPECTED_GESTURES":
        from ..ml.gestures import EXPECTED_GESTURES as n

        return n
    raise AttributeError(name)