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

#: The six original poses, unchanged from the version this replaced.
#: Each is (curls x5, thumb_extend, thumb_opposition, spread, x, y).
_SPECS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("open_palm", (0.05, 0.0, 0.0, 0.0, 0.0, 0.60, 0.50, 0.85, 0.50, 0.68)),
    ("closed_fist", (0.90, 0.85, 0.85, 0.85, 0.80, 0.10, 0.80, 0.30, 0.50, 0.35)),
    ("pointing_up", (0.90, 0.0, 0.85, 0.85, 0.85, 0.10, 0.90, 0.85, 0.50, 0.40)),
    ("thumbs_up", (0.95, 0.85, 0.85, 0.85, 0.85, 1.00, 0.00, 0.80, 0.50, 0.32)),
    ("victory", (0.90, 0.0, 0.0, 0.85, 0.85, 0.10, 1.00, 0.85, 0.50, 0.38)),
    ("i_love_you", (0.90, 0.0, 0.85, 0.85, 0.0, 0.10, 0.90, 0.85, 0.50, 0.35)),
)

EXPECTED_GESTURES = 6


def build_gestures() -> dict[str, Pose]:
    """Turn the compact specs into Pose objects.

    Imported lazily to avoid a cycle: simulated.py imports this module to know
    what to draw, so this cannot import simulated at module scope.
    """

    from .simulated import Pose

    return {
        label: Pose(
            curls=(s[0], s[1], s[2], s[3], s[4]),
            thumb_extend=s[5],
            thumb_opposition=s[6],
            spread=s[7],
            wrist_tilt=0.0,
            x=s[8],
            y=s[9],
        )
        for label, s in _SPECS
    }


def labels() -> list[str]:
    return [label for label, _ in _SPECS]


def label_count() -> int:
    return len(_SPECS)