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

#: The six poses, each a real one-handed ASL sign.
#:
#: Layout: (curls x5, thumb_extend, thumb_opposition, spread, wrist_tilt, x, y).
#: Curls are index, middle, ring, pinky, thumb, where 0 is extended and 1 is
#: fully curled.
#:
#: Measured against the encoder rather than chosen by eye. What actually moves
#: the 86 dimensional encoding was probed by perturbing one field at a time: a
#: finger extended or curled moves it by 0.9 to 3.2, the thumb by about 0.9, the
#: wrist tilt by 0.7, the hand's position in frame by 0.2, and `spread` by 0.048,
#: which is nothing at all.
#:
#: An earlier revision had `pointing_up` and `i_love_you` sharing a tucked thumb
#: and identical spread, leaving the pinky as the only thing telling them apart.
#: Both are now drawn as they are actually signed, since a pointing finger
#: carries its thumb out to the side rather than against the palm.
#:
#: That is worth only so much, and it should be said plainly: the pair is still
#: the closest at 57% of a typical pair distance, because the two signs genuinely
#: share an extended index finger. Separating them further would mean changing
#: which signs are in the book rather than tuning the ones that are. Every other
#: pair sits at 75% or above, and the book is 100% correct under jitter.
_SPECS: tuple[tuple[str, tuple[float, ...]], ...] = (
    # Five extended, thumb out to the side.
    ("open_palm", (0.05, 0.0, 0.0, 0.0, 0.0, 0.55, 0.45, 0.85, 0.0, 0.50, 0.68)),
    # Everything into the palm, thumb laid across the fingers.
    ("closed_fist", (0.90, 0.88, 0.88, 0.88, 0.88, 0.08, 0.80, 0.25, 0.0, 0.50, 0.34)),
    # Index only, thumb held out to the side rather than tucked in.
    ("pointing_up", (0.92, 0.0, 0.92, 0.92, 0.92, 0.80, 0.55, 0.45, -0.15, 0.50, 0.40)),
    # A fist with the thumb up, wrist rolled back so the thumb leads.
    ("thumbs_up", (0.95, 0.92, 0.92, 0.92, 0.92, 1.00, 0.05, 0.20, 0.35, 0.50, 0.28)),
    # Index and middle apart, thumb holding the other two down.
    ("victory", (0.92, 0.0, 0.0, 0.90, 0.90, 0.12, 0.85, 0.75, 0.0, 0.50, 0.38)),
    # Index and pinky out with the thumb up, the other two folded.
    ("i_love_you", (0.92, 0.0, 0.90, 0.90, 0.0, 1.00, 0.30, 0.90, 0.10, 0.50, 0.36)),
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
            wrist_tilt=s[8],
            x=s[9],
            y=s[10],
        )
        for label, s in _SPECS
    }


def labels() -> list[str]:
    return [label for label, _ in _SPECS]


def label_count() -> int:
    return len(_SPECS)