"""The 100 gesture book shared by the demo rig and the gesture recogniser.

Both sides read this table, which is what keeps them honest. The rig animates a
pose and the recogniser matches the features it produces, so a gesture cannot be
declared recognisable unless it is actually produced.

Why templates rather than rules. The original recogniser decided from two
numbers, finger curl and thumb reach, which is enough to separate a handful of
shapes and no more. ASL words mostly share hand shapes and differ by movement
and location, so hand-shape thresholds alone collide badly at any real
vocabulary. Matching against a stored feature template scales to a hundred
entries without a hundred hand-written branches, and each entry can be
overwritten later with a clip the user recorded themselves.

What these poses are, honestly. The finger configurations are generated to be
mutually distinguishable, not reconstructed from the ASL sign alphabet. They
carry the right word name and a plausible hand shape, but a native signer would
not read most of them as that word. This is a testing and demo vocabulary, sized
to exercise the pipeline end to end, not a reference for correct signing. Real
fidelity comes from replacing a template with a clip recorded by the user, which
is what the in-app recorder and finetune path are for.

Why the shapes are generated rather than hand-written. Hand-written poses at
this count collide: an earlier table of 104 hand-picked curls produced a closest
pair only 0.03 apart on an 86 dimensional vector, which is the same vector to
the recogniser, so `sing` and `talk` were the same gesture. The poses here come
from a low-discrepancy sequence over the curl space, quantised to levels a hand
can actually hold, which spreads them apart by construction. scripts/
check_gesture_separation.py reports the closest pair, and the test suite fails
if two gestures ever become indistinguishable again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..vision.simulated import Pose

#: Word order here is the demo cycle order.
WORDS: tuple[str, ...] = (
    "hello", "thank_you", "please", "sorry", "yes", "no", "help", "love",
    "friend", "family", "mother", "father", "sister", "brother", "child",
    "baby", "name", "where", "who", "what", "when", "why", "how", "good",
    "bad", "big", "small", "hot", "cold", "water", "food", "eat", "drink",
    "house", "home", "work", "school", "teacher", "student", "doctor", "car",
    "bus", "money", "buy", "store", "time", "today", "tomorrow", "yesterday",
    "morning", "night", "day", "week", "month", "year", "come", "go", "stop",
    "wait", "look", "see", "hear", "listen", "speak", "talk", "know",
    "think", "want", "need", "like", "learn", "read", "write", "play", "sing",
    "dance", "walk", "run", "sit", "stand", "open", "close", "give", "take",
    "meet", "call", "live", "here", "there", "and", "or", "but", "if",
    "finished", "start", "remember", "forget", "understand", "open_palm",
    "closed_fist", "pointing_up", "thumbs_up", "victory", "i_love_you",
)

#: The last six are the original hand-shaped rules. They are kept verbatim so
#: existing behaviour, and anyone who learned them, keeps working.
LEGACY_WORDS = (
    "open_palm", "closed_fist", "pointing_up", "thumbs_up", "victory", "i_love_you",
)

#: How many gestures the book is expected to hold. Asserted by the tests so a
#: truncated table cannot quietly shrink the demo vocabulary.
EXPECTED_GESTURES = 104

#: Finger curl levels a hand can hold. Five levels rather than three: with three,
#: 98 generated poses collided into 93 distinct finger configurations, and the
#: pairs that shared one differed only in thumb position, which moves the
#: feature vector very little. Five levels gives 3125 combinations for 98
#: gestures, which leaves room for every pose to be genuinely its own shape.
_LEVELS = (0.02, 0.26, 0.5, 0.72, 0.95)


def _halton(index: int, base: int) -> float:
    """One axis of a Halton sequence, for spreading indices evenly over a unit
    interval instead of leaving them bunched the way random picks would."""

    f, r, i = 1.0, 0.0, index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


#: Bases are the first five primes, one per finger.
_BASES = (2, 3, 5, 7, 11)


def _pose_params(index: int) -> tuple[tuple[float, ...], float, float, float, float, float, float]:
    """Derive one pose's parameters from its index.

    Index 0 is skipped so the first generated pose is not the all-zeros corner
    of the sequence, which would be a dead flat hand.
    """

    n = index + 1
    # Five fingers, three levels each: the thumb gets its own axis so thumb
    # position varies independently of the four fingers.
    curls = tuple(_LEVELS[int(_halton(n, b) * len(_LEVELS)) % len(_LEVELS)] for b in _BASES)

    # Thumb extension and palm opposition vary on slower cycles, and placement
    # walks the signing space so position carries some of the load too.
    thumb_extend = 0.15 + 0.8 * _halton(n, 13)
    thumb_opposition = 0.1 + 0.8 * _halton(n, 17)
    spread = 0.15 + 0.7 * _halton(n, 19)
    x = 0.25 + 0.5 * _halton(n, 23)
    y = 0.28 + 0.45 * _halton(n, 29)
    return curls, thumb_extend, thumb_opposition, spread, 0.0, x, y


def _legacy_params(
    label: str,
) -> tuple[tuple[float, ...], float, float, float, float, float, float]:
    """The six original poses, unchanged."""

    table: dict[str, tuple[tuple[float, ...], float, float, float, float, float]] = {
        "open_palm": ((0.05, 0.0, 0.0, 0.0, 0.0), 0.6, 0.5, 0.85, 0.50, 0.68),
        "closed_fist": ((0.90, 0.85, 0.85, 0.85, 0.80), 0.1, 0.8, 0.30, 0.50, 0.35),
        "pointing_up": ((0.90, 0.0, 0.85, 0.85, 0.85), 0.1, 0.9, 0.85, 0.50, 0.40),
        "thumbs_up": ((0.95, 0.85, 0.85, 0.85, 0.85), 1.0, 0.0, 0.80, 0.50, 0.32),
        "victory": ((0.90, 0.0, 0.0, 0.85, 0.85), 0.1, 0.9, 0.85, 0.50, 0.38),
        "i_love_you": ((0.90, 0.0, 0.85, 0.85, 0.0), 0.1, 0.9, 0.85, 0.50, 0.35),
    }
    curls, ext, opp, spread, x, y = table[label]
    return curls, ext, opp, spread, 0.0, x, y


def build_gestures() -> dict[str, Pose]:
    """Turn the word list into Pose objects.

    Imported lazily to avoid a cycle: simulated.py imports this module to know
    what to draw, so this cannot import simulated at module scope.
    """

    from .simulated import Pose

    out: dict[str, Pose] = {}
    for index, label in enumerate(WORDS):
        if label in LEGACY_WORDS:
            curls, ext, opp, spread, tilt, x, y = _legacy_params(label)
        else:
            curls, ext, opp, spread, tilt, x, y = _pose_params(index)
        out[label] = Pose(
            curls=curls,
            thumb_extend=ext,
            thumb_opposition=opp,
            spread=spread,
            wrist_tilt=tilt,
            x=x,
            y=y,
        )
    return out


def labels() -> list[str]:
    """Gesture names in book order."""

    return list(WORDS)


def label_count() -> int:
    return len(WORDS)