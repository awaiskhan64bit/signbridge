"""The curated sign vocabulary.

Selecting the top N words by clip count looks sensible and is not. It ranks by
how many training clips a word happens to have, which correlates with how
common the word is in WLASL's collection rather than how often a person signs
it. The effect was concrete: the previous 500-word vocabulary ended
alphabetically at "yesterday" and did not contain "hello", because "hello" has
only 13 clips and fell below the cut. Testing the app on a clip of someone
signing hello was then guaranteed to fail, and it did.

So the vocabulary is chosen by hand: the words people actually use, grouped so
the intent is obvious and the list can be argued with. Every entry is checked
against WLASL at load time, because WLASL uses spaces rather than underscores
and phrases like "thank you" are single entries there.

Selection is still constrained by data. A word with three clips cannot be
learned, so rare words are marked and the trainer drops them rather than
pretending. Words needed for a first conversation are listed first and are kept
even when rare.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Words a beginner needs before anything else. Kept unconditionally.
CORE: tuple[str, ...] = (
    # greetings and courtesy
    "hello", "goodbye", "thank you", "please", "sorry", "yes", "no", "help",
    # people and questions
    "name", "my", "who", "what", "where", "when", "why", "how",
    # the room
    "home", "house", "work", "school",
    # essentials
    "water", "food", "eat", "drink",
    # time
    "today", "tomorrow", "yesterday", "now", "time",
    # people
    "friend", "family", "mother", "father", "sister", "brother",
)

#: Everyday words by topic. Roughly frequency order within each group.
FAMILIAR: tuple[str, ...] = (
    # describing things
    "good", "bad", "big", "small", "hot", "cold", "new", "old", "happy",
    "sad", "very", "more", "nice", "fine", "understand",
    # body
    "head", "hand", "eye", "ear", "nose", "mouth", "arm",
    # family and people
    "child", "baby", "wife", "husband", "son", "daughter", "aunt", "uncle",
    "teacher", "doctor",
    # food and drink
    "hungry", "thirsty", "breakfast", "lunch", "dinner", "milk", "tea",
    "coffee", "apple", "bread",
    # places
    "store", "hospital", "city", "country", "street", "door", "car", "bus",
    "train",
    # time
    "morning", "night", "day", "week", "month", "year", "late", "early",
    # things to do
    "come", "go", "stop", "wait", "see", "hear", "listen", "speak", "talk",
    "know", "think", "want", "need", "like", "learn", "read", "write", "play",
    "sing", "dance", "walk", "run", "sit", "stand", "open", "close", "give",
    "take", "meet", "call", "live", "remember", "forget", "start",
    # numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten",
)

#: Everything, de-duplicated, core first.
def all_words() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for word in (*CORE, *FAMILIAR):
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def resolve(available: Iterable[str], limit: int = 500) -> list[str]:
    """Return the curated words that WLASL actually has, in priority order.

    `available` is the set of glosses present in the index. Unknown words are
    skipped rather than raising, because a wording change in WLASL should
    degrade the vocabulary, not stop the app.
    """

    have = set(available)
    return [w for w in all_words() if w in have][:limit]


def coverage_report(available: Iterable[str]) -> dict[str, object]:
    """What the curated list resolved to, for logging and for the trainer."""

    have = set(available)
    wanted = all_words()
    kept = [w for w in wanted if w in have]
    missing = [w for w in wanted if w not in have]
    return {
        "wanted": len(wanted),
        "kept": len(kept),
        "missing": missing,
        "core_kept": [w for w in CORE if w in have],
        "core_missing": [w for w in CORE if w not in have],
    }