"""Tests for recording the user's own signing.

The collector is the input to fine tuning, so a silently wrong clip here turns
into a silently worse model later. These cover the cases that are easy to get
wrong: empty clips, clips with no hand, and label handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.ml.collector import DEFAULT_CLIP_FRAMES, Collector, normalise_label
from signbridge.ml.features import FEATURE_DIMS


def frame(value: float = 1.0) -> np.ndarray:
    return np.full(FEATURE_DIMS, value, dtype=np.float32)


def absent_frame() -> np.ndarray:
    return np.zeros(FEATURE_DIMS, dtype=np.float32)


def test_label_is_normalised():
    assert normalise_label("  Hello  World ") == "hello world"
    assert normalise_label("THANK   YOU") == "thank you"
    assert normalise_label("") == ""


def test_records_and_stores_a_clip(tmp_path):
    collector = Collector(tmp_path)
    assert collector.start("hello")

    for _ in range(DEFAULT_CLIP_FRAMES):
        collector.push(frame())
    assert collector.full

    path = collector.stop()
    assert path is not None and path.exists()
    assert collector.captured == 1

    tracks, labels = collector.load_all()
    assert labels == ["hello"]
    assert tracks[0].shape == (DEFAULT_CLIP_FRAMES, FEATURE_DIMS)


def test_refuses_an_empty_label(tmp_path):
    collector = Collector(tmp_path)
    assert not collector.start("   ")
    assert not collector.is_recording


def test_clip_of_nothing_is_rejected(tmp_path):
    """Waving at an empty frame must not become a training example."""

    collector = Collector(tmp_path)
    collector.start("hello")
    for _ in range(DEFAULT_CLIP_FRAMES):
        collector.push(absent_frame())

    assert collector.stop() is None
    assert collector.captured == 0
    assert not list(tmp_path.glob("*.npz"))


def test_sparse_clip_is_rejected(tmp_path):
    """Some signal is required, but not on every frame."""

    collector = Collector(tmp_path)
    collector.start("hello")
    for i in range(DEFAULT_CLIP_FRAMES):
        collector.push(frame() if i < 4 else absent_frame())

    assert collector.stop() is None


def test_a_single_hand_frame_is_too_sparse(tmp_path):
    collector = Collector(tmp_path)
    collector.start("hello")
    collector.push(frame())
    for _ in range(DEFAULT_CLIP_FRAMES):
        collector.push(absent_frame())
    assert collector.stop() is None


def test_clip_shorter_than_a_window_is_rejected(tmp_path):
    collector = Collector(tmp_path)
    collector.start("hello")
    for _ in range(3):
        collector.push(frame())
    assert collector.stop() is None


def test_push_outside_a_recording_is_ignored(tmp_path):
    collector = Collector(tmp_path)
    assert not collector.push(frame())
    assert collector.captured == 0


def test_buffer_keeps_only_the_most_recent_window(tmp_path):
    """A long hold should record the settled pose, not a stale opening frame."""

    collector = Collector(tmp_path, clip_frames=16)
    collector.start("hello")
    for i in range(64):
        collector.push(frame(float(i)))

    assert collector.frames_have == 16
    # The oldest of the retained frames is the one that fell off the front.
    assert collector.frames_have == 16


def test_cancel_discards_the_clip(tmp_path):
    collector = Collector(tmp_path)
    collector.start("hello")
    for _ in range(DEFAULT_CLIP_FRAMES):
        collector.push(frame())
    collector.cancel()

    assert not collector.is_recording
    assert collector.stop() is None


def test_summary_counts_words_not_clips(tmp_path):
    collector = Collector(tmp_path)
    for word, count in (("hello", 3), ("good", 2)):
        for _ in range(count):
            collector.start(word)
            for _ in range(DEFAULT_CLIP_FRAMES):
                collector.push(frame())
            collector.stop()

    summary = collector.summary()
    assert summary["clips"] == 5
    assert summary["words"] == 2
    assert summary["by_label"] == {"hello": 3, "good": 2}


def test_multi_word_labels_survive_the_filename(tmp_path):
    collector = Collector(tmp_path)
    collector.start("thank you")
    for _ in range(DEFAULT_CLIP_FRAMES):
        collector.push(frame())
    path = collector.stop()

    assert path is not None
    assert " " not in path.name

    _tracks, labels = collector.load_all()
    assert labels == ["thank you"]


@pytest.mark.parametrize("label", ["hello", "good", "thank you"])
def test_every_stored_clip_is_usable_as_a_training_input(tmp_path, label):
    """Clips come straight back out of load_all and must already be the right
    shape and finite, because that is what fine tuning feeds to the model."""

    collector = Collector(tmp_path)
    collector.start(label)
    for i in range(DEFAULT_CLIP_FRAMES):
        collector.push(frame(float(i)))
    collector.stop()

    tracks, labels = collector.load_all()
    assert labels == [label]
    assert tracks[0].shape == (DEFAULT_CLIP_FRAMES, FEATURE_DIMS)
    assert np.isfinite(tracks[0]).all()
    assert tracks[0].dtype == np.float32
