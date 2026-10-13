"""Recording the user's own signing, to close the domain gap.

Every landmark dataset that can be downloaded is somebody else's tightly cropped
sign, recorded on a phone. The app runs on a webcam, with the signer filling a
different part of the frame, moving differently, at a different frame rate. That
gap is why the shipped model is accurate on WLASL clips and silent on real video.

Fine tuning on a few minutes of the user's own footage is the cheapest way across
it, and it is far more effective than any amount of extra augmentation. This
module stores those recordings.

A recording is a label plus the encoded feature track for each frame, which is
the same representation the live pipeline already produces. Nothing here knows
about pixels, so a clip recorded through the app is byte for byte the kind of
input the model sees at inference.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_DIMS

#: A clip is a couple of seconds of signing. WLASL clips run 25 to 149 frames at
#: roughly 25 fps, so this matches what the model was trained on.
DEFAULT_CLIP_FRAMES = 64


def normalise_label(text: str) -> str:
    """Fold what a person types into the shape WLASL uses for glosses."""

    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned


@dataclass(slots=True)
class Recording:
    label: str
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return len(self.frames) >= 8

    def track(self, length: int = DEFAULT_CLIP_FRAMES) -> np.ndarray:
        """Right aligned feature track, matching the inference window."""

        from .features import encode_sequence

        return encode_sequence(list(self.frames), length)

    def to_npz(self) -> dict[str, np.ndarray]:
        """Flattened for storage: one (length, features) track plus its label."""

        track = self.track()
        return {
            "frames": track,
            "label": np.array([self.label]),
            "length": np.array([len(self.frames)]),
        }


class Collector:
    """Gathers labelled clips in memory and flushes them to disk.

    Each sign is one clip, matching how WLASL is structured. Recording a few
    dozen clips for a few dozen words is enough to fine tune, and is far less
    work than it sounds because the app does the encoding.
    """

    def __init__(self, root: Path, clip_frames: int = DEFAULT_CLIP_FRAMES) -> None:
        self.root = Path(root)
        self.clip_frames = clip_frames
        self._recording: Recording | None = None
        self._captured = 0
        self._sequence = 0

    # ---------------------------------------------------------------- state

    @property
    def recording_label(self) -> str | None:
        return self._recording.label if self._recording else None

    @property
    def frames_needed(self) -> int:
        return self.clip_frames

    @property
    def frames_have(self) -> int:
        return len(self._recording.frames) if self._recording else 0

    @property
    def captured(self) -> int:
        """Clips stored so far."""

        return self._captured

    @property
    def is_recording(self) -> bool:
        return self._recording is not None

    @property
    def full(self) -> bool:
        return self.frames_have >= self.clip_frames

    # -------------------------------------------------------------- actions

    def start(self, label: str) -> bool:
        """Begin a clip. Refuses a label that would poison the training set."""

        label = normalise_label(label)
        if not label or len(label) > 40:
            return False
        self._recording = Recording(label=label)
        return True

    def cancel(self) -> None:
        self._recording = None

    def push(self, features: np.ndarray) -> bool:
        """Add one frame. Returns True once the clip is full."""

        if self._recording is None:
            return False
        self._recording.frames.append(np.asarray(features, dtype=np.float32))
        if len(self._recording.frames) > self.clip_frames:
            self._recording.frames.pop(0)  # keep only the most recent window
        return self.full

    def stop(self) -> Path | None:
        """Store the clip if it has enough signal. Frames of an absent hand do
        not count, so waving at nothing cannot become a training example."""

        if self._recording is None:
            return None
        clip = self._recording
        self._recording = None
        if not clip.ready:
            return None

        present = sum(1 for f in clip.frames if np.abs(f).sum() > 1e-6)
        if present < clip.frames.__len__() * 0.4:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        # A counter as well as the timestamp. Someone recording the same word
        # five times in a row can easily land inside the same millisecond, and
        # those clips would overwrite each other.
        self._sequence += 1
        stem = normalise_label(clip.label).replace(" ", "_")
        path = self.root / f"{stem}_{time.time_ns()}_{self._sequence:04d}.npz"
        np.savez_compressed(path, **clip.to_npz())
        self._captured += 1
        return path

    # -------------------------------------------------------------- reading

    def load_all(self) -> tuple[list[np.ndarray], list[str]]:
        """Every stored clip, for the fine tuning script."""

        tracks: list[np.ndarray] = []
        labels: list[str] = []
        if not self.root.exists():
            return tracks, labels
        for path in sorted(self.root.glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    frame = data["frames"]
                    label = str(data["label"][0])
            except (OSError, ValueError, KeyError):
                continue
            if frame.ndim != 2 or frame.shape[-1] != FEATURE_DIMS or not label:
                continue
            tracks.append(frame.astype(np.float32))
            labels.append(label)
        return tracks, labels

    def summary(self) -> dict:
        tracks, labels = self.load_all()
        by_label: dict[str, int] = {}
        for label in labels:
            by_label[label] = by_label.get(label, 0) + 1
        return {
            "clips": len(tracks),
            "words": len(by_label),
            "by_label": by_label,
            "root": str(self.root),
        }
