"""WLASL landmark sequences turned into training tensors.

The corpus is stored as ``(frames, 126)`` arrays: two hands, 21 landmarks each,
as (x, y, detection confidence). Three things have to happen before a model can
learn from that.

Hand selection. A clip has two hand slots and the tracker often fills only one.
Picking the most confident hand frame by frame makes the hand swap mid
sequence, which destroys the temporal structure the model depends on. Instead
the dominant hand is chosen once, over the whole clip.

Missing hands. Frames where the chosen hand was never detected are not padding
noise, they are information. They encode as an all-zero feature vector, which is
the same convention the live pipeline uses, so training and inference match.

Augmentation. This is the part that matters most. WLASL has roughly six clips
per word, so the model will memorise rather than generalise unless every
plausible variation of a clip is generated on the fly: different camera
framing, different timing, different handedness.

There is no depth channel in this corpus, so z is zero throughout. The encoder
still works, because curl and reach are computed from x and y.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.types import LANDMARK_COUNT, Hand, Handedness
from .features import FEATURE_DIMS, encode_hand, encode_sequence

WINDOW = 64
HANDS = 2
PER_HAND = LANDMARK_COUNT * 3  # x, y, confidence
RAW_DIMS = HANDS * PER_HAND  # 126

#: Offsets into the feature vector, used by the augmentation below.
_COORDS = LANDMARK_COUNT * 3  # 63: normalised landmark coordinates
_PALM_NORMAL = _COORDS + 10  # 73: after five curls and five reaches


@dataclass(slots=True)
class Sample:
    video_id: str
    gloss: str
    split: str
    signer: int
    path: Path


class WLASLIndex:
    """The joined video_id -> (gloss, split, signer) index."""

    def __init__(self, index_path: Path) -> None:
        raw = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.samples = [
            Sample(
                video_id=s["video_id"],
                gloss=s["gloss"],
                split=s["split"],
                signer=s["signer"],
                path=Path(s["path"]),
            )
            for s in raw["samples"]
        ]
        self.glosses = list(raw["glosses"])
        self.by_id = {s.video_id: i for i, s in enumerate(self.samples)}
        self.label_of = {g: i for i, g in enumerate(self.glosses)}

    def __len__(self) -> int:
        return len(self.samples)

    def split(self, name: str) -> list[Sample]:
        return [s for s in self.samples if s.split == name]


# --------------------------------------------------------------------- decode


def load_clip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (landmarks (T, 21, 3) as x/y/z, confidence (T,))."""

    raw = np.load(path, allow_pickle=False).astype(np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_DIMS:
        raise ValueError(f"{path.name}: unexpected shape {raw.shape}")

    # Split into the two hand slots and reshape each to (T, 21, 3).
    hands = raw.reshape(raw.shape[0], HANDS, LANDMARK_COUNT, 3)
    conf = hands[..., 2].mean(axis=1)  # (T, 2)

    dominant = int(np.argmax((conf > 0.2).sum(axis=0)))
    chosen = hands[:, dominant, :, :2]  # (T, 21, 2)
    confidence = conf[:, dominant]

    # z is absent in this corpus. A constant keeps the shape without inventing
    # depth the model would then learn to rely on.
    out = np.zeros((raw.shape[0], LANDMARK_COUNT, 3), dtype=np.float32)
    out[..., :2] = chosen
    return out, confidence


def clip_to_features(landmarks: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Encode a whole clip into a (frames, FEATURE_DIMS) feature track.

    Frames below the detection threshold become zeros, exactly as an absent
    hand does at inference time.
    """

    tracks = np.zeros((len(landmarks), FEATURE_DIMS), dtype=np.float32)
    seen = confidence > 0.2
    for i in np.flatnonzero(seen):
        tracks[i] = encode_hand(Hand(landmarks=landmarks[i], handedness=Handedness.RIGHT))
    return tracks


def encode_clip(path: Path) -> np.ndarray:
    landmarks, confidence = load_clip(path)
    return clip_to_features(landmarks, confidence)


# ---------------------------------------------------------------- augmentation


def _rotate_2d(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]], dtype=np.float32)


def augment(track: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One augmented copy of a feature track.

    Works on the encoded features rather than raw landmarks because that is
    where the invariances already live, so augmentation only has to cover what
    normalisation cannot: timing, framing and noise.
    """

    out = track.copy()
    t = len(out)
    if t < 4:
        return out

    # Temporal crop: signs start and stop at different moments.
    keep = int(t * rng.uniform(0.55, 1.0))
    keep = max(4, min(t, keep))
    start = int(rng.integers(0, t - keep + 1))
    out = out[start : start + keep]

    # Temporal resize: the same sign played faster or slower.
    speed = rng.uniform(0.8, 1.25)
    new_len = max(4, int(len(out) * speed))
    src = np.linspace(0, len(out) - 1, new_len)
    idx = np.clip(np.round(src).astype(int), 0, len(out) - 1)
    out = out[idx]

    # The coordinate block is the first 63 features. Jitter it to stand in for
    # different camera placement, hand size and viewing angle.
    xyz = out[:, :_COORDS].reshape(-1, LANDMARK_COUNT, 3)
    if rng.random() < 0.8:
        turned = _rotate_2d(xyz[..., :2], rng.uniform(-0.25, 0.25))
        xyz = np.concatenate([turned, xyz[..., 2:3]], axis=-1)
    xyz = xyz * rng.uniform(0.85, 1.2)
    xyz = xyz + rng.normal(0.0, 0.02, size=xyz.shape)
    out[:, :_COORDS] = xyz.reshape(len(out), _COORDS).astype(np.float32)

    # Mirror: flip x and flip the palm normal, which is what keeps the mirrored
    # hand a physically possible hand rather than a left hand seen from behind.
    if rng.random() < 0.5:
        out[:, 0::3] *= -1.0
        out[:, _PALM_NORMAL : _PALM_NORMAL + 3] *= -1.0

    # Frame dropout, standing in for tracker flicker.
    if rng.random() < 0.35:
        out[rng.random(len(out)) < 0.08] = 0.0

    # Amplitude jitter on the derived features.
    derived = out[:, _COORDS:]
    out[:, _COORDS:] = derived + rng.normal(0.0, 0.03, size=derived.shape)

    return out.astype(np.float32)


def resize_track(track: np.ndarray, window: int) -> np.ndarray:
    """Stretch or squeeze a whole clip into exactly `window` frames.

    Used for evaluation, where every frame of the clip is evidence and there is
    no reason to crop. Training uses `fit_window` instead, which deliberately
    crops so the model does not learn that a sign always spans the full window.
    """

    if len(track) == 0:
        return np.zeros((window, FEATURE_DIMS), dtype=np.float32)
    src = np.linspace(0, len(track) - 1, window)
    idx = np.clip(np.round(src).astype(int), 0, len(track) - 1)
    return track[idx].astype(np.float32)


def fit_window(
    track: np.ndarray, window: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Right-align a variable length track into a fixed window.

    With an rng, take a random slice instead of the tail, which stops the model
    from learning that the sign always ends at the last frame.
    """

    if rng is not None and len(track) > window:
        start = int(rng.integers(0, len(track) - window + 1))
        track = track[start : start + window]
    return encode_sequence(list(track), window)


def mixup(
    a: np.ndarray, b: np.ndarray, alpha: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    lam = float(rng.beta(alpha, alpha))
    return (lam * a + (1.0 - lam) * b).astype(np.float32), lam
