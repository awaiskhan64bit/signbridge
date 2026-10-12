"""Fetch the WLASL hand-landmark corpus and build a training index.

Two sources are joined here:

  * ``a7medwa2l/wlasl-signals`` on Hugging Face holds MediaPipe hand landmarks
    as raw ``.npy`` files named after the WLASL ``video_id``. The upload ships no
    labels and no licence, which is why this script exists at all: it attaches
    the labels from the official metadata and records where the data came from.

  * ``WLASL_v0.3.json`` from the official WLASL repository supplies the gloss,
    the official train/val/test split and the signer id for every video.

Each ``.npy`` is ``(frames, 126)``: two hands, 21 landmarks each, stored as
(x, y, detection confidence). There is no depth channel, so the loader treats z
as zero.

Usage
-----
    python scripts/fetch_wlasl.py                 # download and index
    python scripts/fetch_wlasl.py --index-only    # rebuild the index, no download
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ID = "a7medwa2l/wlasl-signals"
REPO_TYPE = "dataset"
WLASL_JSON = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
PATTERNS = ["part_*/*.npy"]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LANDMARKS = DATA / "wlasl-landmarks"
INDEX = DATA / "wlasl-index.json"


def fetch_metadata(dest: Path) -> list[dict]:
    """Download WLASL_v0.3.json unless it is already on disk."""

    if not dest.exists():
        import urllib.request

        print(f"downloading {WLASL_JSON}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WLASL_JSON, dest)
    print(f"metadata: {dest.stat().st_size / 1e6:.1f} MB")
    return json.loads(dest.read_text(encoding="utf-8"))


def download_landmarks() -> Path:
    """Snapshot the landmark files. Resumable, and skips what is already there."""

    from huggingface_hub import snapshot_download

    LANDMARKS.mkdir(parents=True, exist_ok=True)
    print(f"downloading landmark files into {LANDMARKS} (this is ~430 MB)")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(LANDMARKS),
        allow_patterns=PATTERNS,
        max_workers=16,
    )
    return LANDMARKS


def build_index() -> dict:
    """Join every landmark file to its gloss, split and signer."""

    meta = fetch_metadata(DATA / "WLASL_v0.3.json")

    # video_id -> (gloss, split, signer_id)
    lookup: dict[str, tuple[str, str, int]] = {}
    for entry in meta:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            lookup[str(inst["video_id"])] = (gloss, inst["split"], inst["signer_id"])

    files = sorted(LANDMARKS.glob("part_*/*.npy"))
    print(f"found {len(files)} landmark files")

    samples: list[dict] = []
    missing: list[str] = []
    for path in files:
        key = path.stem
        hit = lookup.get(key)
        if hit is None:
            missing.append(key)
            continue
        gloss, split, signer = hit
        samples.append(
            {
                "video_id": key,
                "gloss": gloss,
                "split": split,
                "signer": signer,
                "path": path.relative_to(ROOT).as_posix(),
            }
        )

    glosses = sorted({s["gloss"] for s in samples})
    index = {
        "repo": REPO_ID,
        "metadata": WLASL_JSON,
        "samples": samples,
        "glosses": glosses,
    }
    INDEX.write_text(json.dumps(index), encoding="utf-8")

    counts = Counter(s["split"] for s in samples)
    per_gloss = Counter(s["gloss"] for s in samples)
    print(f"\nwrote {INDEX}")
    print(f"  samples      : {len(samples)}")
    print(f"  glosses      : {len(glosses)}")
    print(f"  splits       : {dict(counts)}")
    print(f"  unmatched    : {len(missing)}")
    if per_gloss:
        sizes = sorted(per_gloss.values())
        print(f"  per gloss    : min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(
        "  note         : a long-tail vocabulary means most words have only a\n"
        "                 handful of clips, which caps achievable accuracy."
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-only", action="store_true", help="skip the download")
    args = parser.parse_args()

    if not args.index_only:
        download_landmarks()
    build_index()
    return 0


if __name__ == "__main__":
    sys.exit(main())
