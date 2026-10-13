"""Join the MuteMotion landmarks to WLASL labels and measure vocabulary coverage.

The archive keys each clip by WLASL video_id and `WLASL_parsed_data.json` carries
the same id inside `video_path`, so the two join without a fuzzy match. That
matters: a loose join would silently attach the wrong word to a clip, which
produces a model that trains to a plausible-looking loss and recognises nothing.

What is checked:

  * how many of the 21,083 clips have landmarks
  * how the official splits line up
  * clips per word for the 137 curated words, against what we had before
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from signbridge.ml import vocabulary

ROOT = Path("data/mutemotion")


def main() -> int:
    meta = json.loads((ROOT / "WLASL_parsed_data.json").read_text(encoding="utf-8"))

    npz = np.load(ROOT / "landmarks_V3.npz", allow_pickle=True)
    have = set(k for k in npz if k.isdigit())
    print(f"parsed entries : {len(meta)}")
    print(f"landmark keys  : {len(npz.keys())} ({len(have)} numeric)")
    print(f"clips matched  : {len(have)}")

    rows = []
    unmatched = 0
    for entry in meta:
        m = re.search(r"(\d+)\.mp4", entry["video_path"])
        vid = m.group(1) if m else None
        if vid is None or vid not in have:
            unmatched += 1
            continue
        rows.append((vid, entry["gloss"], entry["split"]))

    print(f"joined rows    : {len(rows)}  (unmatched {unmatched})")
    print("splits         :", dict(Counter(r[2] for r in rows)))

    # Landmarks per frame, and are pose joints actually populated?
    sample = rows[0][0]
    arr = npz[sample]
    print()
    print(f"sample clip {sample}: frames={arr.shape[0]} joints={arr.shape[1]} dims={arr.shape[2]}")
    frame = arr[0]
    pose = frame[:33]
    print(f"  pose joints populated : {int((np.abs(pose).sum(axis=1) > 1e-9).sum())}/33")
    print(f"  face joints populated : {int((np.abs(frame[33:501]).sum(axis=1) > 1e-9).sum())}/468")
    print(f"  hand joints populated : {int((np.abs(frame[501:]).sum(axis=1) > 1e-9).sum())}/52")

    # Coverage of the curated vocabulary.
    per_word = Counter(r[1] for r in rows)
    wanted = vocabulary.resolve(per_word)
    print()
    print(f"curated words resolvable : {len(wanted)}")

    before = json.loads(Path("data/wlasl-index.json").read_text(encoding="utf-8"))
    old = Counter(s["gloss"] for s in before["samples"])
    n = len(wanted)
    new = sorted(per_word[w] for w in wanted)
    prev = sorted(old[w] for w in wanted if w in old)
    print(f"clips per word, now      : min {new[0]}, median {new[n // 2]}, max {new[-1]}")
    print(
        f"clips per word, hand-only: min {prev[0]}, "
        f"median {prev[len(prev) // 2]}, max {prev[-1]}"
    )
    print(f"total clips for curated  : {sum(per_word[w] for w in wanted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())