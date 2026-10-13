"""Download the MuteMotion landmark archive.

Kaggle serves this dataset as one 14.27GB zip with no per-file endpoint, so the
transfer cannot be narrowed the way it first looks. What can be narrowed is what
gets written to disk: only the metadata json and the landmark array this project
actually uses are extracted, and the archive is streamed rather than buffered so
14GB never lands on a 45GB volume alongside the existing corpus.

Which landmark file to take. The archive holds three: `landmarks_V1.npz` and
`V2.npz` at 2.7GB each and `V3.npz` at 8.9GB. The dataset card describes V3 as
the one carrying all 553 pose, face and hand landmarks. V1 and V2 are the same
clips with a smaller landmark set, and the hand-only data we already hold covers
that. So V3 is the one that adds something, and it is extracted alone.

Transfer is resumable through HTTP range requests, so a dropped connection costs
the partial archive rather than restarting 14GB.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

with open(os.path.expanduser("~/.kaggle/kaggle.json"), encoding="utf-8") as fh:
    KEY = json.load(fh)["key"]
URL = "https://www.kaggle.com/api/v1/datasets/download/abd0kamel/mutemotion-output"

ROOT = Path("data/mutemotion")
ARCHIVE = ROOT / "mutemotion.zip"
#: Metadata plus the full-landmark array. Nothing else is written to disk.
WANTED = {"WLASL_parsed_data.json", "landmarks_V3.npz"}


def total_size() -> int:
    """Archive size in bytes, or 0 when it cannot be established.

    `totalBytes` from the dataset listing describes the dataset, not the zip, and
    it is about 81MB larger. Comparing a completed transfer against it means
    completion is never reached, so the loop retries and every retry asks for a
    range past the end of the file, which Kaggle answers with 416. The first
    pass downloaded all 14.27GB correctly and then failed for twenty attempts
    over a mismatch that could not have affected the data.

    Completion is therefore judged from the archive itself: a zip that opens and
    lists its members is finished, whatever the listing claims.
    """

    req = urllib.request.Request(
        "https://www.kaggle.com/api/v1/datasets/view/abd0kamel/mutemotion-output",
        headers={"Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(json.loads(r.read())["totalBytes"])


def already_complete() -> bool:
    """True when the archive on disk is a readable zip with all members present."""

    if not ARCHIVE.exists() or ARCHIVE.stat().st_size < 1_000_000_000:
        return False
    try:
        with zipfile.ZipFile(ARCHIVE) as zf:
            members = set(zf.namelist())
    except zipfile.BadZipFile:
        return False
    return members >= WANTED


def fetch_archive(attempts: int = 6) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if already_complete():
        print("archive already present and complete, skipping download", flush=True)
        return
    total = total_size()
    have = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    print(f"archive: {total / 1e9:.2f} GB, {have / 1e9:.2f} GB already on disk", flush=True)
    if total and have == total:
        print("  already complete")
        return

    for attempt in range(1, attempts + 1):
        if have >= total:
            break
        headers = {"Authorization": f"Bearer {KEY}"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(URL, headers=headers), timeout=180
            ) as r, open(ARCHIVE, "ab" if have else "wb") as fh:
                if have and r.status != 206:
                    # The server ignored the range and restarted the stream.
                    have = 0
                    fh.seek(0)
                    fh.truncate()
                start, last_report, last_write = time.time(), time.time(), time.time()
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    have += len(chunk)
                    last_write = time.time()
                    if time.time() - last_report > 20:
                        rate = (have - (total - have if False else 0)) / max(
                            time.time() - start, 1e-6
                        )
                        pct = 100.0 * have / total if total else 0.0
                        print(
                            f"  {have / 1e9:6.2f} / {total / 1e9:.2f} GB "
                            f"({pct:5.1f}%)  {rate / 1e6:5.1f} MB/s",
                            flush=True,
                        )
                        last_report = time.time()
                    if time.time() - last_write > 240:
                        print("  stalled, reconnecting", flush=True)
                        break
            print(f"  pass {attempt}: {have / 1e9:.2f} GB", flush=True)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  attempt {attempt} failed at {have / 1e9:.2f} GB: {exc}", flush=True)
            time.sleep(5)
    if total and have < total:
        raise SystemExit(f"incomplete: {have / 1e9:.2f} of {total / 1e9:.2f} GB")


def extract() -> None:
    print("extracting the two files we use", flush=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        names = set(zf.namelist())
        for name in WANTED:
            if name not in names:
                raise SystemExit(f"{name} missing from archive; has {sorted(names)[:8]}")
        info = zf.getinfo("landmarks_V3.npz")
        print(f"  landmarks_V3.npz uncompressed {info.file_size / 1e9:.2f} GB", flush=True)
        for name in WANTED:
            zf.extract(name, ROOT)
            size = (ROOT / name).stat().st_size
            print(f"  extracted {name} ({size / 1e6:.1f} MB)", flush=True)


def cleanup() -> None:
    """Drop the archive once the parts we need are on disk."""

    if ARCHIVE.exists():
        ARCHIVE.unlink()
        print("removed the archive", flush=True)


def main() -> int:
    fetch_archive()
    extract()
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())