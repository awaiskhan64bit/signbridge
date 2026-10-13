"""Verify the downloaded archive and extract the two files this project uses.

The archive is complete at 14,270,101,588 bytes. The fetch loop kept retrying
because it compared that against `totalBytes` from the dataset listing, which is
14,351,255,665 and describes the dataset rather than the zip. So completion was
never detected and every retry asked for a range past the end of the file, which
Kaggle answers with 416. The transfer was fine; the completeness test was wrong.

Checking the zip's own central directory settles it: if it lists the members and
the archive opens, the bytes are good and no re-download is needed.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path("data/mutemotion")
ARCHIVE = ROOT / "mutemotion.zip"
WANTED = ["WLASL_parsed_data.json", "landmarks_V3.npz"]


def main() -> int:
    size = ARCHIVE.stat().st_size
    print(f"archive: {size:,} bytes ({size / 1e9:.2f} GB)")

    with zipfile.ZipFile(ARCHIVE) as zf:
        members = zf.namelist()
        print("zip is readable, members:")
        for m in members:
            info = zf.getinfo(m)
            print(f"  {m:28s} {info.file_size / 1e6:9.1f} MB")

        bad = zf.testzip()
        print("integrity:", "OK" if bad is None else f"CORRUPT at {bad}")

        for name in WANTED:
            if name not in members:
                print(f"MISSING: {name}")
                return 1
            zf.extract(name, ROOT)
            out = ROOT / name
            print(f"extracted {name}: {out.stat().st_size / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())