"""One time asset download, so the app itself never touches the network.

Two things are fetched here, both once, both into the project's ``assets``
folder which is git-ignored:

  * MediaPipe's hand landmarker model, 7.5 MB. Without it the app still runs but
    cannot see hands, and only the simulated signer works.
  * A Piper voice, around 60 MB. Without it the app falls back to Windows SAPI.

After this has run the application is completely offline.

Usage
-----
    python -m signbridge.setup              # everything that is missing
    python -m signbridge.setup --voices     # voices only
    python -m signbridge.setup --list       # what is present, no download
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Absolute, not relative: `python -m signbridge.setup` runs this file as
# __main__, and relative imports above the package root fail in that context.
from signbridge.config import Settings

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_BYTES = 7_819_105

#: Piper voices, as (name, Hugging Face path under rhasspy/piper-voices).
#: lessac leads the list because it is the best sounding voice that is not
#: restricted to non-commercial use. See docs/licensing.md.
VOICES: dict[str, str] = {
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-amy-low": "en/en_US/amy/low",
    "en_US-kathleen-low": "en/en_US/kathleen/low",
    "en_US-ryan-high": "en/en_US/ryan/high",
}
VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
DEFAULT_VOICE = "en_US-lessac-medium"


def _download(url: str, dest: Path, expected: int | None = None) -> bool:
    """Fetch one file. Returns True on success, never raises for a bad network."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print(f"\r  {pct:5.1f}%  {done / 1e6:6.1f} MB", end="", flush=True)
        print()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False

    if expected and tmp.stat().st_size != expected:
        print(f"  size mismatch: got {tmp.stat().st_size}, expected {expected}")
        tmp.unlink(missing_ok=True)
        return False

    shutil.move(str(tmp), str(dest))
    return True


def fetch_hand_model(settings: Settings) -> bool:
    dest = settings.models_dir / "hand_landmarker.task"
    if dest.exists() and dest.stat().st_size == HAND_LANDMARKER_BYTES:
        print(f"hand model: present ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print("hand model: downloading, 7.5 MB")
    if _download(HAND_LANDMARKER_URL, dest, HAND_LANDMARKER_BYTES):
        print(f"  saved to {dest}")
        return True
    print("  the app will run but cannot track hands until this succeeds")
    return False


def fetch_voice(settings: Settings, name: str) -> bool:
    folder = VOICES.get(name)
    if folder is None:
        print(f"unknown voice {name!r}. Known: {', '.join(VOICES)}")
        return False

    base = settings.voices_dir / name
    model, config = base.with_suffix(".onnx"), base.with_suffix(".onnx.json")
    if model.exists() and config.exists():
        print(f"voice {name}: present ({model.stat().st_size / 1e6:.1f} MB)")
        return True

    print(f"voice {name}: downloading")
    # The ?download=true suffix matters. Without it Hugging Face serves an HTML
    # redirect page under a .onnx name and Piper fails with a protobuf error.
    ok = _download(f"{VOICE_BASE}/{folder}/{name}.onnx?download=true", model)
    ok &= _download(f"{VOICE_BASE}/{folder}/{name}.onnx.json?download=true", config)
    if ok:
        print(f"  saved to {model}")
    return ok


def report(settings: Settings) -> None:
    print(f"assets in {settings.models_dir.parent}")
    model = settings.models_dir / "hand_landmarker.task"
    print(f"  hand model : {'ok' if model.exists() else 'MISSING'}")
    classifier = settings.models_dir / "sign_classifier.onnx"
    state = "ok" if classifier.exists() else "not trained (run scripts/train.py)"
    print(f"  classifier : {state}")
    voices = sorted(settings.voices_dir.glob("*.onnx")) if settings.voices_dir.exists() else []
    if voices:
        print("  voices     : " + ", ".join(p.stem for p in voices))
    else:
        print("  voices     : MISSING (falls back to Windows SAPI)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="signbridge.setup", description=__doc__)
    parser.add_argument("--models", action="store_true", help="MediaPipe models only")
    parser.add_argument("--voices", action="store_true", help="Piper voices only")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="which voice to fetch")
    parser.add_argument("--list", action="store_true", help="report and exit")
    args = parser.parse_args(argv)

    settings = Settings.load()
    if args.list:
        report(settings)
        return 0

    want_all = not (args.models or args.voices)
    ok = True
    if want_all or args.models:
        ok &= fetch_hand_model(settings)
    if want_all or args.voices:
        ok &= fetch_voice(settings, args.voice)

    print()
    report(settings)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
