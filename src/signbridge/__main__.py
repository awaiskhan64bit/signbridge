"""``python -m signbridge`` starts the desktop app."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="signbridge",
        description="Offline camera sign language to speech.",
    )
    parser.add_argument(
        "--source",
        choices=("camera", "video", "simulated"),
        help="override the input source for this run (simulated needs no camera)",
    )
    parser.add_argument("--video", help="path to a clip to use with --source video")
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="run with the voice engine muted",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="DEBUG, INFO, WARNING or ERROR",
    )
    args = parser.parse_args()

    from .config import Settings
    from .ui.app import run

    settings = Settings.load()
    if args.source:
        settings.source = args.source
    if args.video:
        settings.video_path = args.video
        settings.source = "video"
    if args.no_speech:
        settings.tts_engine = "off"
    settings.save()

    sys.argv = [sys.argv[0]]
    sys.exit(run())


if __name__ == "__main__":
    main()
