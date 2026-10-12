"""Persisted user settings.

Settings live in %LOCALAPPDATA%\\SignBridge\\settings.json so the repo stays clean
and the app works from any folder. Everything here has a working default, the
app is fully usable before the file is ever written.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "SignBridge"


def repo_root() -> Path:
    """Project root when running from source, otherwise the installed package parent."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


def _assets_dir() -> Path:
    env = os.environ.get("SIGNBRIDGE_ASSETS")
    return Path(env) if env else repo_root() / "assets"


@dataclass(slots=True)
class Settings:
    """Everything the user can tweak from the UI."""

    # Capture
    camera_index: int = 0
    source: str = "camera"  # camera | video | simulated
    video_path: str = ""
    mirror: bool = True
    target_fps: int = 30
    frame_width: int = 1280
    frame_height: int = 720

    # Recognition
    hand_confidence: float = 0.55
    min_sign_confidence: float = 0.62
    window_frames: int = 48
    signs_per_phrase: int = 0  # 0 means speak every confirmed sign on its own
    phrase_pause_ms: float = 900.0

    # Speech
    tts_engine: str = "piper"  # "piper" | "sapi" | "off"
    tts_voice: str = "en_US-hfc_female-medium"
    tts_rate: float = 1.0
    tts_volume: float = 1.0
    speak_automatically: bool = True

    # UI
    show_skeleton: bool = True
    show_confidence: bool = True
    theme: str = "nocturne"

    # Internal, not shown in the UI
    last_vocab: str = field(default="", repr=False)

    # ------------------------------------------------------------------ io

    @classmethod
    def path(cls) -> Path:
        return APP_DIR / "settings.json"

    @classmethod
    def load(cls) -> Settings:
        p = cls.path()
        if not p.exists():
            return cls()
        try:
            raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        try:
            self.path().parent.mkdir(parents=True, exist_ok=True)
            self.path().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass  # read-only profile, settings simply stay in memory

    # -------------------------------------------------------------- assets

    @property
    def models_dir(self) -> Path:
        return _assets_dir() / "models"

    @property
    def voices_dir(self) -> Path:
        return _assets_dir() / "voices"

    @property
    def data_dir(self) -> Path:
        return APP_DIR / "data"
