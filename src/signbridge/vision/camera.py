"""Threaded frame capture.

One background thread owns the camera and pushes frames onto a small queue so the
Qt event loop never blocks on OpenCV. The same class backs three input sources,
which is what makes the app demo-able on a machine with no webcam attached:

    Source.CAMERA      a real OpenCV capture device
    Source.VIDEO       a clip on disk, loops when it reaches the end
    Source.SIMULATED   a synthetic signer rendered from the built in landmark rig
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ..core.types import Frame, Source


class CameraUnavailable(RuntimeError):
    """Raised when a capture device cannot be opened."""


class VideoSource:
    """Base class: pull the next frame or raise StopIteration when drained."""

    def open(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover - interface
        pass

    @property
    def size(self) -> tuple[int, int]:
        return (1280, 720)


class CameraSource(VideoSource):
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        # The DirectShow backend is far more reliable than the default on Windows.
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise CameraUnavailable(f"Could not open camera index {self.index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self) -> np.ndarray | None:
        assert self._cap is not None, "open() first"
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        self.width, self.height = w, h
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FileSource(VideoSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if not self.path.exists():
            raise CameraUnavailable(f"No such video: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise CameraUnavailable(f"Could not decode {self.path.name}")

    def read(self) -> np.ndarray | None:
        assert self._cap is not None, "open() first"
        ok, frame = self._cap.read()
        if not ok or frame is None:
            # Loop the clip so a short sample keeps demonstrating.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self.height, self.width = frame.shape[:2]
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CaptureWorker:
    """Pulls frames in the background and hands them over as :class:`Frame`."""

    def __init__(
        self,
        source: Source = Source.CAMERA,
        *,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        max_queue: int = 2,
    ) -> None:
        self.source = source
        self.fps = max(1, min(fps, 60))
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._max_queue = max_queue
        self._queue: list[Frame] = []
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend: VideoSource | None = None
        self.error: str | None = None
        self.frames_captured = 0

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> CaptureWorker:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ---------------------------------------------------------------- frame

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Block for the next frame. Returns None once stopped and drained."""

        deadline = time.monotonic() + timeout
        with self._cv:
            while not self._queue and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            if not self._queue:
                return None
            return self._queue.pop(0)

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    # ----------------------------------------------------------------- loop

    def _build_backend(self) -> VideoSource:
        if self.source is Source.CAMERA:
            return CameraSource(self.camera_index, self.width, self.height)
        if self.source is Source.VIDEO:
            return FileSource(self._video_path)
        from .simulated import SimulatedSigner

        return SimulatedSigner(width=self.width, height=self.height)

    _video_path: Path = Path(".")

    def set_video(self, path: str | Path) -> None:
        self._video_path = Path(path)
        self.source = Source.VIDEO

    def _push(self, frame: Frame) -> None:
        with self._cv:
            self._queue.append(frame)
            while len(self._queue) > self._max_queue:
                self._queue.pop(0)
            self._cv.notify_all()

    def _run(self) -> None:
        try:
            self._backend = self._build_backend()
            self._backend.open()
        except CameraUnavailable as exc:
            self.error = str(exc)
            return

        period = 1.0 / self.fps
        start = time.perf_counter()
        index = 0
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            image = self._backend.read()
            if image is None:
                time.sleep(0.005)
                continue
            index += 1
            self.frames_captured = index
            # A synthetic backend attaches its own hands, so the pipeline can skip
            # running the tracker over frames that contain no real video anyway.
            hands = list(getattr(self._backend, "hands", ()) or ())
            self._push(
                Frame(
                    image=image,
                    hands=hands,
                    index=index,
                    timestamp_ms=(time.perf_counter() - start) * 1000.0,
                )
            )
            # Pace to the target fps instead of spinning a core at capture speed.
            sleep_for = period - (time.perf_counter() - loop_start)
            if sleep_for > 0:
                self._stop.wait(sleep_for)

        self._backend.release()
