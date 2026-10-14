"""Selecting a voice must never stall the UI thread.

Piper loads a neural voice off disk and builds an inference session, which takes
seconds. That load used to be triggered by `available()`, which engine selection
calls, and engine selection runs on the UI thread. A frozen UI is indistinguishable
from a stalled camera: no frames arrive and nothing raises, which is exactly the
report that sent us looking here.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from signbridge.speech.piper import PiperEngine


@pytest.fixture
def voices(tmp_path: Path) -> Path:
    """A voices directory whose files exist but whose contents are not a model."""

    d = tmp_path / "voices"
    d.mkdir()
    (d / "en_US-lessac-medium.onnx").write_bytes(b"not a real model")
    (d / "en_US-lessac-medium.onnx.json").write_text("{}")
    return d


class TestAvailabilityIsCheap:
    def test_available_does_not_load_the_voice(self, voices):
        engine = PiperEngine(voices)
        # Deliberately not calling warm(). If available() loaded weights, this
        # would either raise on the fake file or take the load path.
        assert engine.available() is True
        assert engine._voice is None, "available() loaded the model, which blocks the caller"

    def test_available_is_false_when_the_voice_is_missing(self, tmp_path):
        assert PiperEngine(tmp_path / "voices").available() is False

    def test_available_reports_the_error_without_raising(self, tmp_path):
        # A missing voice has to be reportable, because build_engine falls back to
        # the Windows voice on the strength of exactly this answer.
        engine = PiperEngine(tmp_path / "voices")
        assert engine.available() is False
        assert engine.load_error


class TestWarming:
    def test_warm_returns_immediately(self, voices):
        engine = PiperEngine(voices)
        start = time.perf_counter()
        engine.warm()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.25, f"warm() blocked for {elapsed:.2f}s"

    def test_warming_twice_does_not_start_two_loads(self, voices):
        engine = PiperEngine(voices)
        engine.warm()
        engine.warm()
        assert engine._warming is True, "the second warm should be ignored"

    def test_warm_does_not_raise_on_a_broken_voice(self, voices):
        # The fake file is not a model. A background load that raised would take
        # the process down with a thread exception nobody sees.
        engine = PiperEngine(voices)
        engine.warm()
        for _ in range(100):
            if not engine._warming:
                break
            time.sleep(0.02)
        assert engine.load_error

    def test_synthesis_still_returns_none_for_a_broken_voice(self, voices):
        engine = PiperEngine(voices)
        engine.warm()
        for _ in range(100):
            if not engine._warming:
                break
            time.sleep(0.02)
        assert engine.synthesize("hello") is None


class TestServiceWarmsOnSelection:
    def test_set_engine_warms_the_new_engine(self):
        from signbridge.speech.service import SpeechService

        class Fake:
            name = "fake"

            def __init__(self):
                self.warmed = False

            def close(self):
                pass

            def warm(self):
                self.warmed = True

            def describe(self):
                return "fake"

        service = SpeechService()
        engine = Fake()
        service.set_engine(engine)
        assert engine.warmed, "a voice left unwarmed pays the load on the first sign"

    def test_an_engine_without_warm_is_tolerated(self):
        from signbridge.speech.service import SpeechService

        class Bare:
            name = "bare"

            def close(self):
                pass

            def describe(self):
                return "bare"

        SpeechService().set_engine(Bare())  # must not raise