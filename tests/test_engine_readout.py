"""Tests for the engine readout shown in the UI.

The ENGINE and VOCAB chips describe which recogniser is actually loaded. They
used to compare the user's preference, which defaults to "auto", against the
string "words", so the chips reported gestures while the trained model was
running and hid the model's accuracy figure. These pin the behaviour down.
"""

from __future__ import annotations

import pytest

from signbridge.config import Settings
from signbridge.core.types import Source
from signbridge.ui.controller import Controller


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    instance = QApplication.instance() or QApplication(["engine-readout"])
    yield instance


def _controller(app, source: str) -> Controller:
    settings = Settings()
    settings.source = source
    settings.recognizer = "auto"
    settings.tts_engine = "off"
    return Controller(settings)


def test_auto_resolves_to_words_for_a_camera(app):
    controller = _controller(app, Source.CAMERA.value)
    assert controller.activeRecognizerKind == "words"


def test_auto_resolves_to_words_for_a_clip(app):
    """A video file is real signing, so it belongs on the trained model."""

    controller = _controller(app, Source.VIDEO.value)
    assert controller.activeRecognizerKind == "words"


def test_auto_resolves_to_gestures_for_the_demo(app):
    """The synthetic hand rig is not WLASL, so pointing the model at it is pointless."""

    controller = _controller(app, Source.SIMULATED.value)
    assert controller.activeRecognizerKind == "gestures"


def test_preference_and_active_engine_are_distinguishable(app):
    """`recognizerKind` is the preference and stays "auto"; the chip reads the other.

    Conflating the two is what made the chip lie.
    """

    controller = _controller(app, Source.VIDEO.value)
    assert controller.recognizerKind == "auto"
    assert controller.activeRecognizerKind != controller.recognizerKind


def test_vocabulary_label_matches_the_loaded_model(app):
    """The word count must come from the model, not a literal in the QML."""

    controller = _controller(app, Source.VIDEO.value)
    expected = len(controller._recognizer.labels)
    assert controller.activeVocabularyLabel == f"{expected:,} words"
    assert expected > 0