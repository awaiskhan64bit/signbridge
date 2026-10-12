"""Application entry point.

Wires the Qt event loop, the QML engine, the frame provider and the sign
pipeline together, then hands control to Qt.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QPalette
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication

from .. import APP_NAME
from ..config import Settings
from .controller import Controller

#: The QML module lives in its own directory so `import SignBridge` resolves,
#: with this directory on the import path.
QML_ROOT = Path(__file__).resolve().parent / "qml"
QML_MODULE = QML_ROOT / "SignBridge"


def _dark_palette() -> QPalette:
    """Native Controls (scrollbars, menus) default to light on Windows."""

    p = QPalette()
    p.setColor(QPalette.Window, QColor("#0B0D17"))
    p.setColor(QPalette.WindowText, QColor("#E9EDFB"))
    p.setColor(QPalette.Base, QColor("#111423"))
    p.setColor(QPalette.AlternateBase, QColor("#161A2B"))
    p.setColor(QPalette.Text, QColor("#E9EDFB"))
    p.setColor(QPalette.Button, QColor("#161A2B"))
    p.setColor(QPalette.ButtonText, QColor("#E9EDFB"))
    p.setColor(QPalette.Highlight, QColor("#5AE0FF"))
    p.setColor(QPalette.HighlightedText, QColor("#06070D"))
    p.setColor(QPalette.ToolTipBase, QColor("#161A2B"))
    p.setColor(QPalette.ToolTipText, QColor("#E9EDFB"))
    p.setColor(QPalette.PlaceholderText, QColor("#5D6785"))
    return p


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("signbridge")

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    QQuickStyle.setStyle("Basic")

    settings = Settings.load()

    # The camera cannot be opened from the GUI thread before Qt is up, so the
    # engine is constructed here but started once the window exists.
    controller = Controller(settings)

    engine = QQmlApplicationEngine()
    engine.addImportPath(str(QML_ROOT))
    engine.addImageProvider("frames", controller.provider)
    engine.rootContext().setContextProperty("app", controller)

    engine.load(QUrl.fromLocalFile(str(QML_MODULE / "main.qml")))
    if not engine.rootObjects():
        log.error("QML failed to load. Errors above point at the cause.")
        return 1

    log.info("Starting pipeline, source=%s", settings.source)
    controller.engine.start()

    try:
        return app.exec()
    finally:
        controller.engine.stop()
        settings.save()


def main() -> None:
    sys.exit(run())
