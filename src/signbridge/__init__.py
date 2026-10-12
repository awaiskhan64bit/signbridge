"""SignBridge: offline camera sign language to speech.

The package is split into layers so each one can be swapped or tested alone:

    vision   camera capture and hand landmark extraction (MediaPipe)
    ml       feature encoding, the trained classifier, sign vocabulary
    speech   text to speech engines (Piper neural, SAPI fallback)
    core     the realtime pipeline that glues the layers together
    ui       PySide6 / Qt Quick front end
"""

__version__ = "0.1.0"
APP_NAME = "SignBridge"

__all__ = ["APP_NAME", "__version__"]
