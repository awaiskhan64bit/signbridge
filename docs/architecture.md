# Architecture

Why the pipeline is shaped the way it is. Each section covers one decision, what
the alternatives were, and what it costs.

## Pipeline

```
CaptureWorker ──► HandTracker ──► encode_frame ──► SignDecoder ──► SpeechService
   (thread)         (MediaPipe)      (features)      (events)        (Piper)
       │                 │               │              │
       └────────────► SignEngine ──────► Recognizer ───┘
                       (orchestrates)     (rules or ONNX)
```

`SignEngine` owns one thread and one of everything else. The Qt front end only
ever receives callbacks, which is why the whole pipeline runs headless in the
tests and can be driven by a synthetic signer.

Nothing in `core/`, `vision/`, `ml/` or `speech/` imports Qt. The dependency
runs one way only, from `ui/` inward.

## 1. Hand landmarks, not video

The single most consequential decision.

Classifying video means running a CNN over frames, which means a model in the
hundreds of megabytes, a GPU or a very fast CPU, and a training pipeline built
around video decoding. Classifying **hand landmarks** means MediaPipe hands,
which is a 7.5 MB model running in about 10 ms per frame on CPU, and a training
pipeline built around 21 numbers per frame.

That trade is only worth making because the thing being classified is a hand
shape, not a scene. For sign language it is, so the trade is strongly in favour.

## 2. The feature encoder

Raw landmarks are a bad model input. They encode where the signer is standing,
how far away they are, and how the camera is angled, none of which has anything
to do with the sign. `ml/features.py` removes all three:

1. **Translate** the wrist to the origin.
2. **Rotate** into a palm anchored frame: up runs wrist to middle knuckle, and
   the palm normal decides which perpendicular is right.
3. **Scale** so the wrist to middle knuckle distance is one.
4. **Mirror** the left hand, so both hands encode identically.

Anchoring on the palm rather than on the index knuckle matters. Aligning to a
single point leaves the frame under-determined once the hand is mirrored, and
left and right hands land 180 degrees apart. Anchoring on the palm and using its
normal to fix the sign of the perpendicular resolves it, which is what lets the
model treat the two hands as one class of shapes.

On top of the 63 normalised coordinates sit 17 derived values: a curl and a
reach per finger, the palm normal, and adjacent fingertip gaps. Curl is the mean
bend at each joint mapped so straight is 0 and folded is 1. Reach separates a
curled finger from one pointing away from the camera, which curl alone cannot
distinguish.

The cost of rotation invariance is that a thumbs up and a thumbs down become
identical features. That is a real loss, and it is why `thumbs_down` is absent
from the built in gesture vocabulary. Documented in `ml/gestures.py` rather than
quietly worked around.

## 3. Two recognisers behind one interface

`Recognizer.predict(window) -> SignPrediction`. Two implementations:

- **`ml/gestures.py`**, rule based. Six static gestures from mean curl and
  thumb reach. Not machine learning, and does not pretend to be. It exists so
  the app is useful before anything is trained, and as a control condition when
  evaluating the network.
- **`ml/neural.py`**, ONNX. The trained classifier. Inference through
  onnxruntime so the app does not carry PyTorch.

The thresholds in the rule based recogniser were measured against the hand rig
rather than guessed. That rig reports curls from 0.0 extended to about 0.47
fist, and the thresholds sit at 0.15 and 0.33 accordingly. Guesswork here
produces a recogniser that confidently reports the same gesture forever, which
is exactly what happened first.

## 4. Temporal decoding

`core/decoder.py`. A classifier re-reads the same held sign on every window, so
the naive loop says HELLO forty times a second. Three mechanisms prevent that:

- **Voting.** A sign is confirmed only after winning `votes_required` consecutive
  windows, which rejects single frame noise.
- **Latching.** Once confirmed, the same label will not confirm again until a
  different one appears. One physical sign, one event.
- **Unlatching.** When the label changes, the latch releases. This is what lets
  the same word be signed twice in a row, provided the hand moves in between.

Words are assembled into phrases that are spoken on a pause, or when the phrase
hits a length limit. People do not announce word boundaries, so time is the only
signal available.

## 5. Confidence as a first class output

A two thousand class classifier on six examples per class will be wrong often.
Pretending otherwise would produce an app that confidently speaks nonsense, which
for an accessibility tool is worse than silence.

So the training script does not stop at accuracy. It sweeps the confidence
threshold on the validation set, finds the cut that maximises accuracy **among
accepted** predictions while still covering a useful share of inputs, and stores
it in the model metadata. The app reads that number as its default.

This is the difference between a demo that sounds impressive and one that
degrades gracefully. The tradeoff curve is in [results.md](results.md).

## 6. Inference stack

| Choice | Alternative | Why |
| --- | --- | --- |
| PySide6 / Qt Quick | Electron, Tkinter | Qt Quick renders at 60 fps on the GPU, so a 30 fps camera feed costs almost nothing to display. No Node toolchain needed. |
| ONNX Runtime | PyTorch, TensorRT | A few MB of weights, no framework at runtime, CPU inference well under a millisecond. TensorRT needs a GPU. |
| Piper | Coqui, Kokoro | Piper ships one abi3 wheel with espeak-ng bundled and no torch. Coqui does not support Python 3.12. Kokoro drags in about 2 GB of dependencies. |

MediaPipe on Windows runs CPU only; GPU support was removed upstream. For this
workload that is not a loss, since the heavy lifting is in an 80 dimensional
feature vector rather than in pixels.

## 7. The simulated signer

`vision/simulated.py` builds 21 point hand rigs from a handful of pose
parameters: curl per finger, thumb opposition, thumb extension, spread, wrist
roll. It exists because a camera is not always available, and because a
deterministic known ground truth is the cheapest way to test a recogniser.

It plugs into the capture interface exactly as a camera does, and the skeleton
is drawn by the same overlay code. Demo mode is the real pipeline with a
different frame source, not a mock.

It also found a genuine bug: the rule based recogniser was averaging over the
whole window and therefore reading the approach to a sign rather than the sign.
Fixing that took weighting toward the most recent frames, which is correct for
real input too. Against the rig's known labels the gesture recogniser scores
100% on held poses.

## 8. Threading

| Thread | Owns |
| --- | --- |
| Capture | The OpenCV device, pushing frames into a bounded queue |
| Engine | Tracking, features, recognition, decoding |
| Speech | Piper synthesis and playback |
| Qt main | QML only |

Frames cross into the GUI thread through a `QQuickImageProvider`, so a QPixmap
is built on the engine thread and the UI only ever touches a finished image. The
alternative, assigning a QImage to `Image.source`, relies on QImage to url
coercion, which is not dependable across Qt versions.

The pipeline catches exceptions per frame. One bad frame must not end the
session.
