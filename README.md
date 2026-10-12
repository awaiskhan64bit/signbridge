# SignBridge

An offline desktop app that watches your webcam, reads your signing, and says it
out loud.

Everything runs on your machine. After the one time setup there is no network
call anywhere in the app: no API key, no account, no inference in the cloud.

---

## What it actually does

```
webcam frame
   -> MediaPipe hand landmarks        21 points per hand, 30 fps
   -> feature encoder                 position, rotation, scale and handedness invariant
   -> sign classifier                 BiLSTM over a 64 frame window, ONNX on CPU
   -> temporal decoder                one sign = one event, phrases form on a pause
   -> Piper neural TTS                offline speech synthesis
```

The middle of that chain is the interesting part, and
[docs/architecture.md](docs/architecture.md) explains the reasoning behind each
choice.

## Try it without a camera

There is a built in simulated signer that performs a phrasebook of gestures
through an anatomically correct hand rig. It drives the real pipeline, not a
mock, so you can see the whole thing working on a machine with no webcam:

```bash
python -m signbridge --source simulated
```

## Install

Requires Python 3.10 to 3.12 on Windows.

```bash
git clone <this repo>
cd signbridge

python -m venv .venv
.venv\Scripts\activate

pip install -e ".[tts]"
python -m signbridge.setup          # models + voice, one time, ~70 MB
python -m signbridge                # start the app
```

`pip install -e .` on its own is enough to run the app: speech falls back to
Windows SAPI and recognition falls back to a small built in gesture vocabulary.
The `[tts]` extra adds Piper, which is what makes the voice sound like a person.

## Using it

| Action | How |
| --- | --- |
| Start and stop | `Space`, or the button top right |
| Speak the current phrase | `Ctrl+Enter` |
| Repeat the last phrase | `Ctrl+R` |
| Change input source | Camera / Clip / Demo, top right of the viewport |
| Tune sensitivity | The threshold slider, which is the model's own calibrated cut |

Settings persist to `%LOCALAPPDATA%\SignBridge\settings.json`.

### When it is quiet

The classifier will refuse to speak unless it is confident. That is deliberate.
WLASL is a hard dataset and this model is honest about its limits, so it stays
silent rather than inventing words. The threshold slider trades silence for
guessing; see the accuracy section for what each setting is worth.

## The model

`sign_classifier.onnx` is trained by `scripts/train.py` on WLASL, the standard
word level sign language benchmark, using the official train/val/test splits.
A two layer BiLSTM with attention pooling over hand landmarks, exported to ONNX
so inference needs no deep learning framework at runtime.

The shipped model covers all **1,992 WLASL words**, scores 13.4% top-1 and 33.9%
top-5 on the held out test split, and runs a prediction in 1.5 ms on a laptop CPU.
At that vocabulary size, with a median of six training clips per word, those are
the honest numbers. The
[precision when it chooses to speak](docs/results.md) table is the one that
describes what the app actually does: it accepts about a third of what it sees
and is right 22% of the time. It degrades into silence, not into nonsense.

```bash
python scripts/fetch_wlasl.py       # ~430 MB of landmarks, once
python scripts/train.py             # CPU, no GPU needed
```

Results from the run that shipped in this repo are in
[docs/results.md](docs/results.md), along with what they do and do not mean.

## Project layout

```
src/signbridge/
  core/        pipeline, decoder, shared types
  vision/      capture, hand tracking, overlay, simulated signer
  ml/          feature encoder, recognisers, dataset loader
  speech/      Piper, SAPI, playback
  ui/          Qt Quick front end
scripts/       dataset fetch, training
tests/         feature and decoder invariants
docs/          architecture, results, licensing
```

## Pushing a fork

If you cloned rather than being the original author, or you want to attach a
remote to an existing checkout:

```bash
git remote add origin https://github.com/awaiskhan64bit/signbridge.git
git push -u origin main
```

Git will prompt for a Personal Access Token (Settings, Developer settings,
Personal access tokens, Tokens classic, scope `repo`) instead of a password.

## Tests

```bash
pytest
```

The feature and decoder tests are the ones that matter. They pin the invariants
that break silently: if normalisation drifts, training data and live input stop
agreeing and accuracy collapses without any error being raised.

## Licences and attribution

This project is MIT. Its dependencies are not all, and in two cases that matters
if you ever distribute a build. [docs/licensing.md](docs/licensing.md) covers
Piper (GPL-3.0), the Piper voices, MediaPipe (Apache-2.0) and WLASL.

## Limitations

Stated plainly, because a sign language tool that overstates what it does is
worse than no tool:

- **Accuracy is bounded by the data, not the code.** WLASL has around six clips
  per word. Words with nine or more training clips score 30.6% top-1; words with
  three or fewer score 9.2%. That gap is the whole story.
- **Two handed signs are hard.** The encoder uses the dominant hand, so signs
  that depend on both hands together lose information.
- **Trained on video, run on a webcam.** Every landmark corpus available is phone
  selfie footage at about 10 fps in portrait. A desktop webcam is landscape, at
  30 fps. Expect a real penalty, and fine tune on your own footage if you need
  the last stretch.
- **Not a replacement for a human interpreter.** It is a demonstration of what
  an offline sign language interface can look like.

## Licence

MIT, see [LICENSE](LICENSE).
