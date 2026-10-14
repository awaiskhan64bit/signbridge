# SignBridge
(took hours to train on an RTX 5060 the computer vision model is made from scratch my gpu has suffered)

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

### One click

Clone the repo, double click **`quickstart.bat`**, and it does the rest:

- installs **Python 3.12** if the machine has none, via `winget`
- creates a virtual environment and installs every dependency
- downloads the models and the voice
- puts a **SignBridge shortcut on your Desktop**
- starts the app

Run it again after an update and it skips whatever is already done. Needs internet
for the setup itself; after that the app makes no network calls at all.

| Flag | Effect |
| --- | --- |
| `--no-python` | Never install Python. Use what is already there, or fail. |
| `--shortcut-only` | Skip straight to creating the Desktop shortcut. |

Python is only installed if it is genuinely missing, and the script says so on
screen while it happens. If `winget` is unavailable it stops and prints the manual
link rather than trying anything else.

### By hand

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

### Building a standalone executable

`signbridge.spec` is a PyInstaller build that produces `dist/SignBridge/SignBridge.exe`
with no Python required on the target machine.

```bash
pip install pyinstaller
pyinstaller --noconfirm signbridge.spec
```

**This build is unverified.** PyInstaller is not a project dependency and has not
been run against this codebase, so treat the spec as a starting point rather than
a known good recipe. The two things most likely to need fixing are the QML files,
which are force-included because they are loaded from disk at runtime, and
MediaPipe's lazily imported modules.

It builds a folder rather than a single exe on purpose. PySide6, MediaPipe and
ONNX Runtime together are over a gigabyte of shared libraries, and a one file
build unpacks all of that into a temp directory on every launch. Zip the folder
instead.

The models and voice are not bundled. They are downloaded once into the user's own
app data, so assets can be updated without rebuilding the executable.

## Using it

| Action | How |
| --- | --- |
## Known limits

Worth stating plainly rather than discovering them later.

**Word recognition is weak.** On held out clips the classifier is right about 7%
of the time at 200 words, and about 16% at 50. That is far below published WLASL
results, and the reason is data rather than architecture: the corpus gives about
twelve training clips per word. Signs performed by different people vary more than
that supports. Fewer words is genuinely more accurate, so the vocabulary setting
is worth lowering if accuracy matters more than range.

**The confidence cut is calibrated for that weak model**, so it accepts roughly a
quarter of clips at about 29% precision. It speaks wrong more often than right by
design, because silence was measured to be less useful. Every prediction shows
its own confidence so a wrong answer looks wrong.

**Gestures work on the camera. Words are the weak part.** The gesture recogniser
decides from finger curl and thumb reach, which survive the difference between a
real hand and a synthetic one; the earlier template matcher scored exactly zero on
a real clip.

**The gesture book is six**, and that is a deliberate limit. It is small because
it is a rule set, and it can be extended, but each addition needs a finger pattern
and thumb state that no existing gesture already uses.
| Speak the current phrase | `Ctrl+Enter` |
| Repeat the last phrase | `Ctrl+R` |
| Change input source | Camera / Clip / Demo, top right of the viewport |
| Get back to the hand rig | **Back to demo**, or the Demo segment |
| Record your own signs | **Teach it your signs** in the right rail |
| Tune sensitivity | The threshold slider |

Settings persist to `%LOCALAPPDATA%\SignBridge\settings.json`.

### When it is quiet

The classifier refuses to speak unless it is confident. On WLASL footage it
accepts about 37% of clips at 22% precision, and on footage from a different
domain it accepts almost nothing. That is deliberate: an accessibility tool that
confidently says the wrong thing is worse than one that stays silent.

Lower the threshold to trade silence for more guesses. See
[Making it work on your webcam](#making-it-work-on-your-webcam) for the better
fix.

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

## Making it work on your webcam

This is the part that matters, and it is not a modelling problem.

WLASL is tightly cropped, isolated signs, recorded on other people's phones. A
webcam is none of those things. The shipped model is accurate on WLASL clips and
quietly silent on real video, which is the honest result of that mismatch rather
than a bug. Measured on an ASL tutorial video, its confidence peaks at 0.12
against a threshold of 0.29.

The fix is to train on your own signing, which the app can record for you:

1. Open **Teach it your signs**, type a word, press **Record**
2. Perform the sign, press **Stop**
3. Repeat a few times per word, across as many words as you care about
4. Run `scripts\finetune.bat`

A few dozen clips takes a couple of minutes to record and a couple of minutes to
train. It works far better than any amount of extra augmentation, because it is
the only data that matches how you actually sign.

Add `--with-wlasl` if you want to keep the full two thousand word vocabulary
rather than leaning towards the words you recorded.

Two knobs help in the meantime. The **threshold** slider goes well below the
model's own calibrated cut, which is what you want for out-of-domain input. And
the **Vocabulary** selector switches between the WLASL model and the six
built-in gestures, which is what makes demo mode work at all, since
`open_palm` and `victory` are not WLASL words.

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
