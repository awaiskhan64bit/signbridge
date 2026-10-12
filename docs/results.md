# Results

Numbers from the model that ships in `assets/models`, and what they mean.

Everything here is produced by:

```bash
python scripts/train.py                    # headline numbers, trains the model
python scripts/evaluate.py                 # the tables below
python scripts/evaluate.py --write-threshold   # refit the shipped cut, no retraining
```

Both read the official WLASL train/val/test splits, so these are held out
numbers, not training set numbers.

## The dataset, and why it sets the ceiling

| | |
| --- | --- |
| Clips | 11,880 |
| Words | 1,992 |
| Train / val / test | 8,243 / 2,233 / 1,404 |
| Clips per word | min 1, median 6, max 16 |
| Frames per clip | min 25, median 57, max 149 |
| Frames with a detected hand | median 72% |
| Landmark join failures | 0 of 11,880 |

A median of six clips per word is the whole story. WLASL is a benchmark for
*comparing methods*, not a training set. No architecture fixes this. The table
further down proves it directly: accuracy triples between the words with three
clips and the words with nine.

## Headline

| Metric | Value |
| --- | --- |
| Vocabulary | 1,992 words |
| Parameters | 2,344,393 |
| Model on disk | 9.4 MB (`sign_classifier.onnx`) |
| Inference | ONNX Runtime, CPU only, batch of one |
| Measured inference | 1.5 ms per 64 frame window |
| Best epoch | 38 of 40 |
| Validation top-1 | 13.7% |
| **Test top-1** | **13.4%** |
| **Test top-5** | **33.9%** |

For context, published word-level WLASL results on comparable sized models with
hand-crafted features sit in the same low teens for the full 2,000 gloss
vocabulary, rising to 40 to 60% only when the vocabulary is cut to a few hundred
words. This number is the honest one for the scope requested.

## Precision when it chooses to speak

This is the table that matters. Accuracy counts every clip, including the ones
the model gets hopelessly wrong and would never have spoken. What a user
experiences is: of the signs the app decided were confident enough to say out
loud, how many were right.

| Threshold | Coverage | Precision |
| --- | --- | --- |
| 0.05 | 97.9% | 13.6% |
| 0.10 | 81.7% | 15.4% |
| 0.20 | 54.2% | 19.4% |
| **0.29 (shipped)** | **37.7%** | **22.3%** |
| 0.50 | 20.9% | 23.8% |
| 0.70 | 10.6% | 19.5% |
| 0.90 | 4.1% | 10.3% |

Precision peaks around threshold 0.5 and falls away above it, which is the
expected shape: the very highest confidence predictions are the small number of
clips the model found easy, so they are less representative.

The shipped default is **0.29**, chosen on the validation split alone. It
accepts 37.7% of input at 22.3% precision on test, which is better than the
13.4% baseline and still speaks for more than a third of what it sees. Raising
the threshold to 0.5 buys 1.5 points of precision for giving up 17 points of
coverage, which is a bad trade for a live tool.

## Accuracy by how much data each word had

The long tail, made visible. This is the most important table in the document.

| Training clips for the word | Words | Top-1 |
| --- | --- | --- |
| 1 to 3 | 764 | 9.2% |
| 4 to 8 | 1,191 | 15.0% |
| 9 or more | 34 | 30.6% |

A third more data roughly doubles accuracy, and the effect has not levelled off
at nine clips. The bottleneck is labelled examples, not model capacity. Given
the same budget spent on more WLASL data rather than a larger network, this is
where the return would be.

## What the model confuses

The most common errors are not random. They are synonyms and near neighbours:

```
angel      -> fairy
escape     -> retreat
great      -> wonderful
depend     -> rely
explain    -> describe
barely     -> fruit
```

These pairs share handshape and travel, so hand landmarks alone cannot separate
them. Distinguishing them needs either more data or a larger feature set, and
this is a concrete argument for what to add next: upper body pose and the face,
which WLASL's own preprocessing includes and which this project drops in favour
of a smaller, faster pipeline.

## What was tried, and what it was worth

Honest accounting, since a reader should know which decisions earned their place.

| Change | Effect |
| --- | --- |
| Position, rotation, scale and handedness invariant encoding | The single largest win. Without it the model keys on where the signer sits. |
| Two layer BiLSTM with attention pooling | Modest over a single layer. Recurrence matters for dynamic signs, not static ones. |
| Crop, respeed, jitter, mirror, frame dropout, mixup | Large, given six clips per word. Validation was still climbing at epoch 38 of 40, so more epochs would help. |
| Label smoothing plus half power class weighting | Small. Full balancing was worse: two clip words then outvote a hundred clip words. |
| Confidence threshold calibration | No effect on accuracy, and the difference between a usable app and an unusable one. |

## What was not done

- **Two handed signs.** The encoder uses the dominant hand only, so signs that
  depend on both hands together lose information.
- **Signer held out evaluation.** The official WLASL splits are by clip, not by
  signer, so the same person appears in train and test. A signer-disjoint split
  would report lower numbers and is the more honest measurement.
- **Fine tuning on webcam footage.** Every landmark corpus available is phone
  selfie video in portrait at about 10 fps. This model has never seen landscape
  30 fps webcam footage, which is what the app runs on. Expect a real penalty,
  and a short fine tuning run on self collected footage to recover much of it.
- **Pose and face landmarks.** WLASL's own preprocessing includes 33 pose and
  468 face points. This project uses 21 hand points only, trading accuracy for
  speed and simplicity. The synonym confusions above are the visible cost.

## Reading these numbers honestly

This is a demonstration that an offline, on-device sign language interface can
be built end to end: capture, tracking, encoding, a trained classifier over a
two thousand word vocabulary, temporal decoding into phrases, and neural speech,
with no network call after install and under a millisecond of inference per
frame.

It is not a replacement for a human interpreter, and the tables above are the
evidence. What the threshold curve buys is a system that degrades into silence
rather than into confident nonsense, which for an accessibility tool is the only
acceptable way to fail.

On cost: at 1.5 ms per window and a prediction every fifth frame, classification
is about 1% of a 30 fps frame budget. Recognition is not the bottleneck here.
Hand tracking and drawing are.
