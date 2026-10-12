# Bundled models

Everything in this folder is fetched by `python -m signbridge.setup` on first run and is
git-ignored on purpose, because each file is large and licensed separately.

| File                | Source                                              | Purpose                          |
| ------------------- | --------------------------------------------------- | -------------------------------- |
| `*.task`            | Google MediaPipe Solutions hand/gesture bundle      | Hand landmark extraction         |
| `sign_classifier.*` | Trained locally by `scripts/train.py`               | Sign sequence classification     |

Nothing here is downloaded at runtime. Once `setup` finishes, SignBridge is fully offline.
