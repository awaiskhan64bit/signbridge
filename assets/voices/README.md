# Bundled voices

Piper voice models (`*.onnx` + `*.onnx.json`) are downloaded once by
`python -m signbridge.setup` and then loaded locally. No network access at runtime.

Default voice: `en_US-hfc_female-medium` (natural, small, fast enough for realtime).
Alternatives are listed in `scripts/fetch_voices.py --list`.
