# Licensing and attribution

This project is MIT. Its dependencies are not, and two of them matter if a build
is ever distributed rather than run locally. Read this before shipping anything.

## Summary

| Component | Licence | Commercial use | Action needed |
| --- | --- | --- | --- |
| SignBridge itself | MIT | yes | none |
| MediaPipe | Apache-2.0 | yes | keep the notice |
| WLASL | academic use | **restricted** | do not redistribute the dataset |
| Piper | **GPL-3.0** | yes, with obligations | see below |
| Piper voice `lessac` | Blizzard 2013 corpus | **restricted** | see below |
| Piper voices `amy`, `kathleen` | LJSpeech derived, permissive | yes | easiest option |
| Piper voices `ryan`, `hfc_female` | CC BY-NC-SA 4.0 | **no** | do not ship |

## Piper is GPL-3.0

`piper-tts` changed licence at version 1.3, when development moved to
OHF-Voice. Versions 1.3 and later are GPL-3.0-or-later, not the MIT of the
older rhasspy line.

GPL obligations attach when you **distribute** Piper, including inside a
standalone desktop build. Running it on your own machine is unaffected. If a
build ever leaves your hands, either publish the whole project under GPL-3.0, or
switch to Kokoro, whose weights are Apache-2.0.

Piper is an optional dependency (`pip install -e ".[tts]"`). The app falls back
to Windows SAPI without it.

## The default voice is not the safest one

`en_US-lessac-medium` is the default because it is the best sounding voice that
is not non-commercial. Its model card is the **Blizzard 2013 corpus licence**,
which permits use and redistribution but is not a standard open licence.

For a build you intend to distribute, `en_US-amy-low` or `en_US-kathleen-low` are
the clean options. Both derive from LJSpeech, which is public domain. Set the
voice with:

```bash
python -m signbridge.setup --voices --voice en_US-amy-low
```

Avoid `ryan-high` and `hfc_female-medium`. Both are CC BY-NC-SA 4.0, which
forbids commercial use outright.

## WLASL

The Word Level American Sign Language dataset is academic use only, under the
WLASL agreement. The repository is public and the landmark mirror on Hugging
Face carries no licence at all, which is arguably worse: no licence means no
permission.

Training on it for a coursework project is fine. Do not redistribute the dataset
or a model trained from it in a commercial product, and resolve the licence with
the dataset authors first if that ever becomes the plan.

`scripts/fetch_wlasl.py` records the provenance of both sources in the index it
writes, so the origin is never lost.

## MediaPipe

Apache-2.0, including the `.task` model bundles. Keep the notice alongside any
redistribution.

## What this project adds

Everything under `src/signbridge/` is original work for this project and is MIT
licensed. The feature encoder, the temporal decoder, the simulated hand rig, the
pipeline and the interface were all written for it. Nothing was copied from
another sign language project.
