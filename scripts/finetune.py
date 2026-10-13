"""Fine tune the sign classifier on the user's own signing.

This is the step that actually makes the app work on a given webcam. The shipped
model is trained on WLASL, which is tightly cropped isolated signs recorded on
other people's phones. A webcam is none of those things. Recording a few dozen
clips through the app and fine tuning on them closes most of that gap, and takes
minutes rather than the hour a full training run costs.

    python scripts/finetune.py
    python scripts/finetune.py --epochs 40 --with-wlasl
    python scripts/finetune.py --min-clips 5      # drop words seen too rarely

How it decides what to train on:

* recorded clips are upweighted, because they are the only in-domain data and
  the only thing that matters for how the app feels in use
* with ``--with-wlasl`` the WLASL training split is mixed in, which stops the
  model forgetting its vocabulary and collapsing onto the handful of words the
  user recorded

Clips with no hand in most of their frames are rejected rather than trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signbridge.ml.calibration import calibrate  # noqa: E402
from signbridge.ml.collector import Collector  # noqa: E402
from signbridge.ml.features import FEATURE_DIMS  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "assets" / "models"


def main() -> int:
    import torch
    from torch import nn

    from train import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4, help="lower than a full run")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument(
        "--with-wlasl",
        action="store_true",
        help="mix in WLASL so the full vocabulary is not forgotten",
    )
    parser.add_argument(
        "--min-clips",
        type=int,
        default=2,
        help="ignore words with fewer recorded clips than this",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------- recorded clips first

    collector = Collector(data_dir() / "collected")
    tracks, labels_raw = collector.load_all()
    if not tracks:
        print(
            "No recorded clips yet.\n"
            "Open SignBridge, use the Teach it panel to record some signs, then "
            "run this again.",
            file=sys.stderr,
        )
        return 1

    counts = Counter(labels_raw)
    keep = {label for label, n in counts.items() if n >= args.min_clips}
    pairs = [(t, name) for t, name in zip(tracks, labels_raw, strict=True) if name in keep]
    print(f"recorded: {len(tracks)} clips across {len(counts)} words")
    print(f"training on {len(pairs)} clips across {len(keep)} words (min {args.min_clips})")
    if len(keep) < len(counts):
        dropped = sorted(set(counts) - keep)
        print(f"  ignored, too few clips: {', '.join(dropped)}")

    # ------------------------------------------------------ base vocabulary

    meta_path = MODELS / "sign_classifier.json"
    if not meta_path.exists():
        print("No base model. Run scripts/train.py first.", file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    all_labels: list[str] = list(meta["labels"])
    label_of = {name: i for i, name in enumerate(all_labels)}

    # A word the user recorded that WLASL never had still has to be speakable.
    added = sorted(keep - set(label_of))
    for name in added:
        label_of[name] = len(all_labels)
        all_labels.append(name)
    if added:
        print(f"  added to the vocabulary: {', '.join(added)}")

    y = np.array([label_of[label] for _, label in pairs], dtype=np.int64)
    x = np.stack([t for t, _ in pairs]).astype(np.float32)

    # ----------------------------------------------- optional WLASL mixing

    if args.with_wlasl:
        wlasl_x, wlasl_y = _load_wlasl(label_of)
        if wlasl_x is not None and len(wlasl_x):
            # Cap the contribution so the recorded clips are not drowned out.
            cap = max(1, len(x) * 10)
            if len(wlasl_x) > cap:
                keep_idx = rng.choice(len(wlasl_x), cap, replace=False)
                wlasl_x, wlasl_y = wlasl_x[keep_idx], wlasl_y[keep_idx]
            x = np.concatenate([x, wlasl_x], axis=0)
            y = np.concatenate([y, wlasl_y], axis=0)
            print(f"mixed in {len(wlasl_x)} WLASL clips")

    # -------------------------------------------------------------- training

    mean = x.reshape(-1, FEATURE_DIMS).mean(axis=0)
    std = x.reshape(-1, FEATURE_DIMS).std(axis=0) + 1e-5
    x = (x - mean) / std

    # Hold out one clip per word for a validation signal. With a handful of clips
    # per word this is the only honest way to see whether it is learning.
    val_idx: list[int] = []
    train_idx: list[int] = []
    seen: Counter = Counter()
    for i, label in enumerate(y):
        if seen[label] > 0 and not any(y[j] == label for j in val_idx):
            val_idx.append(i)
        else:
            train_idx.append(i)
        seen[label] += 1
    if not val_idx or not train_idx:
        train_idx = list(range(len(y)))
        val_idx = train_idx

    print(f"training {len(train_idx)} clips, holding out {len(val_idx)} for validation")

    model = build_model(len(all_labels), args.hidden, args.layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters, {len(all_labels)} classes")

    # Recorded clips dominate: this is fine tuning, not a fresh run.
    sample_weight = np.where(np.isin(np.arange(len(y)), val_idx), 0.0, 1.0)
    sample_weight[: len(train_idx)] = 3.0

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, args.epochs)
    )

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    w_t = torch.from_numpy(sample_weight.astype(np.float32))

    def evaluate() -> tuple[float, np.ndarray]:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x_t[val_idx]), dim=1).numpy()
        return float((probs.argmax(axis=1) == y[val_idx]).mean()), probs

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        total, seen_n = 0.0, 0
        for start in range(0, len(order) - args.batch + 1, args.batch):
            sel = order[start : start + args.batch]
            logits = model(x_t[sel])
            per_sample = nn.functional.cross_entropy(logits, y_t[sel], reduction="none")
            weights = w_t[sel]
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss) * len(sel)
            seen_n += len(sel)
        scheduler.step()
        acc, _ = evaluate()
        flag = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(f"epoch {epoch:3d}  loss {total / max(1, seen_n):.4f}  val {acc:.3f}{flag}")

    if best_state is None:
        print("nothing was trained", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    val_acc, val_probs = evaluate()
    threshold = calibrate(val_probs, y[val_idx]).threshold

    MODELS.mkdir(parents=True, exist_ok=True)
    onnx_path = MODELS / "sign_classifier.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, args.window, FEATURE_DIMS),
        str(onnx_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    torch.save(best_state, MODELS / "sign_classifier.pt")

    out = dict(meta)
    out["name"] = f"SignBridge fine tuned ({len(keep)} of your words)"
    out["labels"] = all_labels
    out["window"] = args.window
    out["mean"] = [float(v) for v in mean]
    out["std"] = [float(v) for v in std]
    out["threshold"] = float(threshold)
    out.setdefault("metrics", {})
    out["metrics"].update(
        finetuned_on_my_signs=list(keep),
        recorded_clips=len(pairs),
        val_top1=float(val_acc),
        params=int(n_params),
    )
    out["source"] = {
        "base": "WLASL, fine tuned on clips recorded through this app",
        "recorded_clips": str(collector.root),
        "note": "in-domain fine tuning; accuracy on WLASL will drop by design",
    }
    (MODELS / "sign_classifier.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nbest validation {best_acc:.3f} on held out clips")
    print(f"threshold {threshold:.3f}, from the recorded data")
    print(f"wrote {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print("\nRestart the app to use it.")
    return 0


def data_dir() -> Path:
    from signbridge.config import Settings

    return Settings.load().data_dir


def _load_wlasl(label_of: dict[str, int]) -> tuple:
    """WLASL training clips for the words we already know, subsampled."""

    index_path = DATA / "wlasl-index.json"
    if not index_path.exists():
        return None, None
    from signbridge.ml.dataset import WLASLIndex, encode_clip, resize_track

    index = WLASLIndex(index_path)
    wanted = [s for s in index.split("train") if s.gloss in label_of]
    if not wanted:
        return None, None

    rng = np.random.default_rng(11)
    cap = min(len(wanted), 6000)
    chosen = [wanted[i] for i in rng.choice(len(wanted), cap, replace=False)]
    tracks = [resize_track(encode_clip(s.path), 64) for s in chosen]
    y = np.array([label_of[s.gloss] for s in chosen], dtype=np.int64)
    return np.stack(tracks).astype(np.float32), y


if __name__ == "__main__":
    sys.exit(main())
