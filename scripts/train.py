"""Train the WLASL sign classifier and export it to ONNX.

The vocabulary is the whole of WLASL, which is the hard version of this problem:
about twelve thousand clips spread across two thousand words, so most words have
only a handful of examples. The design choices below all follow from that.

Model
    A small per-frame projection, a two layer BiLSTM and attention pooling. The
    recurrence is what lets it read a sign as a movement rather than a pose, and
    the size keeps inference well under a millisecond on a laptop CPU.

Augmentation
    Crop, respeed, jitter, mirror, frame dropout and mixup. With this little
    data per class this matters more than the architecture does.

Loss
    Label smoothing plus a mild inverse frequency weight, so the long tail is
    learnable without the common words dominating the gradient.

Threshold
    A classifier this size on a vocabulary this long will be wrong often. Rather
    than pretend otherwise, the script picks the confidence threshold that
    maximises accepted-prediction accuracy on the validation set and stores it
    in the model metadata, so the app stays silent instead of speaking nonsense.

Usage
-----
    python scripts/train.py                    # full run
    python scripts/train.py --epochs 60        # longer
    python scripts/train.py --vocab 400        # easier subset while iterating
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # torch is imported lazily, it is a training-only dependency
    import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signbridge.ml import vocabulary  # noqa: E402
from signbridge.ml.calibration import calibrate, precision_at  # noqa: E402
from signbridge.ml.dataset import (  # noqa: E402
    WINDOW,
    WLASLIndex,
    augment,
    encode_clip,
    fit_window,
    mixup,
    resize_track,
)
from signbridge.ml.features import FEATURE_DIMS  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "assets" / "models"


# --------------------------------------------------------------------- model


def build_model(num_classes: int, hidden: int = 128, layers: int = 2, proj: int = 256):
    import torch
    from torch import nn

    class SignNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(FEATURE_DIMS, proj),
                nn.LayerNorm(proj),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            self.lstm = nn.LSTM(
                proj,
                hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden * 2)
            # One learned query gives a soft, sign-aware pooling over time.
            self.attn = nn.Linear(hidden * 2, 1)
            self.drop = nn.Dropout(0.4)
            self.head = nn.Linear(hidden * 2 * 3, num_classes)

        def forward(self, x):
            h, _ = self.lstm(self.stem(x))
            h = self.norm(h)

            weights = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
            attended = (h * weights).sum(dim=1)
            pooled = torch.cat([attended, h.mean(dim=1), h.max(dim=1).values], dim=1)
            return self.head(self.drop(pooled))

    return SignNet()


# ---------------------------------------------------------------------- data


def load_tracks(
    index: WLASLIndex, samples, cache: dict, workers: int = 8
) -> list[np.ndarray]:
    """Encode clips once and cache them. Encoding is the slow part of training.

    Clips come back at their natural lengths, so this returns a list rather than
    a stacked array. Everything that needs a fixed size window goes through
    `fit_window` or `resize_track`.
    """

    from concurrent.futures import ThreadPoolExecutor

    todo = [s for s in samples if s.video_id not in cache]

    def one(sample):
        try:
            return sample.video_id, encode_clip(sample.path)
        except Exception:
            return sample.video_id, np.zeros((1, FEATURE_DIMS), dtype=np.float32)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (vid, track) in enumerate(pool.map(one, todo), 1):
                cache[vid] = track
                if i % 500 == 0:
                    print(f"  encoded {i}/{len(todo)}")
    return [cache[s.video_id] for s in samples]


def class_weights(
    labels: np.ndarray, num_classes: int, power: float, device: torch.device
) -> torch.Tensor:
    """Mild inverse frequency weighting.

    The long tail of WLASL is severe, but fully balancing it lets two clip words
    outvote a hundred clip words and tanks the overall score. A half power keeps
    the tail learnable without letting it dominate.
    """

    import torch

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.where(counts > 0, 1.0 / np.power(np.maximum(counts, 1.0), power), 0.0)
    if weights.sum() == 0:
        return torch.ones(num_classes)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--vocab", type=int, default=0, help="cap classes by count")
    parser.add_argument(
        "--mode",
        choices=("curated", "frequent"),
        default="curated",
        help="how --vocab selects words: a curated everyday list, or top-N by "
        "clip count",
    )
    parser.add_argument("--mixup", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda or cpu. Training only; the app always runs on the CPU.",
    )
    parser.add_argument("--save-pt", default="", help="also write a .pt checkpoint here")
    args = parser.parse_args()

    import torch
    from torch import nn

    # The GPU is for training. Inference deliberately never touches it, see
    # ml/neural.py, so a missing or broken card costs training speed only.
    device = torch.device("cpu")
    if args.device != "cpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif args.device == "cuda":
            print("CUDA requested but not available, falling back to CPU", file=sys.stderr)
    if device.type == "cpu":
        # Leave the cores to the encoder threads rather than oversubscribing them.
        torch.set_num_threads(max(1, (args.threads or 4)))
    else:
        torch.set_num_threads(max(1, (args.threads or 4)))
    print(f"training device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    index_path = DATA / "wlasl-index.json"
    if not index_path.exists():
        print("No dataset index. Run: python scripts/fetch_wlasl.py", file=sys.stderr)
        return 1

    index = WLASLIndex(index_path)
    print(f"index: {len(index)} clips, {len(index.glosses)} glosses")

    train_s = index.split("train")
    val_s = index.split("val")
    test_s = index.split("test")
    print(f"official splits: train {len(train_s)}, val {len(val_s)}, test {len(test_s)}")

    # Optionally narrow the vocabulary.
    #
    # `--mode curated` uses a hand-picked list of words people actually sign.
    # `--mode frequent` keeps the old top-N-by-clip-count behaviour, which is
    # retained only as a comparison: it ranked by how many clips WLASL happens to
    # hold for a word, so "hello" (13 clips) fell outside the top 500 and the app
    # could not say the most basic greeting at all.
    glosses = index.glosses
    if args.vocab:
        freq = Counter(s.gloss for s in train_s)
        if args.mode == "curated":
            wanted = vocabulary.resolve(index.glosses, args.vocab)
            report = vocabulary.coverage_report(index.glosses)
            print(
                f"curated vocabulary: {len(wanted)} of {report['wanted']} requested "
                f"words are present in WLASL"
            )
            if report["core_missing"]:
                print(
                    f"  core words absent from WLASL: {report['core_missing']}",
                    file=sys.stderr,
                )
            keep = set(wanted)
        else:
            keep = {g for g, _ in freq.most_common(args.vocab)}
        glosses = [g for g in glosses if g in keep]
        train_s = [s for s in train_s if s.gloss in keep]
        val_s = [s for s in val_s if s.gloss in keep]
        test_s = [s for s in test_s if s.gloss in keep]
        clips = Counter(s.gloss for s in train_s)
        print(f"vocabulary: {len(glosses)} words, {len(train_s)} train clips")
        if clips:
            per = sorted(clips.values())
            print(
                f"  clips per word: min {per[0]}, median {per[len(per) // 2]}, "
                f"max {per[-1]}"
            )

    label_of = {g: i for i, g in enumerate(glosses)}
    num_classes = len(glosses)

    # A class missing from train is unusable as a label. Drop those clips.
    have = set(label_of)
    train_s = [s for s in train_s if s.gloss in have]
    val_s = [s for s in val_s if s.gloss in have]
    test_s = [s for s in test_s if s.gloss in have]
    if not train_s or not val_s:
        print("Not enough data after filtering.", file=sys.stderr)
        return 1

    print("encoding clips, this is the slow part on the first run")
    cache: dict = {}
    t0 = time.perf_counter()
    train_tracks = load_tracks(index, train_s, cache)
    val_tracks = load_tracks(index, val_s, cache)
    test_tracks = load_tracks(index, test_s, cache)
    print(f"encoded in {time.perf_counter() - t0:.1f}s")

    y_train = np.array([label_of[s.gloss] for s in train_s], dtype=np.int64)
    y_val = np.array([label_of[s.gloss] for s in val_s], dtype=np.int64)
    y_test = np.array([label_of[s.gloss] for s in test_s], dtype=np.int64)

    # Standardise on the training set only, so the val and test numbers stay
    # honest. Clips are variable length, so the statistics come from the
    # concatenated frames rather than from a stack.
    flat = np.concatenate(train_tracks, axis=0)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-5
    del flat

    def standardise(tracks):
        return [(t - mean) / std for t in tracks]

    def to_windows(tracks):
        return np.stack([resize_track(t, args.window) for t in tracks])

    train_tracks = standardise(train_tracks)
    val_windows = to_windows(standardise(val_tracks))
    test_windows = to_windows(standardise(test_tracks))

    model = build_model(num_classes, args.hidden, args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters, {num_classes} classes")

    weights = class_weights(y_train, num_classes, args.weight_power, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_s) // args.batch),
    )

    y_t = torch.from_numpy(y_train).to(device)

    def to_device(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(device, non_blocking=True)

    def evaluate(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        model.eval()
        with torch.no_grad():
            logits = model(to_device(x))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        top5 = np.argsort(-probs, axis=1)[:, :5]
        acc1 = float((top5[:, 0] == y).mean())
        acc5 = float((top5 == y[:, None]).any(axis=1).mean())
        return acc1, acc5, probs

    best_acc, best_state, best_epoch, stale = 0.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_s))
        total, seen = 0.0, 0
        for start in range(0, len(order) - args.batch + 1, args.batch):
            sel = order[start : start + args.batch]
            batch = np.stack(
                [
                    fit_window(augment(train_tracks[i], rng), args.window)
                    for i in sel
                ]
            )
            targets = y_t[sel]

            if args.mixup > 0 and rng.random() < 0.5:
                perm = torch.from_numpy(rng.permutation(len(sel))).to(device)
                batch, lam = mixup(batch, batch[perm.cpu().numpy()], args.mixup, rng)
                logits = model(to_device(batch))
                loss = lam * criterion(logits, targets) + (1 - lam) * criterion(
                    logits, targets[perm]
                )
            else:
                logits = model(to_device(batch))
                loss = criterion(logits, targets)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            total += loss.item() * len(sel)
            seen += len(sel)

        acc1, acc5, _ = evaluate(val_windows, y_val)
        flag = ""
        if acc1 > best_acc:
            best_acc, best_epoch, stale = acc1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        else:
            stale += 1
        print(
            f"epoch {epoch:3d}  loss {total / max(1, seen):.4f}  "
            f"val top1 {acc1:.3f}  top5 {acc5:.3f}{flag}"
        )
        if stale >= args.patience:
            print(f"early stop, no improvement for {stale} epochs")
            break

    if best_state is None:
        print("training produced no usable checkpoint", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    test1, test5, test_probs = evaluate(test_windows, y_test)
    _, _, val_probs = evaluate(val_windows, y_val)

    # Calibrate on validation, never on test. The test numbers below are what the
    # chosen cut actually buys, measured on data the cut never saw.
    fitted = calibrate(val_probs, y_val)
    threshold = fitted.threshold
    test_cov, test_prec = precision_at(test_probs, y_test, threshold)

    print(f"\nbest epoch {best_epoch}, val top1 {best_acc:.3f}")
    print(f"test top1 {test1:.3f}  top5 {test5:.3f}")
    print(f"val  {fitted.describe()}")
    print(f"test {test_cov:.1%} coverage at {test_prec:.1%} precision")

    if args.save_pt:
        ckpt = Path(args.save_pt)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "labels": glosses,
                "window": args.window,
                "features": FEATURE_DIMS,
                "hidden": args.hidden,
                "layers": args.layers,
                "num_classes": num_classes,
                "mean": mean,
                "std": std,
                "threshold": float(threshold),
                "metrics": {
                    "val_top1": float(best_acc),
                    "test_top1": float(test1),
                    "test_top5": float(test5),
                },
            },
            ckpt,
        )
        print(f"wrote checkpoint {ckpt.name}")

    # Export from a CPU copy. The app runs the graph through onnxruntime on the
    # CPU, so tracing it while the weights sit on a CUDA device would bake device
    # specific assumptions into the graph for no benefit.
    model = model.to("cpu").eval()
    OUT.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT / "sign_classifier.onnx"

    dummy = torch.from_numpy(np.zeros((1, args.window, FEATURE_DIMS), dtype=np.float32))
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # PyTorch 2.9+ defaults to the torch.export exporter, which wants
        # onnxscript installed for ops the LSTM lowers to. The TorchScript
        # exporter only needs onnx, which we already have.
        dynamo=False,
    )

    meta = {
        "name": f"WLASL-{num_classes}",
        "labels": glosses,
        "window": args.window,
        "features": FEATURE_DIMS,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "threshold": float(threshold),
        "metrics": {
            "val_top1": float(best_acc),
            "test_top1": float(test1),
            "test_top5": float(test5),
            "accept_accuracy": float(fitted.precision),
            "accept_coverage": float(fitted.coverage),
            "accept_accuracy_test": float(test_prec),
            "accept_coverage_test": float(test_cov),
            "params": int(n_params),
            "epochs_trained": int(best_epoch),
            "train_clips": len(train_s),
            "val_clips": len(val_s),
            "test_clips": len(test_s),
        },
        "source": {
            "dataset": "WLASL via a7medwa2l/wlasl-signals (unlicensed mirror)",
            "metadata": "dxli94/WLASL start_kit/WLASL_v0.3.json",
            "note": "WLASL itself is academic-use; the Hugging Face mirror has no licence.",
        },
    }
    (OUT / "sign_classifier.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size = onnx_path.stat().st_size / 1e6
    print(f"\nwrote {onnx_path.name} ({size:.1f} MB) and sign_classifier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
