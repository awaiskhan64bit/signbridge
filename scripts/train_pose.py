"""Train the sign classifier on the MuteMotion corpus and export ONNX.

The hand-only corpus capped this at 39.7% top-1 over 137 words, and the ceiling
was structural: 21 landmarks per hand gives a sign no reference beyond itself,
and 5.5 clips per word is thin for anything but memorisation. This path uses the
full 21,083-clip archive with body pose, so a hand's position relative to the
shoulders and face is available, and the median word has 15 clips rather than 7.

Everything else is deliberately the same as scripts/train.py: the same model, the
same augmentation, the same calibration on validation only. Comparing two runs
is only meaningful if one thing changes at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from signbridge.ml import holistic_dataset as H  # noqa: E402
from signbridge.ml import vocabulary  # noqa: E402
from signbridge.ml.calibration import calibrate, precision_at  # noqa: E402
from signbridge.ml.dataset import augment, fit_window, mixup, resize_track  # noqa: E402

DATA = ROOT / "data" / "mutemotion"
OUT = ROOT / "assets" / "models"


def build_model(num_classes: int, dims: int, hidden: int, layers: int, proj: int, device):
    import torch
    from torch import nn

    class SignNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(dims, proj),
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

    return SignNet().to(device)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--vocab", type=int, default=500)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.04)
    p.add_argument("--weight-power", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-pt", default="")
    args = p.parse_args()

    import torch
    from torch import nn

    device = torch.device("cpu")
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(6)
    print(f"training device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if not H.CACHE.exists():
        print("No encoded tracks. Run: python -c \"from signbridge.ml import "
              "holistic_dataset as H; H.build()\"", file=sys.stderr)
        return 1

    tracks, labels, splits, _ = H.load()
    dims = tracks[0].shape[1]
    print(f"corpus: {len(tracks)} clips, {dims} features per frame")

    counts = Counter(labels)
    keep = set(vocabulary.resolve(counts, args.vocab))
    label_of = {g: i for i, g in enumerate(sorted(keep))}
    print(f"vocabulary: {len(label_of)} words")

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    per_split: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    for t, g, s in zip(tracks, labels, splits, strict=True):
        if g in label_of:
            buckets[s].append(t)
            per_split[s][g] += 1
    for k, v in buckets.items():
        vals = sorted(per_split[k].values())
        if vals:
            print(
                f"  {k:5s}: {len(v):5d} clips, {len(per_split[k])} words, "
                f"clips/word min {vals[0]} median {vals[len(vals) // 2]} max {vals[-1]}"
            )

    y = {
        k: np.array([label_of[g] for t, g, s in zip(tracks, labels, splits, strict=True)
                     if g in label_of and s == k], dtype=np.int64)
        for k in buckets
    }
    train_t = buckets["train"]

    flat = np.concatenate(train_t, axis=0)
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-5
    del flat
    train_t = [(t - mean) / std for t in train_t]

    def windows(bs):
        return np.stack([resize_track(t, args.window) for t in bs])

    val_x, test_x = windows(buckets["val"]), windows(buckets["test"])

    model = build_model(len(label_of), dims, args.hidden, args.layers, 256, device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")

    cnt = np.bincount(y["train"], minlength=len(label_of)).astype(np.float64)
    w = np.where(cnt > 0, 1.0 / np.power(np.maximum(cnt, 1.0), args.weight_power), 0.0)
    weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, len(train_t) // args.batch),
    )
    y_t = torch.from_numpy(y["train"]).to(device)

    def dev(x):
        return torch.from_numpy(x).to(device, non_blocking=True)

    def evaluate(x, yy):
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(dev(x)), dim=1).cpu().numpy()
        top5 = np.argsort(-probs, axis=1)[:, :5]
        return (float((top5[:, 0] == yy).mean()),
                float((top5 == yy[:, None]).any(axis=1).mean()), probs)

    best, best_state, best_epoch, stale = 0.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_t))
        total = seen = 0
        for start in range(0, len(order) - args.batch + 1, args.batch):
            sel = order[start : start + args.batch]
            batch = np.stack(
                [fit_window(augment(train_t[i], rng), args.window) for i in sel]
            )
            targets = y_t[sel]
            if args.mixup > 0 and rng.random() < 0.5:
                perm = torch.from_numpy(rng.permutation(len(sel))).to(device)
                batch, lam = mixup(batch, batch[perm.cpu().numpy()], args.mixup, rng)
                logits = model(dev(batch))
                loss = lam * criterion(logits, targets) + (1 - lam) * criterion(
                    logits, targets[perm]
                )
            else:
                logits = model(dev(batch))
                loss = criterion(logits, targets)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(sel)
            seen += len(sel)

        a1, a5, _ = evaluate(val_x, y["val"])
        flag = ""
        if a1 > best:
            best, best_epoch, stale = a1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        else:
            stale += 1
        print(
            f"epoch {epoch:3d}  loss {total / max(1, seen):.4f}  val top1 {a1:.3f}  "
            f"top5 {a5:.3f}  {time.time() - t0:5.1f}s{flag}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stop, no improvement for {stale} epochs")
            break

    if best_state is None:
        print("no usable checkpoint", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    t1, t5, test_probs = evaluate(test_x, y["test"])
    _, _, val_probs = evaluate(val_x, y["val"])
    fitted = calibrate(val_probs, y["val"])
    cov, prec = precision_at(test_probs, y["test"], fitted.threshold)

    print(f"\nbest epoch {best_epoch}, val top1 {best:.3f}")
    print(f"test top1 {t1:.3f}  top5 {t5:.3f}")
    print(f"val  {fitted.describe()}")
    print(f"test {cov:.1%} coverage at {prec:.1%} precision")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.save_pt:
        Path(args.save_pt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "labels": sorted(label_of),
                "dims": dims,
                "window": args.window,
                "hidden": args.hidden,
                "layers": args.layers,
                "mean": mean,
                "std": std,
                "threshold": float(fitted.threshold),
            },
            args.save_pt,
        )
        print(f"wrote {Path(args.save_pt).name}")

    model = model.to("cpu").eval()
    onnx_path = OUT / "sign_classifier.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, args.window, dims),
        str(onnx_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    meta = {
        "name": f"WLASL-{len(label_of)}-pose",
        "labels": sorted(label_of),
        "window": args.window,
        "features": dims,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "threshold": float(fitted.threshold),
        "metrics": {
            "val_top1": float(best),
            "test_top1": float(t1),
            "test_top5": float(t5),
            "accept_accuracy": float(fitted.precision),
            "accept_coverage": float(fitted.coverage),
            "accept_accuracy_test": float(prec),
            "accept_coverage_test": float(cov),
            "params": int(sum(p.numel() for p in model.parameters())),
        },
        "source": {
            "dataset": "MuteMotion: WLASL MediaPipe encoded (Apache 2.0)",
            "note": "21,083 clips, 553 joints per frame including 33 pose.",
        },
    }
    (OUT / "sign_classifier.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())