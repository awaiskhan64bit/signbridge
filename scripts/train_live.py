"""Train the sign classifier against the features the live pipeline actually emits.

The shipped model was trained on 120 features per frame, hand plus body pose,
while the app captures 21 MediaPipe hand landmarks and encodes them into 86.
A graph that expects 120 columns fed an 86 column window cannot do anything
useful, which is why the app sat at 0% confidence no matter what the model
scored. Training and inference have to agree on the shape before accuracy is even
the right question, so this trains on the 86 columns the camera path produces.

That means no body pose, which was the whole point of the MuteMotion download.
The honest position is that pose only helps if the app can also capture it, and
capturing it needs MediaPipe Holistic wired into the live path. Until that exists
the 86 column model is the only one that can work at all, so it is what ships.

Corpus is MuteMotion because it is the only one here with trustworthy splits.
The hand-only mirror leaked 493 of 500 words between train and test, which is
where the earlier 39.7% came from.
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

from signbridge.ml import holistic_dataset as H  # noqa: E402
from signbridge.ml import vocabulary  # noqa: E402
from signbridge.ml.calibration import calibrate, precision_at  # noqa: E402
from signbridge.ml.dataset import augment, fit_window, mixup, resize_track  # noqa: E402
from signbridge.ml.features import FEATURE_DIMS  # noqa: E402

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
                proj, hidden, num_layers=layers, batch_first=True,
                bidirectional=True, dropout=0.3 if layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden * 2)
            self.attn = nn.Linear(hidden * 2, 1)
            self.drop = nn.Dropout(0.4)
            self.head = nn.Linear(hidden * 2 * 3, num_classes)

        def forward(self, x):
            h, _ = self.lstm(self.stem(x))
            h = self.norm(h)
            w = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
            pooled = torch.cat([(h * w).sum(1), h.mean(1), h.max(1).values], dim=1)
            return self.head(self.drop(pooled))

    return SignNet().to(device)


def run(vocab_size: int, args, tracks, labels, splits, device, torch, nn):
    """Train once at a given vocabulary size and return the test metrics."""

    counts = Counter(labels)
    keep = set(vocabulary.resolve(counts, vocab_size))
    label_of = {g: i for i, g in enumerate(sorted(keep))}

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    per: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    for t, g, s in zip(tracks, labels, splits, strict=True):
        if g in label_of:
            buckets[s].append(t)
            per[s][g] += 1

    if not buckets["train"] or not buckets["val"] or not buckets["test"]:
        return None

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

    win = lambda bs: np.stack([resize_track(t, args.window) for t in bs])  # noqa: E731
    val_x, test_x = win(buckets["val"]), win(buckets["test"])

    rng = np.random.default_rng(args.seed)
    model = build_model(len(label_of), FEATURE_DIMS, args.hidden, args.layers, args.proj, device)

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
    dev = lambda a: torch.from_numpy(a).to(device)  # noqa: E731

    def evaluate(x, yy):
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(dev(x)), dim=1).cpu().numpy()
        top5 = np.argsort(-probs, axis=1)[:, :5]
        return (float((top5[:, 0] == yy).mean()),
                float((top5 == yy[:, None]).any(axis=1).mean()), probs)

    best, best_state, stale = 0.0, None, 0
    for _epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_t))
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
                    logits, targets[perm])
            else:
                logits = model(dev(batch))
                loss = criterion(logits, targets)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        a1, _, _ = evaluate(val_x, y["val"])
        if a1 > best:
            best, stale = a1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    t1, t5, test_probs = evaluate(test_x, y["test"])
    _, _, val_probs = evaluate(val_x, y["val"])
    fitted = calibrate(val_probs, y["val"])
    cov, prec = precision_at(test_probs, y["test"], fitted.threshold)
    tr = sorted(per["train"].values())
    return {
        "vocab": len(label_of),
        "val_top1": best,
        "test_top1": t1,
        "test_top5": t5,
        "train_median": tr[len(tr) // 2],
        "precision": prec,
        "coverage": cov,
        "threshold": float(fitted.threshold),
        "model": model,
        "labels": sorted(label_of),
        "mean": mean,
        "std": std,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--proj", type=int, default=256)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--vocab", type=int, default=500)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.04)
    p.add_argument("--weight-power", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.add_argument("--sweep", default="", help="comma separated vocab sizes to compare")
    p.add_argument("--save-pt", default="")
    args = p.parse_args()

    import torch
    from torch import nn

    device = torch.device("cpu")
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(6)
    print(f"device {device} | {FEATURE_DIMS} features per frame (matches live capture)")

    tracks, labels, splits, _ = H.load()
    tracks = [t[:, :FEATURE_DIMS] for t in tracks]
    print(f"corpus {len(tracks)} clips\n")

    sizes = [int(s) for s in args.sweep.split(",")] if args.sweep else [args.vocab]
    results = []
    for n in sizes:
        print(f"--- vocab {n} ---", flush=True)
        r = run(n, args, tracks, labels, splits, device, torch, nn)
        if r is None:
            print("  unusable, skipped")
            continue
        print(
            f"  {r['vocab']} words | train med {r['train_median']} clips | "
            f"val {r['val_top1']:.3f} | test top1 {r['test_top1']:.3f} "
            f"top5 {r['test_top5']:.3f} | {r['coverage']:.0%} at {r['precision']:.0%}",
            flush=True,
        )
        results.append(r)

    if not results:
        return 1

    # Ship the run the user asked for, not the sweep winner.
    chosen = next((r for r in results if r["vocab"] == args.vocab), results[0])
    print(f"\nshipping the {chosen['vocab']}-word model")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.save_pt:
        torch.save(
            {"state_dict": chosen["model"].state_dict(), "labels": chosen["labels"],
             "dims": FEATURE_DIMS, "window": args.window, "hidden": args.hidden,
             "layers": args.layers, "mean": chosen["mean"], "std": chosen["std"],
             "threshold": chosen["threshold"]},
            args.save_pt,
        )

    model = chosen["model"].to("cpu").eval()
    onnx_path = OUT / "sign_classifier.onnx"
    torch.onnx.export(
        model, torch.zeros(1, args.window, FEATURE_DIMS), str(onnx_path),
        input_names=["window"], output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    meta = {
        "name": f"WLASL-{chosen['vocab']}",
        "labels": chosen["labels"],
        "window": args.window,
        "features": FEATURE_DIMS,
        "mean": [float(v) for v in chosen["mean"]],
        "std": [float(v) for v in chosen["std"]],
        "threshold": chosen["threshold"],
        "metrics": {
            "val_top1": chosen["val_top1"],
            "test_top1": chosen["test_top1"],
            "test_top5": chosen["test_top5"],
            "accept_accuracy": chosen["precision"],
            "accept_coverage": chosen["coverage"],
            "params": int(sum(q.numel() for q in model.parameters())),
        },
        "source": {
            "dataset": "MuteMotion: WLASL MediaPipe encoded (Apache 2.0)",
            "note": "Official disjoint splits. Hand landmarks only, matching live capture.",
        },
    }
    (OUT / "sign_classifier.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())