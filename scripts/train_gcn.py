"""Train the skeleton graph model on WLASL and export it for CPU inference.

Uses the MuteMotion corpus, which is the only one here with trustworthy splits:
the hand-only mirror leaked 493 of 500 words between train and test, so every
accuracy number measured on it was optimistic.

Cache design. Only the body-relative joint track is cached, 64 frames of 75
joints in float16, which is 0.61GB for the whole corpus. The other three streams
are derived from it per batch. Caching all four would be 17GB, and it would also
freeze augmentation: bone vectors computed before augmentation describe joints
that no longer exist. Deriving after augmentation keeps them consistent and
costs a fraction of a millisecond.

Augmentation happens on joints, in the body frame, so it corresponds to
something a different signer or camera would actually do. In-plane rotation
covers camera roll, per-joint jitter covers tracking noise and body proportion,
temporal crop covers a sign performed at a different speed, frame dropout covers
the tracker losing the hands, and mirroring swaps the two hand subtrees rather
than negating x, which would be a hand seen from behind.

Calibration keeps the precision floor, so a model that cannot clear it is told to
stay silent rather than to speak its best guess.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Set before torch is imported anywhere. The backbone keeps a (batch, width,
# 64, 75) activation per block and the graph needs all of them for the backward
# pass, so allocator fragmentation is the difference between a 12 second epoch
# and an 8 GB card thrashing at 100% utilisation with no progress.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from signbridge.ml import skeleton as S  # noqa: E402
from signbridge.ml import vocabulary  # noqa: E402
from signbridge.ml.augment import augment  # noqa: E402
from signbridge.ml.calibration import calibrate, precision_at  # noqa: E402
from signbridge.ml.holistic import MuteMotion  # noqa: E402

OUT = ROOT / "assets" / "models"
DATA = ROOT / "data" / "mutemotion"
#: Body-relative joint tracks, one array of (clips, 64, 75, 3) float16.
JOINTS = DATA / "skel_joints.npy"
INDEX = DATA / "skel_index.json"

WINDOW = 64


# --------------------------------------------------------------------- cache


def encode_all() -> None:
    """Body-relative joint track for every clip, written once.

    Memory-mapped on load rather than read, so training never holds 0.61GB of
    skeleton it is not currently using.
    """

    mm = MuteMotion()
    n = len(mm)
    print(f"encoding {n} clips, {WINDOW} frames of {S.JOINTS} joints, float16", flush=True)

    buf = np.lib.format.open_memmap(
        JOINTS, mode="w+", dtype=np.float16, shape=(n, WINDOW, S.JOINTS, 3)
    )
    t0 = time.time()
    for i in range(n):
        buf[i] = S.build_input(mm.clip(i), WINDOW).astype(np.float16)
        if (i + 1) % 4000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{n}  {rate:.0f}/s  {n / rate / 60:.1f} min left", flush=True)
    buf.flush()
    del buf

    INDEX.write_text(
        json.dumps({"labels": mm.labels(), "splits": mm.splits()}), encoding="utf-8"
    )
    print(f"wrote {JOINTS.name} in {time.time() - t0:.0f}s", flush=True)


def load_cache() -> tuple[np.ndarray, list[str], list[str]]:
    if not JOINTS.exists():
        encode_all()
    data = np.load(JOINTS, mmap_mode="r")
    meta = json.loads(INDEX.read_text(encoding="utf-8"))
    return data, meta["labels"], meta["splits"]


def streams_of(joints: np.ndarray) -> dict[str, np.ndarray]:
    """The four input streams from a body-relative joint track.

    `joints` is (N, T, V, 3); each stream is returned in the (N, C, T, V) layout
    the model consumes.
    """

    b = S.bone_stream(joints)
    return {"j": joints, "jm": S.motion(joints), "b": b, "bm": S.motion(b)}


# --------------------------------------------------------------------- train


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--vocab", type=int, default=200)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.04)
    p.add_argument("--weight-power", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--aug", type=float, default=1.0, help="0 disables augmentation")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    p.add_argument("--encode", action="store_true", help="rebuild the cache first")
    args = p.parse_args()

    import torch
    from torch import nn

    if args.encode:
        encode_all()

    device = torch.device("cpu")
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(6)
    print(f"device {device}", flush=True)
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data, labels, splits = load_cache()
    counts = Counter(labels)
    keep = set(vocabulary.resolve(counts, args.vocab))
    label_of = {g: i for i, g in enumerate(sorted(keep))}
    num_classes = len(label_of)
    print(f"vocabulary {num_classes} words from {len(labels)} clips", flush=True)

    index: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, (g, s) in enumerate(zip(labels, splits, strict=True)):
        if g in label_of:
            index[s].append(i)
    for name, ids in index.items():
        per = sorted(Counter(labels[i] for i in ids).values())
        print(
            f"  {name:5s}: {len(ids):5d} clips, {len(per)} words, "
            f"clips/word min {per[0]} median {per[len(per) // 2]} max {per[-1]}",
            flush=True,
        )

    targets = {
        k: torch.tensor([label_of[labels[i]] for i in v], dtype=torch.long, device=device)
        for k, v in index.items()
    }

    from signbridge.ml.gcn import build

    model = build(
        num_classes, channels=args.channels, blocks=args.blocks, drop=args.dropout
    ).to(device)
    print(f"model {sum(q.numel() for q in model.parameters()):,} params", flush=True)

    freq = np.bincount(
        [label_of[labels[i]] for i in index["train"]], minlength=num_classes
    ).astype(np.float64)
    w = 1.0 / np.power(np.maximum(freq, 1.0), args.weight_power)
    weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(index["train"]) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps
    )

    def to_device(streams: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        # The cache and the augmenter both speak float16 to halve the disk and
        # the read bandwidth; the model is float32 throughout, so the cast
        # happens once here rather than at four call sites.
        return {
            k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            .permute(0, 3, 1, 2)
            .to(device, non_blocking=True)
            for k, v in streams.items()
        }

    @torch.no_grad()
    def evaluate(ids: list[int], gold: torch.Tensor) -> tuple[float, float, np.ndarray]:
        model.eval()
        chunks = []
        for start in range(0, len(ids), args.batch):
            j = np.asarray(data[ids[start : start + args.batch]], dtype=np.float32)
            chunks.append(torch.softmax(model(to_device(streams_of(j))), dim=1).cpu())
        probs = torch.cat(chunks).numpy()
        y = gold.cpu().numpy()
        top5 = np.argsort(-probs, axis=1)[:, :5]
        return (
            float((top5[:, 0] == y).mean()),
            float((top5 == y[:, None]).any(axis=1).mean()),
            probs,
        )

    train_ids = index["train"]
    best, best_state, best_epoch, stale = 0.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_ids))
        running = counted = 0
        for start in range(0, len(order) - args.batch + 1, args.batch):
            sel = order[start : start + args.batch]
            j = np.asarray(data[[train_ids[k] for k in sel]], dtype=np.float32)
            batch = to_device(streams_of(augment(j, rng, args.aug)))
            gold = targets["train"][sel]

            if args.mixup > 0 and rng.random() < 0.5:
                lam = float(rng.beta(args.mixup, args.mixup))
                order2 = torch.randperm(len(sel), device=device)
                batch = {k: lam * v + (1 - lam) * v[order2] for k, v in batch.items()}
                logits = model(batch)
                loss = lam * criterion(logits, gold) + (1 - lam) * criterion(
                    logits, gold[order2]
                )
            else:
                loss = criterion(model(batch), gold)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.detach().item() * len(sel)
            counted += len(sel)

        top1, top5, _ = evaluate(index["val"], targets["val"])
        note = ""
        if top1 > best:
            best, best_epoch, stale = top1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            note = "  <- best"
        else:
            stale += 1
        print(
            f"epoch {epoch:3d}  loss {running / max(1, counted):.4f}  "
            f"val top1 {top1:.3f}  top5 {top5:.3f}  {time.time() - t0:5.1f}s{note}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stop, no improvement for {stale} epochs", flush=True)
            break

    if best_state is None:
        print("no usable checkpoint", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    test1, test5, test_probs = evaluate(index["test"], targets["test"])
    _, _, val_probs = evaluate(index["val"], targets["val"])
    fitted = calibrate(val_probs, targets["val"].cpu().numpy())
    coverage, accept = precision_at(test_probs, targets["test"].cpu().numpy(), fitted.threshold)

    print(f"\nbest epoch {best_epoch}, val top1 {best:.3f}")
    print(f"test top1 {test1:.3f}  top5 {test5:.3f}")
    print(f"val  {fitted.describe()}")
    print(f"test {coverage:.1%} coverage at {accept:.1%} precision")

    OUT.mkdir(parents=True, exist_ok=True)
    classes = sorted(label_of)
    torch.save(
        {
            "state_dict": best_state,
            "labels": classes,
            "channels": args.channels,
            "blocks": args.blocks,
            "window": WINDOW,
            "threshold": float(fitted.threshold),
            "metrics": {
                "val_top1": float(best),
                "test_top1": float(test1),
                "test_top5": float(test5),
                "accept_accuracy": float(accept),
                "accept_coverage": float(coverage),
            },
        },
        OUT / "sign_classifier_gcn.pt",
    )
    (OUT / "sign_classifier_gcn.json").write_text(
        json.dumps(
            {
                "name": f"WLASL-{num_classes}-gcn",
                "labels": classes,
                "window": WINDOW,
                "channels": args.channels,
                "blocks": args.blocks,
                "joints": S.JOINTS,
                "edges": len(S.EDGES),
                "threshold": float(fitted.threshold),
                "metrics": {
                    "val_top1": float(best),
                    "test_top1": float(test1),
                    "test_top5": float(test5),
                    "accept_accuracy": float(accept),
                    "accept_coverage": float(coverage),
                },
                "source": "MuteMotion WLASL MediaPipe (Apache 2.0), official splits",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote assets/models/sign_classifier_gcn.pt and .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())