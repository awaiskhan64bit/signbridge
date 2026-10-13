"""Sweep the factors that could be holding the graph model below the flat one.

Each config trains at 50 words for 40 epochs, which is about 3 minutes, and
prints the val and test top-1. The flat BiLSTM reached 16.4% test on this same
split, so anything at or below that is a regression the graph was supposed to fix.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN = ROOT / "scripts" / "train_gcn.py"

BASE = ["--epochs", "40", "--vocab", "50", "--batch", "32", "--patience", "40"]

CONFIGS = [
    ("baseline", []),
    ("no-aug", ["--aug", "0"]),
    ("no-regularisers", ["--aug", "0", "--mixup", "0", "--label-smoothing", "0",
                         "--weight-power", "0", "--dropout", "0.1"]),
    ("lr-3e-4", ["--lr", "0.0003"]),
    ("lr-3e-3", ["--lr", "0.003"]),
    ("wide-64", ["--channels", "64", "--batch", "16"]),
    ("no-aug + wide-64", ["--aug", "0", "--channels", "64", "--batch", "16"]),
    ("no-aug + lr-3e-3", ["--aug", "0", "--lr", "0.003"]),
]

rows = []
for name, extra in CONFIGS:
    cmd = [str(PY), str(TRAIN), *BASE, *extra]
    print(f"=== {name} ===", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = proc.stdout
    val = test = cov = None
    best = None
    for line in out.splitlines():
        if line.startswith("epoch ") and "top1" in line:
            bits = line.split("val top1")[1].split()
            best = max(best or 0.0, float(bits[0]))
        if line.startswith("test top1"):
            test = float(line.split()[2])
        if line.startswith("best epoch"):
            best = float(line.split()[-1])
    rows.append({"name": name, "val_top1": best, "test_top1": test})
    print(f"  val {best}  test {test}", flush=True)

print("\n" + "=" * 56)
print(f"{'config':22s} {'val top1':>10s} {'test top1':>10s}")
print("-" * 56)
for r in sorted(rows, key=lambda r: -(r["test_top1"] or 0)):
    print(f"{r['name']:22s} {(r['val_top1'] or 0):10.3f} {(r['test_top1'] or 0):10.3f}")
print("=" * 56)
print("flat BiLSTM reference on this split: 0.164 test")
(ROOT / "_sweep.json").write_text(json.dumps(rows, indent=2))