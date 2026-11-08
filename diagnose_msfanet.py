import os
import sys
import csv
import json
import zipfile
import hashlib
import platform
import re
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
REPORT = ROOT / "msfanet_diagnostic_report.txt"
BUNDLE = ROOT / "msfanet_diagnostic_bundle.zip"

CODE_CANDIDATES = [
    "projects/msfanet/train.py",
    "projects/msfanet/engine.py",
    "projects/msfanet/losses.py",
    "projects/msfanet/augmentations.py",
    "projects/msfanet/complexity.py",
    "projects/msfanet/visualization.py",
    "torchreid/models/msfanet/msfanet.py",
    "torchreid/models/msfanet/aggregation_gate.py",
    "torchreid/models/msfanet/fa_block.py",
    "torchreid/models/msfanet/msfa_block.py",
    "torchreid/models/msfanet/se_block.py",
    "torchreid/metrics/rank.py",
]

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".idea", ".vscode", "site-packages"
}
TEXT_SUFFIXES = {".py", ".txt", ".log", ".csv", ".yaml", ".yml", ".json", ".ini", ".cfg"}

def add(lines, value=""):
    lines.append(str(value))

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def walk_dataset(dataset_root, lines):
    add(lines, f"\nDATASET ROOT: {dataset_root}")
    if not dataset_root.exists():
        add(lines, "STATUS: NOT FOUND")
        return
    for split in ("bounding_box_train", "query", "bounding_box_test"):
        folder = dataset_root / split
        if not folder.exists():
            add(lines, f"{split}: MISSING")
            continue
        files = [p for p in folder.iterdir() if p.is_file()]
        image_files = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        pids, camids = set(), set()
        malformed = []
        for p in image_files:
            m = re.match(r"^(-?\d+)_c(\d+)", p.name)
            if not m:
                malformed.append(p.name)
                continue
            pid, cam = int(m.group(1)), int(m.group(2))
            if pid == -1:
                continue
            pids.add(pid)
            camids.add(cam)
        add(lines, f"{split}: files={len(files)}, images={len(image_files)}, identities={len(pids)}, cameras={sorted(camids)}")
        if malformed:
            add(lines, f"  filenames not matching Market-1501 pattern: {len(malformed)}; examples={malformed[:5]}")
        if split == "bounding_box_train":
            add(lines, f"  expected reference: about 12936 train images, 751 train identities (excluding junk)")
        elif split == "query":
            add(lines, f"  expected reference: 3368 query images")
        elif split == "bounding_box_test":
            add(lines, f"  expected reference: 15913 gallery images")

def main():
    lines = []
    add(lines, "MSFANET DIAGNOSTIC REPORT")
    add(lines, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    add(lines, f"Working directory: {ROOT}")
    add(lines, f"Python executable: {sys.executable}")
    add(lines, f"Python version: {sys.version}")
    add(lines, f"Platform: {platform.platform()}")

    add(lines, "\nPACKAGE / DEVICE INFORMATION")
    try:
        import torch
        add(lines, f"PyTorch: {torch.__version__}")
        add(lines, f"CUDA available: {torch.cuda.is_available()}")
        add(lines, f"CUDA version: {torch.version.cuda}")
        add(lines, f"CPU threads: {torch.get_num_threads()}")
        if torch.cuda.is_available():
            add(lines, f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        add(lines, f"PyTorch import failed: {repr(e)}")
    try:
        import torchreid
        add(lines, f"torchreid module: {getattr(torchreid, '__file__', 'unknown')}")
        add(lines, f"torchreid version: {getattr(torchreid, '__version__', 'unknown')}")
    except Exception as e:
        add(lines, f"torchreid import failed: {repr(e)}")

    add(lines, "\nDATASET CHECK")
    dataset_options = [
        ROOT / "data/market1501/Market-1501-v15.09.15",
        ROOT / "data/Market-1501-v15.09.15",
        ROOT / "datasets/Market-1501-v15.09.15",
        ROOT / "Market-1501-v15.09.15",
    ]
    found = [p for p in dataset_options if p.exists()]
    if found:
        for p in found:
            walk_dataset(p, lines)
    else:
        add(lines, "Market-1501 root not found in common locations.")
        add(lines, "Search your train.py for the actual dataset path and verify it manually.")

    add(lines, "\nSOURCE FILE INVENTORY")
    for rel in CODE_CANDIDATES:
        p = ROOT / rel
        add(lines, f"{rel}: {'FOUND' if p.is_file() else 'MISSING'}")
        if p.is_file():
            add(lines, f"  bytes={p.stat().st_size}, sha256={sha256(p)}")

    add(lines, "\nCHECKPOINT INVENTORY")
    checkpoint_dirs = [ROOT / "log", ROOT / "checkpoints", ROOT / "output", ROOT / "runs"]
    checkpoints = []
    for base in checkpoint_dirs:
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".pth", ".pt", ".ckpt"}:
                    checkpoints.append(p)
    if not checkpoints:
        add(lines, "No .pth/.pt/.ckpt files found in log/checkpoints/output/runs.")
    for p in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
        add(lines, f"{p.relative_to(ROOT)} | bytes={p.stat().st_size} | modified={datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds')}")

    add(lines, "\nTRAINING / EVALUATION CSV INVENTORY")
    csv_files = []
    for base in (ROOT / "log", ROOT / "checkpoints", ROOT / "output", ROOT / "runs", ROOT):
        if base.exists():
            for p in base.rglob("*.csv"):
                if any(part in SKIP_DIRS for part in p.parts):
                    continue
                if p not in csv_files:
                    csv_files.append(p)
        if base == ROOT:
            break
    if not csv_files:
        add(lines, "No CSV files found.")
    for p in sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
        add(lines, f"\n{p.relative_to(ROOT)}")
        try:
            with p.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                rows = list(csv.reader(f))
            add(lines, f"  rows={len(rows)}")
            for row in rows[:4]:
                add(lines, "  " + " | ".join(row[:20]))
            if len(rows) > 4:
                add(lines, "  ...")
                for row in rows[-5:]:
                    add(lines, "  " + " | ".join(row[:20]))
        except Exception as e:
            add(lines, f"  read error: {repr(e)}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    included = []
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(REPORT, REPORT.name)
        for rel in CODE_CANDIDATES:
            p = ROOT / rel
            if p.is_file():
                z.write(p, str(p.relative_to(ROOT)))
                included.append(str(p.relative_to(ROOT)))
        # Include relevant small text logs and CSVs, not image data or model weights.
        for base in (ROOT / "log", ROOT / "output", ROOT / "runs", ROOT / "checkpoints"):
            if base.exists():
                for p in base.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
                        continue
                    if any(part in SKIP_DIRS for part in p.parts):
                        continue
                    try:
                        if p.stat().st_size <= 5 * 1024 * 1024:
                            z.write(p, str(p.relative_to(ROOT)))
                    except OSError:
                        pass

    print("\n".join(lines))
    print("\nCreated:")
    print(REPORT)
    print(BUNDLE)
    print(f"\nSource files included: {included}")

if __name__ == "__main__":
    main()
