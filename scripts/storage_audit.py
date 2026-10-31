#!/usr/bin/env python3
"""Create external asset directories and audit the registered 50/200 GiB plan."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

GIB = 1024**3
PROJECT = Path(os.environ.get("UNSGRPO_PROJECT_DIR", "/root/autodl-tmp/unsgrpo"))
WORK = Path(os.environ.get("UNSGRPO_WORK_DIR", "/root/autodl-tmp/unsgrpo-work"))
STORE = Path(os.environ.get("UNSGRPO_STORAGE_DIR", "/root/autodl-fs/unsgrpo"))


def allocated_bytes(path: Path) -> int:
    """Count allocated blocks once per inode, including hidden files."""
    seen: set[tuple[int, int]] = set()
    total = 0
    if not path.exists():
        return 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in [".", *files]:
            p = Path(root) if name == "." else Path(root) / name
            try:
                stat = p.lstat()
            except FileNotFoundError:
                continue  # Active training may rotate a checkpoint during the audit.
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                total += stat.st_blocks * 512
                seen.add(key)
    return total


def throughput(path: Path, mib: int) -> dict:
    """Small sequential I/O probe; fsync write, cached read (not a cold benchmark)."""
    block = os.urandom(1024**2)
    fd, name = tempfile.mkstemp(prefix=".unsgrpo-storage-probe-", dir=path)
    try:
        t = time.monotonic()
        with os.fdopen(fd, "wb", buffering=0) as f:
            for _ in range(mib):
                f.write(block)
            os.fsync(f.fileno())
        write_s = time.monotonic() - t
        t = time.monotonic()
        with open(name, "rb", buffering=0) as f:
            while f.read(1024**2):
                pass
        read_s = time.monotonic() - t
        return {"bytes": mib * 1024**2, "write_fsync_seconds": write_s,
                "write_mib_per_s": mib / write_s, "cached_read_seconds": read_s,
                "cached_read_mib_per_s": mib / read_s,
                "limitation": "16 MiB-style sequential warm-cache probe; not sustained/random throughput"}
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--throughput-mib", type=int, default=0)
    parser.add_argument("--reserve-path", type=Path,
                        help="Before a download/merge, check peak extra bytes on this filesystem")
    parser.add_argument("--peak-extra-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.throughput_mib < 0 or args.throughput_mib > 256 or args.peak_extra_bytes < 0:
        parser.error("probe must be 0..256 MiB and peak bytes must be nonnegative")
    dirs = [WORK / p for p in ("envs", "cache", "prepared", "runs", "checkpoints", "tmp")]
    dirs += [STORE / p for p in ("models", "datasets/raw", "datasets/labels_eval_only",
                                "datasets/pseudo", "datasets/trainer_inputs", "archives")]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    groups = {
        "local_code_env_cache": ([PROJECT, WORK / "envs", WORK / "cache", WORK / "prepared"], 15),
        "local_active": ([WORK / "runs", WORK / "checkpoints"], 25),
        "local_temporary": ([WORK / "tmp"], 10),
        "file_models": ([STORE / "models"], 80),
        "file_datasets": ([STORE / "datasets"], 50),
        "file_archives": ([STORE / "archives"], 50),
    }
    result = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "budget_unit": "GiB", "local_capacity_gib": 50, "file_capacity_gib": 200,
              "budgets_are_hard_quotas": False, "groups": {}, "filesystems": {}, "errors": []}
    for name, (paths, budget) in groups.items():
        actual = sum(allocated_bytes(p) for p in paths)
        result["groups"][name] = {"paths": list(map(str, paths)), "allocated_bytes": actual,
                                  "budget_bytes": budget * GIB}
        if actual > budget * GIB:
            result["errors"].append(f"{name} exceeds registered budget")
    for name, path, expected, reserve in [("local", WORK, 50, 10), ("file", STORE, 200, 20)]:
        usage = shutil.disk_usage(path)
        entry = {"path": str(path.resolve()), "device": path.stat().st_dev,
                 "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free,
                 "minimum_free_bytes": reserve * GIB}
        result["filesystems"][name] = entry
        if usage.total < expected * GIB:
            result["errors"].append(f"{name} capacity smaller than {expected} GiB plan")
        if usage.free < reserve * GIB:
            result["errors"].append(f"{name} free space below registered reserve")
        if args.throughput_mib:
            entry["probe"] = throughput(path, args.throughput_mib)
    if WORK.stat().st_dev == STORE.stat().st_dev:
        result["errors"].append("local and file storage unexpectedly share one device")
    if args.reserve_path:
        target = args.reserve_path.resolve()
        while not target.exists():
            target = target.parent
        usage = shutil.disk_usage(target)
        device = target.stat().st_dev
        registered = [v for v in result["filesystems"].values() if v["device"] == device]
        if not registered:
            result["errors"].append("reservation target is outside registered filesystems")
            reserve = 0
        else:
            reserve = registered[0]["minimum_free_bytes"]
        result["peak_reservation"] = {"path": str(args.reserve_path), "extra_bytes": args.peak_extra_bytes,
                                      "free_after_bytes": usage.free - args.peak_extra_bytes,
                                      "minimum_free_bytes": reserve}
        if usage.free - args.peak_extra_bytes < reserve:
            result["errors"].append("estimated peak would consume required safety reserve")
    result["ok"] = not result["errors"]
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
