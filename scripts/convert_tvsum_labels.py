#!/usr/bin/env python
"""Convert evaluation-only TVSum labels after frozen six-run predictions pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unsgrpo.evaluation import file_sha256, validate_six_run_freeze


def _scalar(value, name):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"TVSum {name} must be a scalar")
    value = float(arr.item())
    if not np.isfinite(value) or value < 1 or value != int(value):
        raise ValueError(f"TVSum {name} must be a positive integer")
    return int(value)


def convert(mat_path, freeze_path, out_dir, *, allow_synthetic_fixture=False):
    audited = validate_six_run_freeze(freeze_path)  # Must precede label import/read.
    if not audited["require_run_provenance"] and not allow_synthetic_fixture:
        raise ValueError("real label conversion requires freeze.require_run_provenance=true and verified six-run receipts")
    out_dir = Path(out_dir)
    test_videos = audited["test_videos"]
    if any((out_dir / f"{video['video_id']}.npz").exists() for video in test_videos):
        raise FileExistsError("refusing to overwrite existing converted labels")
    from scipy.io import loadmat

    source = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "tvsum50" not in source:
        raise ValueError("MAT source is missing documented tvsum50 structure")
    by_id = {}
    for item in np.atleast_1d(source["tvsum50"]):
        video_id = str(item.video)
        if video_id in by_id:
            raise ValueError(f"duplicate TVSum source identity: {video_id}")
        by_id[video_id] = item
    validated = []
    for video in test_videos:
        video_id = video["video_id"]
        if video_id not in by_id:
            raise ValueError(f"missing TVSum annotation identity: {video_id}")
        item = by_id[video_id]
        ratings = np.asarray(item.user_anno, dtype=np.float64)
        source_nframes = _scalar(item.nframes, "nframes")
        native_nframes = int(video["n_frames"])
        bounds_path = Path(video["frame_bounds_path"])
        if not bounds_path.is_absolute():
            bounds_path = Path(audited["manifest_path"]).parent / bounds_path
        frame_bounds = np.load(bounds_path, allow_pickle=False)
        if source_nframes != native_nframes:
            raise ValueError(f"annotation/native frame mismatch for {video_id}: {source_nframes} != {native_nframes}")
        if ratings.shape != (source_nframes, 20):
            raise ValueError(f"ratings axes must be [nframes,20] for {video_id}; got {ratings.shape}")
        if frame_bounds.shape != (native_nframes, 2) or not np.isfinite(frame_bounds).all():
            raise ValueError(f"invalid native frame_bounds for {video_id}")
        if not np.isfinite(ratings).all() or np.any((ratings < 1) | (ratings > 5)):
            raise ValueError(f"invalid TVSum 1-5 ratings for {video_id}")
        if np.any(frame_bounds[:, 1] <= frame_bounds[:, 0]):
            raise ValueError(f"nonpositive frame interval for {video_id}")
        if not np.allclose(frame_bounds[1:, 0], frame_bounds[:-1, 1], rtol=0, atol=1e-6):
            raise ValueError(f"native frame time axis has gaps/overlap for {video_id}")
        if not np.isclose(frame_bounds[0, 0], 0, atol=1e-6) or not np.isclose(frame_bounds[-1, 1], video["duration"], atol=1e-6):
            raise ValueError(f"native frame time axis does not cover registered duration for {video_id}")
        validated.append((video_id, frame_bounds, ratings, str(bounds_path), file_sha256(bounds_path)))
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "COMPLETED",
        "source_mat_sha256": file_sha256(mat_path),
        "freeze_sha256": audited["freeze_sha256"],
        "manifest_sha256": audited["manifest_sha256"],
        "evaluation_unit": "native_frame",
        "axes": ["frame_interval", "annotator"],
        "training_labels_written": False,
        "synthetic_fixture_only": bool(allow_synthetic_fixture),
        "require_run_provenance": audited["require_run_provenance"],
        "videos": [],
    }
    for video_id, bounds, ratings, bounds_path, bounds_hash in validated:
        dest = out_dir / f"{video_id}.npz"
        np.savez_compressed(dest, bounds=bounds, ratings=ratings)
        receipt["videos"].append({"video_id": video_id, "n_frames": len(bounds),
                                  "native_bounds_path": bounds_path,
                                  "native_bounds_sha256": bounds_hash,
                                  "converted_sha256": file_sha256(dest)})
    (out_dir / "conversion_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = convert(args.mat, args.freeze, args.out)
    print(json.dumps({"status": result["status"], "n_test_videos": len(result["videos"])}))
