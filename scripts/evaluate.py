#!/usr/bin/env python
"""Evaluate all six frozen predictions using V0; V1 remains unaligned."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unsgrpo.evaluation import (
    RUN_IDS,
    evaluate_v0,
    file_sha256,
    paired_bootstrap,
    validate_six_run_freeze,
)


def evaluate(freeze_path, annotation_dir, output_path, *, allow_synthetic_fixture=False):
    audited = validate_six_run_freeze(freeze_path)  # No annotations before this gate.
    if not audited["require_run_provenance"] and not allow_synthetic_fixture:
        raise ValueError("real evaluation requires freeze.require_run_provenance=true and verified six-run receipts")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("use a new independent evaluation output path")
    annotation_dir = Path(annotation_dir)
    receipt_path = annotation_dir / "conversion_receipt.json"
    conversion = json.loads(receipt_path.read_text())
    if conversion.get("status") != "COMPLETED" or conversion.get("freeze_sha256") != audited["freeze_sha256"]:
        raise ValueError("label conversion is not bound to the current frozen predictions")
    if conversion.get("manifest_sha256") != audited["manifest_sha256"]:
        raise ValueError("label conversion manifest hash mismatch")
    converted = {row["video_id"]: row for row in conversion["videos"]}
    report = {
        "status": "COMPLETED",
        "protocol": "V0-strict-median-noninterpolated-AP-v1",
        "synthetic_fixture_only": bool(allow_synthetic_fixture),
        "require_run_provenance": audited["require_run_provenance"],
        "evaluation_unit": conversion["evaluation_unit"],
        "prediction_freeze_sha256": audited["freeze_sha256"],
        "manifest_sha256": audited["manifest_sha256"],
        "conversion_receipt_sha256": file_sha256(receipt_path),
        "V1": {"status": "BLOCKED", "reason": "real benchmark clip alignment and reference split unverified"},
        "runs": {},
    }
    labels = {}
    for video in audited["test_videos"]:
        video_id = video["video_id"]
        path = annotation_dir / f"{video_id}.npz"
        if video_id not in converted or file_sha256(path) != converted[video_id]["converted_sha256"]:
            raise ValueError(f"converted label hash mismatch: {video_id}")
        with np.load(path, allow_pickle=False) as data:
            labels[video_id] = (data["bounds"].copy(), data["ratings"].copy())
    for run_id in RUN_IDS:
        details = []
        run = audited["runs"][run_id]
        for video in audited["test_videos"]:
            video_id = video["video_id"]
            pred = run["videos"][video_id]
            bounds, ratings = labels[video_id]
            result = evaluate_v0(pred["bounds"], pred["scores"], bounds, ratings)
            details.append({"video_id": video_id, "source_group": video["source_group"], **result})
        defined = [row["spearman"]["value"] for row in details if row["spearman"]["defined"]]
        report["runs"][run_id] = {
            "prediction_sha256": run["sha256"],
            "n_videos": len(details),
            "video_macro_ap": float(np.mean([row["mean_ap"] for row in details])),
            "mean_defined_spearman": float(np.mean(defined)) if defined else None,
            "n_undefined_spearman": len(details) - len(defined),
            "n_no_positive_annotators": sum(row["n_no_positive_annotators"] for row in details),
            "per_video": details,
        }
    first = report["runs"]["G-RANK"]["per_video"]
    second = report["runs"]["S+"]["per_video"]
    report["primary_paired_difference"] = {
        "contrast": "G-RANK minus S+",
        **paired_bootstrap([a["mean_ap"] - b["mean_ap"] for a, b in zip(first, second)],
                           source_groups=[row["source_group"] for row in first], n_boot=10000, seed=42),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(args.freeze, args.annotations, args.out)
    print(json.dumps({"status": result["status"], "protocol": result["protocol"],
                      "V1_status": result["V1"]["status"]}))
