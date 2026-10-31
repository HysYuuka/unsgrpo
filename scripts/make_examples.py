#!/usr/bin/env python3
"""Materialize a fixed per-video sampled pool and unique all-covering eval inputs."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unsgrpo.data import audit_split, audit_window_record, build_window_record, canonical_hash, window_indices


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pseudo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows-per-video", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-media-file-check", action="store_true",
                        help="Synthetic debugging only; never use in the real preprocessing acceptance")
    args = parser.parse_args()
    if args.windows_per_video < 1:
        parser.error("windows per video must be positive")
    project = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(project):
        parser.error("generated trainer input files must be outside the code repository")
    if any((args.out / name).exists() for name in ("train_sft.jsonl", "train_grpo.jsonl", "eval.jsonl", "eval_train_diagnostic.jsonl", "window_pool.jsonl", "data_audit.json", "data.sha256")):
        parser.error("output already exists; use a new directory to preserve data identities")
    rows = sorted([json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()], key=lambda x: x["video_id"])
    split_audit = audit_split(rows)
    pseudo = json.loads(args.pseudo.read_text())
    if pseudo["prepared_manifest_sha256"] != file_hash(args.manifest):
        raise ValueError("pseudo labels were built from a different prepared manifest")
    if set(pseudo["videos"]) != {r["video_id"] for r in rows}:
        raise ValueError("pseudo/manifest video IDs differ")
    rng = np.random.default_rng(args.seed)
    outputs = {"train_sft": [], "train_grpo": [], "eval": [], "eval_train_diagnostic": []}
    pool, counts, selected_window_counts = [], {}, collections.Counter()
    for row in rows:
        vid = row["video_id"]
        score_row = pseudo["videos"][vid]
        if score_row["split"] != row["split"]:
            raise ValueError("pseudo/manifest split mismatch")
        if score_row["features_sha256"] != file_hash(Path(row["features_path"])):
            raise ValueError(f"features changed after pseudo construction: {vid}")
        q = score_row["q"]
        if len(q) != len(row["bounds"]):
            raise ValueError("pseudo/full-video clip count mismatch")
        common = {k: row[k] for k in ("video_id", "source_group", "split", "duration", "bounds", "frame_paths", "sampled_pts")}
        common["overview_path"] = row["overview"]
        windows = window_indices(len(row["bounds"]))
        for ids in windows:
            record = build_window_record(**common, clip_ids=ids, task="eval")
            record["dataset_request_id"] = record["request_id"]
            audit_window_record(record, check_media_files=not args.skip_media_file_check)
            target = "eval_train_diagnostic" if row["split"] == "train" else "eval"
            outputs[target].append(record)
        if row["split"] != "train":
            continue
        selected = rng.integers(0, len(windows), size=args.windows_per_video)
        counts[vid] = {"all_covering_candidate_windows": len(windows),
                       "pool_instances": len(selected), "unique_selected_windows": len(np.unique(selected)),
                       "selected_candidate_indices": selected.tolist()}
        for instance, candidate_index in enumerate(selected):
            ids = windows[int(candidate_index)]
            sft = build_window_record(**common, clip_ids=ids, pseudo_scores=q, task="sft")
            grpo = build_window_record(**common, clip_ids=ids, pseudo_scores=q, task="grpo")
            # ms-swift 4.6 replaces request_id with a completion UUID during rollout.
            # Keep this extra field in remove_unused_columns=False metadata for ledgers.
            sft["dataset_request_id"] = sft["request_id"]
            grpo["dataset_request_id"] = grpo["request_id"]
            instance_id = f"{vid}__pool_{instance:04d}"
            sft["pool_instance_id"] = instance_id
            grpo["pool_instance_id"] = instance_id
            if sft["request_id"] != grpo["request_id"] or sft["media_signature"] != grpo["media_signature"]:
                raise AssertionError("SFT/GRPO request or media identity mismatch")
            if sft["messages"][:2] != grpo["messages"] or sft["pseudo_scores"] != grpo["pseudo_scores"]:
                raise AssertionError("SFT/GRPO prompt or proxy alignment differs")
            for record in (sft, grpo):
                audit_window_record(record, check_media_files=not args.skip_media_file_check)
            outputs["train_sft"].append(sft); outputs["train_grpo"].append(grpo)
            selected_window_counts[sft["request_id"]] += 1
            pool.append({"pool_instance_id": instance_id, "request_id": sft["request_id"],
                         "dataset_request_id": sft["dataset_request_id"],
                         "video_id": vid, "clip_ids": ids, "media_signature": sft["media_signature"],
                         "pseudo_sha256": canonical_hash(sft["pseudo_scores"])})
    for name in ("eval", "eval_train_diagnostic"):
        ids = [r["request_id"] for r in outputs[name]]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation request IDs are not unique")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in outputs.items():
        with (args.out / f"{name}.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.out / "window_pool.jsonl").open("w") as f:
        for record in pool:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    hashes = {path.name: file_hash(path) for path in sorted(args.out.glob("*.jsonl"))}
    report = {"schema_version": 1, "status": "COMPLETED", "seed": args.seed,
              "windows_per_training_video": args.windows_per_video,
              "sampling": "sorted videos; NumPy default_rng(seed); uniform candidate-index draws with replacement",
              "candidate_windows": "width 8, stride 4, forced complete end coverage; no padding clips",
              "training_request_id_duplicates": "intentional same-window repeats; pool_instance_id distinguishes sampled instances",
              "stable_dataset_identity_field": "dataset_request_id equals source request_id; preserve it when ms-swift 4.6 overwrites request_id with rollout completion UUID",
              "exposure_note": "pool counts are not actual training exposures; trainers must log sampler visits and successful updates",
              "split_audit": split_audit, "media_files_checked": not args.skip_media_file_check,
              "human_highlight_labels_used": False,
              "counts": {name: len(records) for name, records in outputs.items()},
              "per_training_video": counts,
              "selected_request_id_repetitions": dict(sorted(selected_window_counts.items())),
              "unique_training_windows": len(selected_window_counts),
              "hashes": {"prepared_manifest": file_hash(args.manifest), "pseudo_json": file_hash(args.pseudo), **hashes}}
    write_json(args.out / "data_audit.json", report)
    (args.out / "data.sha256").write_text("".join(f"{h}  {name}\n" for name, h in sorted(hashes.items())))
    if args.report:
        write_json(args.report, report)
    print(json.dumps({"status": "COMPLETED", "out": str(args.out), "counts": report["counts"],
                      "unique_training_windows": len(selected_window_counts), "hashes": hashes}))


if __name__ == "__main__":
    main()
