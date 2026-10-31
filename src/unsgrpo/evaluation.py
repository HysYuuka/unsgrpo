"""Independent evaluation; never import this module from training code.

V0 is the project's registered diagnostic, not TVSum paper top-5 mAP.
All public metric values use [0, 1], with percentages reserved for reporting.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

RUN_IDS = ("B0", "B1", "S0", "S+", "G-CAL", "G-RANK")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path):
    """Read a JSON list, {videos: list}, one JSON record, or JSONL records."""
    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("videos"), list):
        return value["videos"]
    if isinstance(value, dict) and "video_id" in value:
        return [value]
    raise ValueError(f"unsupported records schema: {path}")


def validate_six_run_freeze(freeze_path):
    """Verify all six full-video predictions before any annotation I/O.

    Freeze schema: {manifest: {path, sha256}, runs: {run: {path, sha256}}}.
    Paths may be absolute or relative to the freeze file's parent directory.
    """
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_text())
    aliases = {"S_PLUS": "S+", "G_CAL": "G-CAL", "G_RANK": "G-RANK"}
    runs = {}
    for key, entry in freeze.get("runs", {}).items():
        key = aliases.get(key, key)
        if key in runs:
            raise ValueError(f"duplicate canonical run ID: {key}")
        runs[key] = entry
    if set(runs) != set(RUN_IDS):
        raise ValueError(f"freeze requires exactly six runs: {RUN_IDS}")

    def checked_file(entry):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("freeze entry needs path and sha256")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = freeze_path.parent / path
        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or file_sha256(path) != expected:
            raise ValueError(f"frozen file hash mismatch: {path}")
        return path.resolve()

    manifest_path = checked_file(freeze.get("manifest"))
    manifest = load_records(manifest_path)
    test = [row for row in manifest if row.get("split") == "test"]
    if not test:
        raise ValueError("frozen manifest contains no test videos")
    expected_ids = [row["video_id"] for row in test]
    if any(not isinstance(video_id, str) or not video_id for video_id in expected_ids):
        raise ValueError("test video IDs must be nonempty strings")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("duplicate test video IDs in manifest")
    if any(not row.get("source_group") for row in test):
        raise ValueError("every test video needs a source_group")
    # Source groups must not cross split boundaries in the complete manifest.
    group_splits = defaultdict(set)
    for row in manifest:
        group_splits[row["source_group"]].add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("source_group crosses manifest split boundaries")

    result = {}
    for run in RUN_IDS:
        path = checked_file(runs[run])
        rows = load_records(path)
        by_id = {}
        for row in rows:
            video_id = row.get("video_id")
            if video_id in by_id:
                raise ValueError(f"duplicate video prediction in {run}: {video_id}")
            by_id[video_id] = row
        if set(by_id) != set(expected_ids):
            raise ValueError(f"full-video prediction coverage mismatch in {run}")
        for video in test:
            pred = by_id[video["video_id"]]
            bounds = np.asarray(pred["bounds"], dtype=np.float64)
            registered = np.asarray(video["bounds"], dtype=np.float64)
            if (registered.ndim != 2 or registered.shape[1] != 2 or not len(registered)
                    or not np.isfinite(registered).all()
                    or not np.isclose(registered[0, 0], 0, rtol=0, atol=1e-6)
                    or not np.isclose(registered[-1, 1], video["duration"], rtol=0, atol=1e-6)
                    or not np.allclose(registered[1:, 0], registered[:-1, 1], rtol=0, atol=1e-6)):
                raise ValueError(f"registered intervals do not cover complete video: {video['video_id']}")
            if bounds.shape != registered.shape or not np.allclose(bounds, registered, rtol=0, atol=1e-6):
                raise ValueError(f"prediction interval mismatch in {run}/{video['video_id']}")
            # Identity mapping checks score shape, finite values, disjoint bins
            # and coverage without opening any human annotation.
            interval_average(bounds, pred["scores"], registered)
        provenance = None
        if freeze.get("require_run_provenance") is True:
            receipt_path = checked_file(runs[run].get("receipt"))
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("status") != "COMPLETED" or receipt.get("run_id") != run
                    or receipt.get("predictions_sha256") != runs[run]["sha256"]
                    or receipt.get("manifest_sha256") != freeze["manifest"]["sha256"]
                    or Path(receipt.get("predictions_path", "")).resolve() != path
                    or receipt.get("human_annotations_read") is not False
                    or receipt.get("n_videos") != len(test)):
                raise ValueError(f"inference receipt does not bind completed {run} predictions to the frozen manifest")
            if run in {"S0", "S+", "G-CAL", "G-RANK"}:
                training_path = checked_file(receipt.get("training_receipt"))
                training = json.loads(training_path.read_text())
                if (training.get("status") != "COMPLETED" or training.get("mode") != "formal"
                        or training.get("gpu_training_verified") is not True or training.get("method_id") != run):
                    raise ValueError(f"{run} is not bound to a verified formal training endpoint")
                expected_exposures = 4096 if run == "S0" else 1024
                expected_rollouts = 4096 if run in {"G-CAL", "G-RANK"} else 0
                if (training.get("n_successful_optimizer_updates") != 256
                        or training.get("n_prompt_exposures") != expected_exposures
                        or training.get("n_rollout_completions") != expected_rollouts):
                    raise ValueError(f"{run} formal training budget is incomplete or differs from registration")
                if run != "S0" and training.get("common_s0_sha256") != receipt.get("common_s0_sha256"):
                    raise ValueError(f"{run} inference and training use different common S0 identities")
                endpoint = training.get("endpoint", {})
                root = Path(endpoint.get("path", ""))
                if not root.is_dir() or not endpoint.get("files"):
                    raise ValueError(f"{run} formal endpoint file inventory is absent")
                for name, digest in endpoint["files"].items():
                    target = (root / name).resolve()
                    if not target.is_relative_to(root.resolve()) or file_sha256(target) != digest:
                        raise ValueError(f"{run} formal endpoint file changed: {name}")
            provenance = {"receipt_path": str(receipt_path), "receipt_sha256": runs[run]["receipt"]["sha256"],
                          "run_id": receipt["run_id"]}
        result[run] = {"path": str(path), "sha256": runs[run]["sha256"], "videos": by_id,
                       "inference_provenance": provenance}
    if freeze.get("require_run_provenance") is True:
        common_ids = set()
        for run in ("S0", "S+", "G-CAL", "G-RANK"):
            receipt = json.loads(Path(result[run]["inference_provenance"]["receipt_path"]).read_text())
            identity = receipt.get("common_s0_sha256")
            if not isinstance(identity, str) or len(identity) != 64:
                raise ValueError(f"{run} inference does not identify the common S0")
            common_ids.add(identity)
        if len(common_ids) != 1:
            raise ValueError("trained inference runs do not share one common S0")
    return {
        "freeze_path": str(freeze_path),
        "freeze_sha256": file_sha256(freeze_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": freeze["manifest"]["sha256"],
        "test_videos": test,
        "runs": result,
        "require_run_provenance": freeze.get("require_run_provenance") is True,
    }


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def interval_average(source_bounds, scores, target_bounds, *, atol=1e-6):
    """Integrate a piecewise-constant prediction over real annotation intervals.

    Source bins must be ordered and disjoint. Every target interval needs full
    coverage: gaps, extrapolation and renormalization of partial coverage fail.
    """
    src = np.asarray(source_bounds, dtype=np.float64)
    dst = np.asarray(target_bounds, dtype=np.float64)
    values = _vector(scores, "scores")
    if src.shape != (len(values), 2) or dst.ndim != 2 or dst.shape[1] != 2 or not len(dst):
        raise ValueError("invalid interval shapes")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("non-finite intervals")
    if np.any(src[:, 1] <= src[:, 0]) or np.any(dst[:, 1] <= dst[:, 0]):
        raise ValueError("nonpositive interval duration")
    if np.any(src[1:, 0] < src[:-1, 1] - atol):
        raise ValueError("source intervals overlap or are out of order")
    if np.any(dst[1:, 0] < dst[:-1, 1] - atol):
        raise ValueError("target intervals overlap or are out of order")
    overlap = np.maximum(
        0.0,
        np.minimum(dst[:, None, 1], src[None, :, 1])
        - np.maximum(dst[:, None, 0], src[None, :, 0]),
    )
    mass = overlap.sum(axis=1)
    lengths = dst[:, 1] - dst[:, 0]
    if np.any(mass <= 0) or not np.allclose(mass, lengths, rtol=1e-8, atol=atol):
        raise ValueError("partial evaluation coverage; fix the time axis")
    return overlap @ values / lengths


def average_precision_noninterpolated(labels, scores) -> float:
    """Sklearn-style AP: a tied score group enters at one threshold.

    This avoids arbitrary temporal ordering inside score ties. If no positives
    exist the registered result is zero; callers also report their count.
    """
    truth = np.asarray(labels)
    pred = _vector(scores, "scores")
    if truth.shape != pred.shape or not np.isin(truth, [0, 1]).all():
        raise ValueError("binary labels and scores must have identical shapes")
    truth = truth.astype(np.int64)
    positives = int(truth.sum())
    if not positives:
        return 0.0
    order = np.argsort(-pred, kind="stable")
    sorted_pred, sorted_truth = pred[order], truth[order]
    ends = np.r_[np.flatnonzero(sorted_pred[1:] != sorted_pred[:-1]), len(pred) - 1]
    true_positive = np.cumsum(sorted_truth)[ends]
    recall = true_positive / positives
    precision = true_positive / (ends + 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def midranks(values):
    """One-based average ranks, preserving exact ties."""
    arr = _vector(values, "values")
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.float64)
    left = 0
    while left < len(arr):
        right = left + 1
        while right < len(arr) and arr[order[right]] == arr[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + 1 + right) / 2
        left = right
    return ranks


def spearman_with_status(first, second) -> dict:
    """Tied midrank Spearman; undefined constant cases are explicit nulls."""
    first, second = _vector(first, "first"), _vector(second, "second")
    if first.shape != second.shape:
        raise ValueError("Spearman vectors must have identical shapes")
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"value": None, "defined": False, "reason": "constant_or_singleton"}
    return {
        "value": float(np.corrcoef(midranks(first), midranks(second))[0, 1]),
        "defined": True,
        "reason": None,
    }


def evaluate_v0(prediction_bounds, scores, annotation_bounds, ratings) -> dict:
    """One video's V0: per-user strict-above-median AP, then user mean.

    ``ratings`` must be explicitly converted to [annotation_interval, user].
    Frame-level repetition in TVSum's MAT file must not silently be treated as
    two-second rows: conversion is a separately audited operation.
    """
    human = np.asarray(ratings, dtype=np.float64)
    bounds = np.asarray(annotation_bounds, dtype=np.float64)
    if human.ndim != 2 or human.shape[0] != len(bounds) or not human.shape[1]:
        raise ValueError("ratings must have axes [interval, annotator]")
    if not np.isfinite(human).all():
        raise ValueError("non-finite ratings")
    mapped = interval_average(prediction_bounds, scores, bounds)
    thresholds = np.median(human, axis=0)
    binary = human > thresholds[None, :]
    aps = [average_precision_noninterpolated(binary[:, user], mapped)
           for user in range(human.shape[1])]
    return {
        "protocol": "V0-strict-median-noninterpolated-AP-v1",
        "mean_ap": float(np.mean(aps)),
        "ap_by_annotator": aps,
        "n_annotators": human.shape[1],
        "n_intervals": len(bounds),
        "n_no_positive_annotators": int((binary.sum(axis=0) == 0).sum()),
        "spearman": spearman_with_status(mapped, human.mean(axis=1)),
    }


def evaluate_top5_umt(scores, ratings, *, k=5) -> dict:
    """Source-bound UMT TVSum evaluator component, on a caller-audited time axis.

    Source: TencentARC/UMT datasets/tvsum.py, evaluate(), pinned in protocol.
    CPU torch.argsort(descending=True, stable=False) matches the source's default
    tie behavior on the recorded Torch build. This metric uses trapezoidal PR
    integration and positives *inside* the top k as recall denominator. It must
    never be renamed to sklearn AP@5. Real benchmark comparability additionally
    requires matching annotation clips, dataset split and aggregation protocol.
    """
    import torch

    pred = _vector(scores, "scores")
    human = np.asarray(ratings, dtype=np.float64)
    if human.ndim != 2 or human.shape != (len(pred), 20) or not np.isfinite(human).all():
        raise ValueError("UMT TVSum ratings must have axes [clip, 20 annotators]")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    # The upstream saliency outputs and annotation Tensor are float32.
    order = torch.argsort(torch.as_tensor(pred, dtype=torch.float32), descending=True)
    human_t = torch.as_tensor(human, dtype=torch.float32)
    aps = []
    no_positive = 0
    for user in range(20):
        original = human_t[:, user]
        binary = (original > original.median()).to(torch.int64)[order][:k].tolist()
        positives = sum(binary)
        if not positives:
            aps.append(0.0)
            no_positive += 1
            continue
        hits = 0
        previous_recall, previous_precision, area = 0.0, 1.0, 0.0
        for rank, relevant in enumerate(binary, start=1):
            hits += relevant
            recall, precision = hits / positives, hits / rank
            area += (recall - previous_recall) * (previous_precision + precision) * 0.5
            previous_recall, previous_precision = recall, precision
        aps.append(float(area))
    return {
        "protocol": "V1-UMT-top5-evaluator-component",
        "mean_ap": float(np.mean(aps)),
        "ap_by_annotator": aps,
        "n_no_positive_topk_annotators": no_positive,
        "k": k,
        "tie_backend": f"torch-{torch.__version__}-cpu-argsort-stable-false-float32",
        "benchmark_alignment_verified": False,
    }


def validate_prediction_records(expected_request_ids: Sequence[str], records: Iterable[Mapping]) -> dict:
    """Reject duplicates, missing/extra IDs, and non-string identifiers.

    Invalid generated answers are valid *records*: they remain in evaluation and
    are handled by aggregate_windows. This check enforces run completeness only.
    """
    expected = list(expected_request_ids)
    if not expected or any(not isinstance(x, str) or not x for x in expected):
        raise ValueError("expected request IDs must be nonempty strings")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected request IDs")
    by_id = {}
    for record in records:
        rid = record.get("request_id")
        if not isinstance(rid, str) or not rid:
            raise ValueError("prediction record has no valid request_id")
        if rid in by_id:
            raise ValueError(f"duplicate prediction request_id: {rid}")
        by_id[rid] = record
    missing = sorted(set(expected) - by_id.keys())
    extra = sorted(by_id.keys() - set(expected))
    if missing or extra:
        raise ValueError(f"prediction coverage mismatch: missing={missing}, extra={extra}")
    return {rid: by_id[rid] for rid in expected}


def paired_bootstrap(differences, n_boot=10000, seed=42, source_groups=None) -> dict:
    """Equal-source-group paired percentile bootstrap, excluding training variance.

    If multiple videos share a source group, first average their differences;
    then sample groups with replacement. The point estimate has the same equal-
    group weighting. Report video macro means separately when they differ.
    """
    d = _vector(differences, "differences")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    groups = list(source_groups) if source_groups is not None else list(range(len(d)))
    if len(groups) != len(d):
        raise ValueError("one source group is required per video")
    grouped = defaultdict(list)
    for group, difference in zip(groups, d):
        if group is None or group == "":
            raise ValueError("empty source group")
        grouped[group].append(difference)
    group_means = np.array([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        samples[i] = rng.choice(group_means, size=len(group_means), replace=True).mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(group_means.mean()),
        "video_macro_mean": float(d.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_videos": len(d),
        "n_source_groups": len(group_means),
        "n_boot": n_boot,
        "seed": seed,
        "unit": "source_group",
        "training_seed_uncertainty_included": False,
    }
