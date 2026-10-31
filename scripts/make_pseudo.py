#!/usr/bin/env python3
"""Fit train-only UMAP/K-means and materialize the fixed visual R0 proxy.

No fallback to PCA, manual classes, or evaluation labels is allowed. A failed
clustering is a failed prerequisite, not permission to change the proxy silently.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unsgrpo.core import midrank_percentile, recurrence_loo_video, unit_rows
from unsgrpo.data import audit_split, window_indices


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def summary(x) -> dict:
    a = np.asarray(x, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("empty or nonfinite diagnostic")
    return {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std_population": float(a.std()),
            "p05": float(np.quantile(a, .05)), "p50": float(np.quantile(a, .5)),
            "p95": float(np.quantile(a, .95))}


def tie_summary(x) -> dict:
    a = np.asarray(x)
    _, counts = np.unique(a, return_counts=True)
    pairs = len(a) * (len(a) - 1) // 2
    return {"n_values": len(a), "n_distinct": len(counts),
            "duplicate_value_fraction": float(1 - len(counts) / len(a)),
            "pair_tie_fraction": float(np.sum(counts * (counts - 1) / 2) / pairs) if pairs else 0.0}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(project):
        parser.error("pseudo assets and fitted models must be stored outside the code repository")
    if any((args.out / name).exists() for name in ("pseudo.json", "cluster_model.joblib", "diagnostics.json", "window_weights.jsonl")):
        parser.error("output artifacts already exist; use a new versioned directory")
    started = time.monotonic()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    split_audit = audit_split(rows)
    rows = sorted(rows, key=lambda row: row["video_id"])
    train = [r for r in rows if r["split"] == "train"]
    heldout = [r for r in rows if r["split"] != "train"]
    if len(train) < 5:
        raise ValueError("K search from 4 needs at least 5 training videos; repair the data collection")
    feature_map, feature_hashes = {}, {}
    for row in rows:
        path = Path(row["features_path"])
        features = np.load(path, allow_pickle=False)
        if len(features) != len(row["bounds"]):
            raise ValueError(f"feature/time-axis clip count mismatch: {row['video_id']}")
        feature_map[row["video_id"]] = unit_rows(features)
        feature_hashes[row["video_id"]] = file_hash(path)
    dims = {f.shape[1] for f in feature_map.values()}
    if len(dims) != 1:
        raise ValueError("feature dimensions are inconsistent")
    train_ids = [r["video_id"] for r in train]
    video_features = unit_rows(np.stack([feature_map[k].mean(axis=0) for k in train_ids]))
    if len(np.unique(video_features, axis=0)) < 4:
        raise ValueError("degenerate video embeddings; need at least four distinct training videos")

    import joblib
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import umap

    umap_config = {"n_neighbors": min(15, len(train) - 1),
                   "n_components": min(5, len(train) - 2), "metric": "cosine",
                   "min_dist": .1, "random_state": args.seed,
                   "transform_seed": args.seed, "n_jobs": 1}
    reducer = umap.UMAP(**umap_config)
    embedded_train = reducer.fit_transform(video_features)
    if not np.isfinite(embedded_train).all() or np.max(np.ptp(embedded_train, axis=0)) < 1e-8:
        raise ValueError("nonfinite or collapsed UMAP embedding")
    candidates, best_model, best_score, best_k = [], None, -np.inf, None
    for k in range(4, min(15, len(train) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=args.seed, algorithm="lloyd")
        assignment = model.fit_predict(embedded_train)
        sizes = dict(sorted(collections.Counter(map(int, assignment)).items()))
        candidate = {"k": k, "cluster_sizes": sizes, "inertia": float(model.inertia_)}
        if len(sizes) != k:
            candidate.update(status="FAILED", reason="fewer distinct clusters than requested")
        else:
            score = float(silhouette_score(embedded_train, assignment, metric="euclidean"))
            if not np.isfinite(score):
                raise ValueError("nonfinite silhouette score")
            candidate.update(status="COMPLETED", silhouette=score)
            # Ascending K and strict > resolve exact silhouette ties in favor of smaller K.
            if score > best_score:
                best_model, best_score, best_k = model, score, k
        candidates.append(candidate)
    if best_model is None:
        raise ValueError("no valid K candidate; stop rather than substitute another clustering")
    assignment = best_model.predict(embedded_train)
    clusters = {k: int(c) for k, c in zip(train_ids, assignment)}
    embedded_heldout = None
    if heldout:
        heldout_features = unit_rows(np.stack([feature_map[r["video_id"]].mean(axis=0) for r in heldout]))
        embedded_heldout = reducer.transform(heldout_features)
        if not np.isfinite(embedded_heldout).all():
            raise ValueError("nonfinite held-out UMAP transform")
        clusters.update({r["video_id"]: int(c) for r, c in zip(heldout, best_model.predict(embedded_heldout))})
    banks, bank_sizes, cluster_videos = {}, {}, {}
    for c in range(best_k):
        ids = [k for k in train_ids if clusters[k] == c]
        bank = np.concatenate([feature_map[k] for k in ids], axis=0)
        banks[c] = bank.mean(axis=0)  # Do NOT unit-normalize the prototype.
        bank_sizes[c] = len(bank)
        cluster_videos[c] = ids
    train_features = {k: feature_map[k] for k in train_ids}
    train_clusters = {k: clusters[k] for k in train_ids}
    scores, per_video, window_diagnostics, raw_all, q_all, integer_all = {}, [], [], [], [], []
    for row in rows:
        vid = row["video_id"]
        f, c = feature_map[vid], clusters[vid]
        raw = f @ banks[c]
        q = midrank_percentile(raw)
        if not np.isclose(q.mean(), .5, atol=1e-12):
            raise AssertionError("whole-video percentile mean is not 0.5")
        scores[vid] = {"split": row["split"], "cluster": c, "raw": raw.tolist(), "q": q.tolist(),
                       "features_sha256": feature_hashes[vid]}
        if row["split"] != "train":
            continue
        integer = np.rint(q * 100).astype(int)
        loo, loo_audit = recurrence_loo_video(vid, train_features, train_clusters)
        order = np.argsort(raw, kind="stable")
        diag = {"video_id": vid, "source_group": row["source_group"], "n_clips": len(f),
                "cluster": c, "cluster_video_count": len(cluster_videos[c]),
                "self_clip_fraction_of_cluster": len(f) / bank_sizes[c],
                "raw": summary(raw), "raw_span": float(np.ptp(raw)), "q": summary(q),
                "q_ties": tie_summary(q), "integer_target_ties": tie_summary(integer),
                "r0_minus_video_equal_loo": summary(raw - loo),
                "r0_vs_loo_pearson": (float(np.corrcoef(raw, loo)[0, 1])
                                      if np.std(raw) > 0 and np.std(loo) > 0 else None),
                "loo_global_fallback": loo_audit["global_fallback"], "loo_peer_count": len(loo_audit["peers"]),
                "low_raw_clips": [{"clip_id": int(i), "bounds": row["bounds"][int(i)], "raw": float(raw[i]), "q": float(q[i])}
                                  for i in order[:3]],
                "high_raw_clips": [{"clip_id": int(i), "bounds": row["bounds"][int(i)], "raw": float(raw[i]), "q": float(q[i])}
                                   for i in order[-3:][::-1]]}
        per_video.append(diag)
        for ids in window_indices(len(f)):
            local = q[ids]
            ii, jj = np.triu_indices(len(ids), k=1)
            weights = np.abs(local[ii] - local[jj])
            window_diagnostics.append({"request_id": f"{vid}__{ids[0]:06d}_{ids[-1]:06d}",
                                       "n_clips": len(ids), "n_pairs": len(weights),
                                       "positive_weight_pairs": int(np.count_nonzero(weights)),
                                       "W": float(weights.sum()),
                                       "max_pair_weight": float(weights.max()) if len(weights) else 0.0})
        raw_all.extend(raw.tolist()); q_all.extend(q.tolist()); integer_all.extend(integer.tolist())

    args.out.mkdir(parents=True, exist_ok=True)
    # This object is a trusted local artifact; never load untrusted joblib files.
    joblib.dump({"umap": reducer, "kmeans": best_model, "prototypes": banks,
                 "train_ids": train_ids, "train_clusters": train_clusters,
                 "train_feature_sha256": {k: feature_hashes[k] for k in train_ids},
                 "umap_config": umap_config}, args.out / "cluster_model.joblib", compress=3)
    protocol = {"method": "visual_R0_mean_cosine", "embedding": "umap",
                "seed": args.seed, "umap_config": umap_config,
                "kmeans_config": {"k_min": 4, "k_max": min(15, len(train) - 1),
                                   "n_init": 20, "algorithm": "lloyd", "random_state": args.seed},
                "selected_k": best_k, "selected_silhouette": best_score,
                "fit_video_ids": train_ids, "heldout_operation": "UMAP.transform then KMeans.predict only",
                "reference_bank": "all clips of training videos in predicted cluster, including self during train",
                "prototype_l2_renormalized": False,
                "quantization": "numpy.rint(q*100), ties to nearest even integer",
                "percentiles": "whole-video exact midrank; equal raw floats remain tied",
                "versions": {"numpy": np.__version__, "umap_learn": umap.__version__,
                             "sklearn": sklearn.__version__, "joblib": joblib.__version__}}
    pseudo = {"schema_version": 1, "protocol": protocol,
              "prepared_manifest_sha256": file_hash(args.manifest), "videos": scores}
    write_json(args.out / "pseudo.json", pseudo)
    diagnostics = {"schema_version": 1, "status": "COMPLETED",
                   "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "train_only_diagnostics": True, "used_human_highlight_labels": False,
                   "protocol": protocol, "split_audit": split_audit, "k_search": candidates,
                   "clusters": [{"cluster": c, "video_ids": cluster_videos[c],
                                 "video_count": len(cluster_videos[c]), "clip_count": bank_sizes[c]}
                                for c in range(best_k)],
                   "singleton_cluster_count": sum(len(x) == 1 for x in cluster_videos.values()),
                   "raw": summary(raw_all), "q": summary(q_all),
                   "integer_target_ties_across_training_clips": tie_summary(integer_all),
                   "window_W": summary([x["W"] for x in window_diagnostics]),
                   "zero_weight_window_count": sum(x["W"] == 0 for x in window_diagnostics),
                   "per_training_video": per_video, "elapsed_seconds": time.monotonic() - started,
                   "hashes": {"prepared_manifest": file_hash(args.manifest),
                              "pseudo_json": file_hash(args.out / "pseudo.json"),
                              "cluster_model_joblib": file_hash(args.out / "cluster_model.joblib"),
                              "features": feature_hashes}}
    write_json(args.out / "diagnostics.json", diagnostics)
    with (args.out / "window_weights.jsonl").open("w") as f:
        for diag in window_diagnostics:
            f.write(json.dumps(diag, allow_nan=False) + "\n")
    if args.report:
        write_json(args.report, diagnostics)
    print(json.dumps({"status": "COMPLETED", "out": str(args.out), "selected_k": best_k,
                      "train_videos": len(train), "heldout_videos": len(heldout),
                      "pseudo_sha256": diagnostics["hashes"]["pseudo_json"]}))


if __name__ == "__main__":
    main()
