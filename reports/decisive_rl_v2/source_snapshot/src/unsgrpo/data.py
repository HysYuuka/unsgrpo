"""Time, split, prompt and rollout identity contracts; no evaluation labels accepted.

A media_signature binds paths and recorded timestamps. It is not a claim of equal
processed tensors: the GPU integration must supply media_tensor_sha256 as well.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .core import parse_scores


def clip_intervals(duration: float, seconds: float = 2.0) -> np.ndarray:
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration and clip seconds must be finite and positive")
    starts = np.arange(math.ceil(duration / seconds), dtype=np.float64) * seconds
    return np.column_stack((starts, np.minimum(starts + seconds, duration)))


def sample_targets(bounds: Sequence[Sequence[float]], frames_per_clip: int = 4) -> np.ndarray:
    b = np.asarray(bounds, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2 or not len(b) or not np.isfinite(b).all() or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("expected finite positive clip intervals")
    if type(frames_per_clip) is not int or frames_per_clip < 1:
        raise ValueError("invalid frame count")
    return b[:, :1] + (b[:, 1:] - b[:, :1]) * ((np.arange(frames_per_clip) + .5) / frames_per_clip)


def window_indices(n_clips: int, width: int = 8, stride: int = 4) -> list[list[int]]:
    if any(type(x) is not int for x in (n_clips, width, stride)) or n_clips < 1 or width < 1 or not 1 <= stride <= width:
        raise ValueError("invalid window parameters")
    if n_clips <= width:
        return [list(range(n_clips))]
    starts = list(range(0, n_clips - width + 1, stride))
    if starts[-1] != n_clips - width:
        starts.append(n_clips - width)
    return [list(range(s, s + width)) for s in starts]


def split_by_source_group(rows: Sequence[Mapping], train_fraction: float = .8, seed: int = 42) -> list[dict]:
    """Seeded split before windows; title/category/labels have no role in assignment."""
    if not rows or not 0 < train_fraction < 1:
        raise ValueError("need rows and a train fraction in (0,1)")
    ids, groups, hashes = set(), set(), {}
    for row in rows:
        vid, group = row.get("video_id"), row.get("source_group")
        if not isinstance(vid, str) or not vid or vid in ids or not isinstance(group, str) or not group:
            raise ValueError("unique nonempty video IDs and source groups required")
        ids.add(vid)
        groups.add(group)
        digest = row.get("sha256")
        if digest:
            if digest in hashes and hashes[digest] != group:
                raise ValueError("identical assets have different source groups")
            hashes[digest] = group
    if len(groups) < 2:
        raise ValueError("at least two independent source groups required")
    ordered = np.asarray(sorted(groups), dtype=object)
    shuffled = np.random.default_rng(seed).permutation(ordered)
    n_train = min(len(groups) - 1, max(1, int(len(groups) * train_fraction)))
    train = set(shuffled[:n_train])
    result = [dict(row, split="train" if row["source_group"] in train else "test") for row in rows]
    audit_split(result)
    return result


def audit_split(rows: Sequence[Mapping]) -> dict:
    ids, assignments = set(), {}
    for row in rows:
        vid, group, split = row["video_id"], row["source_group"], row["split"]
        if vid in ids or not group or split not in {"train", "test", "dev"}:
            raise ValueError("duplicate ID, missing source group, or unknown split")
        if group in assignments and assignments[group] != split:
            raise ValueError("source_group crosses splits")
        assignments[group] = split
        ids.add(vid)
    return {"n_videos": len(ids), "n_source_groups": len(assignments),
            "counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "test", "dev")}}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prompt(clip_ids, bounds, duration) -> list[dict]:
    lines = [f"{i}: [{a:.6f}, {b:.6f}) seconds" for i, (a, b) in zip(clip_ids, bounds)]
    return [
        {"role": "system", "content": "You score video highlights without a text query. Return only valid JSON with the single key scores. Scores must be integers from 0 to 100, one per listed clip in its given order, describing relative highlight strength within the complete video. Do not add explanations."},
        {"role": "user", "content": "<image>\nThe image is a 16-cell chronological overview of the complete video.\n<video>\nThe video contains four sampled frames for each listed consecutive clip. "
         f"Complete video duration: {duration:.6f} seconds.\nClip IDs and original time intervals:\n" + "\n".join(lines)},
    ]


def build_window_record(*, video_id: str, source_group: str, split: str,
                        duration: float, bounds: Sequence[Sequence[float]],
                        frame_paths: Sequence[Sequence[str]], sampled_pts: Sequence[Sequence[float]],
                        overview_path: str, clip_ids: Sequence[int],
                        pseudo_scores: Sequence[float] | None = None,
                        task: str = "grpo") -> dict:
    """Build swift-compatible image + video-frame-list record.

    bounds, frame_paths, sampled_pts and pseudo_scores refer to the COMPLETE video.
    clip_ids index them and preserve the same q for overlapping windows. Human
    category, title and evaluation labels are deliberately absent from the API.
    """
    b = np.asarray(bounds, dtype=np.float64)
    pts = np.asarray(sampled_pts, dtype=np.float64)
    ids = list(clip_ids)
    if task not in {"sft", "grpo", "eval"} or split not in {"train", "test", "dev"}:
        raise ValueError("invalid task or split")
    if task != "eval" and split != "train":
        raise ValueError("training records must come from the train split")
    if not video_id or not source_group or not np.isfinite(duration) or duration <= 0:
        raise ValueError("missing identity or invalid duration")
    if b.ndim != 2 or b.shape[1:] != (2,) or not len(b) or not np.isfinite(b).all():
        raise ValueError("invalid full-video bounds")
    if np.any(b[:, 1] <= b[:, 0]) or not np.isclose(b[0, 0], 0) or not np.isclose(b[-1, 1], duration) or not np.allclose(b[1:, 0], b[:-1, 1]):
        raise ValueError("full-video intervals must cover exactly [0,duration)")
    if not ids or any(type(i) is not int for i in ids) or ids != list(range(ids[0], ids[0] + len(ids))) or min(ids) < 0 or max(ids) >= len(b) or len(ids) > 8:
        raise ValueError("clip IDs must be consecutive valid indices, at most eight")
    if len(frame_paths) != len(b) or any(len(x) != 4 or any(not isinstance(p, str) or not p for p in x) for x in frame_paths):
        raise ValueError("four explicit frames are required for every full-video clip")
    if pts.shape != (len(b), 4) or not np.isfinite(pts).all() or np.any(np.diff(pts, axis=1) < 0):
        raise ValueError("invalid actual PTS layout")
    # First-after-target decoding can fall just beyond a clip boundary, but never
    # beyond the whole video; target errors are audited separately by the decoder.
    if np.any(pts < 0) or np.any(pts >= duration):
        raise ValueError("actual PTS lies outside the complete video")
    if not isinstance(overview_path, str) or not overview_path:
        raise ValueError("missing overview")
    messages = _prompt(ids, b[ids], duration)
    result = {
        "request_id": f"{video_id}__{ids[0]:06d}_{ids[-1]:06d}", "video_id": video_id,
        "source_group": source_group, "split": split, "task": task,
        "duration": float(duration), "clip_ids": ids, "clip_bounds": b[ids].tolist(),
        "sampled_pts": pts[ids].tolist(), "images": [overview_path],
        "videos": [[p for i in ids for p in frame_paths[i]]], "messages": messages,
    }
    result["media_signature"] = canonical_hash({k: result[k] for k in ("images", "videos", "clip_ids", "clip_bounds", "sampled_pts")})
    if task != "eval":
        q = np.asarray(pseudo_scores, dtype=np.float64)
        if q.shape != (len(b),) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("pseudo scores must cover the entire video")
        result["pseudo_scores"] = q[ids].tolist()
        if task == "sft":
            answer = json.dumps({"scores": np.rint(q[ids] * 100).astype(int).tolist()}, separators=(",", ":"))
            messages.append({"role": "assistant", "content": answer})
    elif pseudo_scores is not None:
        raise ValueError("evaluation model inputs must not carry pseudo answers")
    audit_window_record(result)
    return result


def audit_window_record(row: Mapping, check_media_files: bool = False) -> dict:
    """Reject unexpected prompt text, answer leakage and media/clip reassignment."""
    ids = row["clip_ids"]
    if not ids or any(type(i) is not int for i in ids) or ids != list(range(ids[0], ids[0] + len(ids))) or len(ids) > 8:
        raise ValueError("invalid clip order")
    if row["request_id"] != f'{row["video_id"]}__{ids[0]:06d}_{ids[-1]:06d}':
        raise ValueError("request ID does not bind its video/window")
    expected = _prompt(ids, row["clip_bounds"], row["duration"])
    if row["messages"][:2] != expected:
        raise ValueError("unexpected prompt text or pseudo/GT metadata leakage")
    if len(row["images"]) != 1 or len(row["videos"]) != 1 or len(row["videos"][0]) != len(ids) * 4:
        raise ValueError("media placeholder/frame count mismatch")
    if np.asarray(row["sampled_pts"]).shape != (len(ids), 4):
        raise ValueError("PTS/clip mismatch")
    signature = canonical_hash({k: row[k] for k in ("images", "videos", "clip_ids", "clip_bounds", "sampled_pts")})
    if signature != row["media_signature"]:
        raise ValueError("media signature mismatch")
    task = row["task"]
    if task in {"sft", "grpo"}:
        if row["split"] != "train":
            raise ValueError("training split leakage")
        q = np.asarray(row["pseudo_scores"], dtype=np.float64)
        if q.shape != (len(ids),) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("pseudo/clip mismatch")
    if task == "sft":
        if len(row["messages"]) != 3 or row["messages"][-1]["role"] != "assistant":
            raise ValueError("SFT must have exactly one assistant target")
        answer = parse_scores(row["messages"][-1]["content"], len(ids))
        if not np.array_equal(answer, np.rint(q * 100) / 100):
            raise ValueError("SFT assistant target does not correspond to q/clip IDs")
    elif task in {"grpo", "eval"}:
        if len(row["messages"]) != 2:
            raise ValueError("GRPO/eval must not contain assistant answers")
        if task == "eval" and "pseudo_scores" in row:
            raise ValueError("evaluation input contains pseudo answers")
    else:
        raise ValueError("unknown task")
    forbidden = {"gt", "labels", "ratings", "category", "title", "query"}.intersection(row)
    if forbidden:
        raise ValueError(f"forbidden training/model-input fields: {sorted(forbidden)}")
    if check_media_files:
        for p in row["images"] + row["videos"][0]:
            if not Path(p).is_file():
                raise ValueError(f"missing media file: {p}")
    return {"request_id": row["request_id"], "n_clips": len(ids), "n_frames": 4 * len(ids), "media_signature": signature}


def validate_rollout_groups(rows: Sequence[Mapping], group_size: int = 4, *, require_tensor_hash: bool = False) -> dict:
    """Rows MUST be prompt-major; compare content as well as metadata lengths."""
    if group_size < 2 or not rows or len(rows) % group_size:
        raise ValueError("invalid rollout group count")
    keys = ("request_id", "images", "videos", "clip_ids", "clip_bounds", "sampled_pts", "pseudo_scores", "messages", "media_signature")
    for offset in range(0, len(rows), group_size):
        group = rows[offset:offset + group_size]
        for row in group:
            audit_window_record(row)
            if row["task"] != "grpo":
                raise ValueError("rollouts require GRPO inputs")
            if require_tensor_hash and not row.get("media_tensor_sha256"):
                raise ValueError("GPU evidence requires a processed-media tensor hash")
        baseline = {k: group[0][k] for k in keys}
        for row in group[1:]:
            if canonical_hash({k: row[k] for k in keys}) != canonical_hash(baseline):
                raise ValueError("same-group completions do not share input/q/order")
            if row.get("media_tensor_sha256") != group[0].get("media_tensor_sha256"):
                raise ValueError("same-group processed media tensors differ")
    return {"n_groups": len(rows) // group_size, "n_completions": len(rows),
            "group_size": group_size, "processed_media_hashes_required": require_tensor_hash}


def aggregate_windows(n_clips: int, predictions: Sequence[tuple[Sequence[int], str]]) -> tuple[np.ndarray, float]:
    """Invalid JSON contributes zero and remains in the averaging denominator."""
    if type(n_clips) is not int or n_clips < 1 or not predictions:
        raise ValueError("nonempty full-video predictions required")
    total, count = np.zeros(n_clips), np.zeros(n_clips, dtype=np.int64)
    invalid = 0
    for raw_ids, text in predictions:
        if not raw_ids or any(type(i) is not int for i in raw_ids):
            raise ValueError("clip indices must be explicit integers")
        ids = np.asarray(raw_ids)
        if len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= n_clips)):
            raise ValueError("invalid clip indices")
        try:
            scores = parse_scores(text, len(ids))
        except (ValueError, TypeError):
            invalid += 1
            scores = np.zeros(len(ids))
        total[ids] += scores
        count[ids] += 1
    if np.any(count == 0):
        raise ValueError("uncovered evaluation clips")
    return total / count, invalid / len(predictions)


def match_prediction_rows(examples: Sequence[Mapping], predictions: Sequence[Mapping]) -> list[dict]:
    """Strict ID join; rejects append leftovers, duplicates and missing windows."""
    expected, found = {}, {}
    for row in examples:
        key = row["request_id"]
        if key in expected:
            raise ValueError("duplicate example request ID")
        expected[key] = row
    for row in predictions:
        key = row["request_id"]
        if key in found:
            raise ValueError("duplicate prediction request ID (possibly appended output)")
        if not isinstance(row.get("response"), str):
            raise ValueError("prediction response must be a string")
        found[key] = row
    if set(expected) != set(found):
        raise ValueError(f"prediction coverage differs: missing={sorted(set(expected)-set(found))}, extra={sorted(set(found)-set(expected))}")
    return [dict(found[key]) for key in expected]
