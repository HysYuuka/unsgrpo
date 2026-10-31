"""CPU-testable core for label-free highlight experiments.

No human highlight labels are accepted by reward or pseudo-label functions.
The module is intentionally independent of Qwen / ms-swift.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence
import numpy as np


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("features must be a nonempty finite [clips, dim] array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("zero feature: fix extraction; do not silently create a cosine")
    return x / norms


def recurrence_original(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Exact mean cosine, using a NON-renormalized mean of unit bank vectors."""
    q, b = unit_rows(query), unit_rows(bank)
    if q.shape[1] != b.shape[1]:
        raise ValueError("feature dimensions differ")
    return q @ b.mean(axis=0)


def recurrence_loo_video(
    video_id: str,
    feature_map: Mapping[str, np.ndarray],
    clusters: Mapping[str, int],
    *,
    global_fallback: bool = True,
) -> tuple[np.ndarray, dict]:
    """Optional adaptation: exclude the entire query video; weight peer videos equally.

    feature_map/clusters MUST contain training videos only. Cluster membership can
    still have been fitted using this video's embedding: this is not full cross-fitting.
    """
    if set(feature_map) != set(clusters) or video_id not in feature_map:
        raise ValueError("feature/cluster keys differ or unknown video")
    peers = [k for k in feature_map if k != video_id and clusters[k] == clusters[video_id]]
    fallback = False
    if not peers:
        if not global_fallback:
            raise ValueError("no other video in this cluster")
        peers = [k for k in feature_map if k != video_id]
        fallback = True
    if not peers:
        raise ValueError("at least two training videos are required")
    means = np.stack([unit_rows(feature_map[k]).mean(axis=0) for k in peers])
    score = unit_rows(feature_map[video_id]) @ means.mean(axis=0)
    return score, {"peers": peers, "global_fallback": fallback}


def midrank_percentile(values: Sequence[float]) -> np.ndarray:
    """Whole-video midrank ECDF. Ties receive equal scores; constant input -> 0.5."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("values must be finite nonempty 1D")
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        # Occupied 1-based ranks are i+1,...,j. (average_rank - 0.5) / T.
        result[order[i:j]] = (i + j) / (2.0 * len(x))
        i = j
    return result


def _no_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_scores(text: str, n_clips: int, max_score: int = 100) -> np.ndarray:
    """Strict JSON parser. No eval, regex repair, dropped clips, or clamping."""
    if type(n_clips) is not int or n_clips < 1 or type(max_score) is not int or max_score <= 0:
        raise ValueError("invalid score dimensions")
    if not isinstance(text, str) or len(text) > 20000:
        raise ValueError("invalid completion input")
    obj = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_unique_object)
    if not isinstance(obj, dict) or set(obj) != {"scores"}:
        raise ValueError('expected exactly {"scores": [...]}')
    scores = obj["scores"]
    if not isinstance(scores, list) or len(scores) != n_clips:
        raise ValueError("wrong score count")
    if any(type(s) is not int or s < 0 or s > max_score for s in scores):
        raise ValueError("scores must be integers in the declared range")
    return np.asarray(scores, dtype=np.float64) / max_score


@dataclass(frozen=True)
class RewardParts:
    total: float
    ranking: float
    calibration: float
    valid: bool


def reward_parts(completion: str, pseudo_scores: Sequence[float], rank_weight: float = 0.8) -> RewardParts:
    q = np.asarray(pseudo_scores, dtype=np.float64)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        # Dataset bugs are not policy failures: fail rather than silently rewarding.
        raise ValueError("pseudo_scores must be finite in [0,1]")
    if not 0 <= rank_weight <= 1:
        raise ValueError("rank_weight outside [0,1]")
    try:
        s = parse_scores(completion, len(q))
    except (ValueError, TypeError, json.JSONDecodeError):
        return RewardParts(-1.0, 0.0, 0.0, False)
    ii, jj = np.triu_indices(len(q), k=1)
    delta_q, delta_s = q[ii] - q[jj], s[ii] - s[jj]
    w = np.abs(delta_q)
    correct = (delta_q * delta_s > 0).astype(np.float64)
    correct += 0.5 * (delta_s == 0)
    ranking = float(np.dot(w, correct) / w.sum()) if w.sum() > 0 else 0.5
    calibration = float(1.0 - np.mean((s - q) ** 2))
    total = rank_weight * ranking + (1.0 - rank_weight) * calibration
    return RewardParts(float(total), ranking, calibration, True)


def group_advantages(rewards: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Explicit population std (ddof=0). Framework std convention must be logged."""
    x = np.asarray(rewards, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("expected finite [prompts, generations>=2]")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be nonnegative")
    centered = x - x.mean(axis=1, keepdims=True)
    denominator = x.std(axis=1, keepdims=True) + eps
    return np.divide(centered, denominator, out=np.zeros_like(x), where=denominator > 0)



def grpo_loss(logp_new, logp_old, logp_ref, rewards, mask, group_size=4, beta=0.04, clip=0.2, eps=1e-4):
    """Inputs: logps/mask [B*G,L], rewards [B*G]; all rows in prompt-major order.

    Uses population std explicitly. K3 is evaluated on rollout samples and is not an
    exact off-policy KL estimator. This reproduces a common local surrogate only.
    """
    import torch
    if group_size < 2 or beta < 0 or clip < 0 or eps <= 0:
        raise ValueError("invalid GRPO hyperparameters")
    if logp_new.ndim != 2 or logp_new.shape != logp_old.shape or logp_new.shape != logp_ref.shape or mask.shape != logp_new.shape:
        raise ValueError("log-prob/mask shape mismatch")
    n, length = logp_new.shape
    if n % group_size or rewards.shape != (n,):
        raise ValueError("invalid grouping")
    if not torch.isfinite(rewards).all() or not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("rewards must be finite and mask must be binary")
    if not all(torch.isfinite(x).all() for x in (logp_new, logp_old, logp_ref)):
        raise FloatingPointError("non-finite token log probabilities")
    if n == 0 or length == 0:
        raise ValueError("nonempty completions required")
    mask = mask.to(logp_new.dtype)
    lengths = mask.sum(-1)
    if torch.any(lengths <= 0):
        raise ValueError("empty completion")
    with torch.no_grad():
        grouped = rewards.detach().float().reshape(-1, group_size)
        adv = (grouped - grouped.mean(-1, keepdim=True)) / (grouped.std(-1, keepdim=True, unbiased=False) + eps)
        adv = adv.reshape(n, 1)
    delta = logp_new.float() - logp_old.detach().float()
    if not torch.isfinite(delta).all():
        raise FloatingPointError("non-finite policy log-ratio")
    ratio = delta.exp()
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
    dref = logp_ref.detach().float() - logp_new.float()
    k3 = torch.expm1(dref) - dref
    per_seq = ((-surrogate + beta*k3) * mask).sum(-1) / lengths
    loss = per_seq.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite GRPO loss")
    return loss
