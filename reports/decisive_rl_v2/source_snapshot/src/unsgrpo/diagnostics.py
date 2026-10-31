"""Small CPU diagnostics; human labels never enter student optimization."""
from __future__ import annotations

import numpy as np


class StableMapper:
    """Cache overlap indices; preserve constant ties like the registered mapper."""
    def __init__(self, source, target):
        self.source = np.asarray(source, dtype=float)
        self.target = np.asarray(target, dtype=float)
        for x in (self.source, self.target):
            if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
                raise ValueError("invalid intervals")
            if np.any(x[:, 1] <= x[:, 0]) or np.any(x[1:, 0] < x[:-1, 1] - 1e-8):
                raise ValueError("nonpositive or overlapping intervals")
        self.first = np.searchsorted(self.source[:, 1], self.target[:, 0], side="right")
        if np.any(self.first >= len(self.source)):
            raise ValueError("uncovered target")
        self.mixtures = []
        for j, ((left, right), first) in enumerate(zip(self.target, self.first)):
            if left >= self.source[first, 0] and right <= self.source[first, 1]:
                continue
            stop = np.searchsorted(self.source[:, 0], right, side="left")
            ids = np.arange(first, stop)
            overlap = np.maximum(0, np.minimum(right, self.source[ids, 1]) - np.maximum(left, self.source[ids, 0]))
            if not np.isclose(overlap.sum(), right-left, atol=1e-6, rtol=1e-8):
                raise ValueError("partial coverage")
            self.mixtures.append((j, ids, overlap / (right-left)))

    def __call__(self, scores):
        v = np.asarray(scores, dtype=float)
        if v.shape != (len(self.source),) or not np.isfinite(v).all():
            raise ValueError("bad scores")
        result = v[self.first].copy()
        for j, ids, weights in self.mixtures:
            result[j] = v[ids[0]] + np.dot(weights, v[ids] - v[ids[0]])
        return result


def aggregate(n_clips, ids, values):
    if type(n_clips) is not int or n_clips < 1 or not ids or len(ids) != len(values):
        raise ValueError("invalid window count")
    total, count = np.zeros(n_clips), np.zeros(n_clips, dtype=int)
    for ii, vv in zip(ids, values):
        ii, vv = np.asarray(ii, dtype=int), np.asarray(vv, dtype=float)
        if ii.ndim != 1 or vv.shape != ii.shape or not len(ii) or not np.isfinite(vv).all() or len(np.unique(ii)) != len(ii) or np.any(ii < 0) or np.any(ii >= n_clips):
            raise ValueError("bad window indices")
        total[ii] += vv
        count[ii] += 1
    if np.any(count == 0):
        raise ValueError("uncovered clip")
    return total/count


def offset_consensus(ids, values, ridge=1.0):
    """Fit window offsets from shared clips only, with ridge to prevent drift.

    min_b sum_{shared clips}(s_w-b_w-s_v+b_v)^2 + ridge*sum_w b_w^2.
    No teacher or human labels. Returns unclipped scores and inferred offsets.
    """
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("positive ridge required")
    if not ids or len(ids) != len(values):
        raise ValueError("window count mismatch")
    for ii, vv in zip(ids, values):
        if not len(ii) or len(ii) != len(vv) or len(set(ii)) != len(ii) or not np.isfinite(vv).all():
            raise ValueError("invalid offset observations")
    values = [np.asarray(v, dtype=float) for v in values]
    matrix = np.eye(len(ids))*ridge
    rhs = np.zeros(len(ids))
    lookup = [{int(i): float(v) for i, v in zip(ii, vv)} for ii, vv in zip(ids, values)]
    for w in range(len(ids)):
        for v in range(w+1, len(ids)):
            common = sorted(lookup[w].keys() & lookup[v].keys())
            weight = len(common)
            if not weight:
                continue
            delta = sum(lookup[w][i]-lookup[v][i] for i in common)
            matrix[w, w] += weight; matrix[v, v] += weight
            matrix[w, v] -= weight; matrix[v, w] -= weight
            rhs[w] += delta; rhs[v] -= delta
    offsets = np.linalg.solve(matrix, rhs)
    return [v-b for v, b in zip(values, offsets)], offsets


def rank_agreement(scores, target):
    s, q = np.asarray(scores), np.asarray(target)
    if s.ndim != 1 or s.shape != q.shape or not len(q) or not np.isfinite(s).all() or not np.isfinite(q).all():
        raise ValueError("invalid rank vectors")
    i, j = np.triu_indices(len(q), 1)
    ds, dq = s[i]-s[j], q[i]-q[j]
    weight = np.abs(dq)
    if weight.sum() == 0:
        return .5
    return float(np.dot(weight, (ds*dq > 0) + .5*(ds == 0))/weight.sum())


def student_loss(pred, q, raw, objective, temperature=.1, rank_weight=.8):
    import torch
    mse = (pred-q).square().mean()
    if objective == "mse_q":
        return mse
    if objective == "mse_raw_then_percentile":
        return (pred-(raw+1)/2).square().mean()
    if objective not in {"rank_cal", "rank_pair"}:
        raise ValueError(objective)
    i, j = torch.triu_indices(q.shape[1], q.shape[1], 1, device=q.device)
    delta_q, delta_s = q[:, i]-q[:, j], pred[:, i]-pred[:, j]
    weights = delta_q.abs()
    pair_loss = torch.nn.functional.softplus(-delta_q.sign()*delta_s/temperature)
    ranks = (weights*pair_loss).sum(1)/weights.sum(1).clamp_min(1e-12)
    if objective == "rank_pair":
        return ranks.mean()
    return rank_weight*ranks.mean()+(1-rank_weight)*mse
