"""Label-free data, provenance and fitting utilities for the bounded second round."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .core import unit_rows, midrank_percentile, parse_scores
from .data import window_indices
from .diagnostics import rank_agreement

PROJECT = Path('/root/autodl-tmp/unsgrpo')
ROOT = Path('/root/autodl-tmp/unsgrpo-work/runs/round2_diagnostic_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def reference(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write(path, value, *, fresh=True):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if fresh and p.exists():
        raise FileExistsError(p)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(t, p)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_assets(root=ROOT):
    plan = read(root/'protocol.json'); paths = plan['paths']
    rows = sorted(records(paths['prepared']), key=lambda r:r['video_id'])
    by = {r['video_id']:r for r in rows}
    train = [r['video_id'] for r in rows if r['split']=='train']
    dev = [r['video_id'] for r in rows if r['split']=='test']
    if len(train)!=40 or len(dev)!=10:
        raise ValueError('Keep exactly original 40 train and reclassified 10 development videos')
    feat = {k:unit_rows(np.load(v['features_path'],allow_pickle=False)) for k,v in by.items()}
    pseudo = read(paths['pseudo'])['videos']
    for k,v in by.items():
        if len(feat[k])!=len(v['bounds']) or sha(v['features_path'])!=pseudo[k]['features_sha256']:
            raise ValueError(f'Feature/time identity mismatch: {k}')
    return plan, by, train, dev, feat, pseudo


def numerical_summary(x):
    a=np.asarray(x,dtype=float)
    a=a[np.isfinite(a)]
    return {'n':int(a.size),'mean':float(a.mean()) if a.size else None,
            'std':float(a.std()) if a.size else None,'min':float(a.min()) if a.size else None,
            'median':float(np.median(a)) if a.size else None,'max':float(a.max()) if a.size else None}


def dispersion(s):
    a=np.asarray(s,dtype=float); _,count=np.unique(a,return_counts=True)
    return {'mean':float(a.mean()),'std':float(a.std()),'range':float(np.ptp(a)),
            'n_distinct':len(count),'pair_tie_fraction':float(np.sum(count*(count-1))/(len(a)*(len(a)-1))) if len(a)>1 else 1.0}


def pseudo_metrics(s,q):
    s,q=np.asarray(s,dtype=float),np.asarray(q,dtype=float)
    return {'mse':float(np.mean((s-q)**2)),'calibration':float(1-np.mean((s-q)**2)),
            'weighted_pair_agreement':rank_agreement(s,q)}


def overlap_statistics(ids, values):
    grouped=defaultdict(list)
    for w,(ii,vv) in enumerate(zip(ids,values)):
        for pos,(i,s) in enumerate(zip(ii,vv)):
            grouped[int(i)].append((w,pos,float(s)))
    repeated=[v for v in grouped.values() if len(v)>1]
    ranges=[np.ptp([x[2] for x in v]) for v in repeated]
    variances=[np.var([x[2] for x in v]) for v in repeated]
    pairs=defaultdict(list)
    for ii,vv in zip(ids,values):
        for a in range(len(ii)):
            for b in range(a+1,len(ii)):
                pairs[(int(ii[a]),int(ii[b]))].append(float(np.sign(vv[a]-vv[b])))
    flips=[abs(v[a]-v[b])/2 for v in pairs.values() for a in range(len(v)) for b in range(a+1,len(v))]
    sd=float(np.std(np.concatenate(values)))
    return {'n_repeated_clips':len(repeated),'mean_range':float(np.mean(ranges)) if ranges else 0.0,
            'p95_range':float(np.quantile(ranges,.95)) if ranges else 0.0,
            'mean_std':float(np.mean(np.sqrt(variances))) if variances else 0.0,
            'mean_variance':float(np.mean(variances)) if variances else 0.0,
            'window_score_std':sd,
            'scale_normalized_mean_range':float(np.mean(ranges)/sd) if ranges and sd>0 else 0.0,
            'overlap_pair_order_disagreement':float(np.mean(flips)) if flips else 0.0,
            'n_repeated_pair_comparisons':len(flips)}


def cached_windows(plan, method, by, dev):
    base=Path(plan['paths']['original_run'])/'inference'/method.replace('+','PLUS')
    lines=records(base/'windows.jsonl'); result={}
    for k in dev:
        group=[r for r in lines if r['video_id']==k]
        expected=window_indices(len(by[k]['bounds']))
        if [r['clip_ids'] for r in group]!=expected:
            raise ValueError(f'Cached window coverage/order differs: {method}/{k}')
        scores=[]; legal=[]
        for r in group:
            try:
                scores.append(parse_scores(r['response'],len(r['clip_ids'])));legal.append(True)
            except (ValueError,TypeError):
                scores.append(np.zeros(len(r['clip_ids'])));legal.append(False)
        result[k]={'ids':expected,'values':scores,'legal':legal,'records':group}
    return result


def prototypes(features, train_ids, clusters, *, video_equal=False, exclude=None):
    result={}; fallbacks={}
    for c in sorted(set(clusters[k] for k in train_ids)):
        peers=[k for k in train_ids if clusters[k]==c and k!=exclude]
        if not peers:
            peers=[k for k in train_ids if k!=exclude];fallbacks[int(c)]=True
        if video_equal:
            result[int(c)]=np.mean([features[k].mean(0) for k in peers],0)
        else:
            result[int(c)]=np.concatenate([features[k] for k in peers]).mean(0)
    return result,fallbacks


def teacher_scores(features, ids, clusters, banks):
    return {k:{'raw':features[k]@banks[int(clusters[k])],
               'q':midrank_percentile(features[k]@banks[int(clusters[k])])} for k in ids}
