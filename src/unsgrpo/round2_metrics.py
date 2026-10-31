"""Development-only V0, preserving the registered time mapper and tie policy."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .diagnostics import StableMapper
from .evaluation import average_precision_noninterpolated, spearman_with_status, paired_bootstrap
from .round2_utils import read, sha


class DevelopmentEvaluator:
    def __init__(self, plan, by, dev):
        self.dev=dev;self.by=by;self.labels={};self.mapping={};self.clip_labels={};self.binary={}
        base=Path(plan['paths']['labels']); receipt=read(base/'conversion_receipt.json')
        converted={v['video_id']:v for v in receipt['videos']}
        if receipt['status']!='COMPLETED' or receipt['evaluation_unit']!='native_frame':
            raise ValueError('Use the verified original conversion and native-frame clock')
        for k in dev:
            if by[k]['split']!='test':raise ValueError('Only original 10 development IDs can load human labels')
            p=base/(k+'.npz')
            if sha(p)!=converted[k]['converted_sha256']:raise ValueError(f'Changed evaluation labels: {k}')
            with np.load(p,allow_pickle=False) as data:
                bounds,ratings=data['bounds'].copy(),data['ratings'].copy()
            if ratings.shape!=(len(bounds),20):raise ValueError('Wrong native_frame/annotator axes')
            self.labels[k]=(bounds,ratings)
            self.mapping[k]=StableMapper(by[k]['bounds'],bounds)
            self.binary[k]=ratings>np.median(ratings,axis=0)[None,:]
            self.clip_labels[k]=StableMapper(bounds,by[k]['bounds'])(ratings.mean(1))

    def mapped(self,k,s):
        return self.mapping[k](s)

    def score(self,k,s):
        mapped=self.mapped(k,s); canonical=np.round(mapped,12);binary=self.binary[k];human=self.labels[k][1]
        def metrics(x):
            # Preserve the exact registered AP implementation; grouping is not rewritten.
            ap=[average_precision_noninterpolated(binary[:,j],x) for j in range(20)]
            return {'mean_ap':float(np.mean(ap)),'ap_by_annotator':ap,
                    'spearman':spearman_with_status(x,human.mean(1)),
                    'n_distinct_mapped':int(len(np.unique(x)))}
        primary=metrics(canonical);unrounded=metrics(mapped)
        return {'video_id':k,'source_group':self.by[k]['source_group'],'evaluation_status':'DEVELOPMENT',
                **primary,'unrounded_stable_ap':unrounded['mean_ap'],
                'unrounded_stable_spearman':unrounded['spearman'],
                'n_no_positive_annotators':int((binary.sum(0)==0).sum())}

    def evaluate(self,scores):
        if set(scores)!=set(self.dev):raise ValueError('All 10 development videos required')
        rows=[self.score(k,scores[k]) for k in self.dev]
        defined=[r['spearman']['value'] for r in rows if r['spearman']['defined']]
        return {'evaluation_status':'DEVELOPMENT_ONLY','n_videos':len(rows),
                'video_macro_ap':float(np.mean([r['mean_ap'] for r in rows])),
                'unrounded_stable_macro_ap':float(np.mean([r['unrounded_stable_ap'] for r in rows])),
                'mean_defined_spearman':float(np.mean(defined)) if defined else None,
                'n_undefined_spearman':len(rows)-len(defined),'per_video':rows}

    def affine_invariance(self,k,s,slope=.25,intercept=.375):
        canonical=np.round(self.mapped(k,s),12); transformed=slope*canonical+intercept
        first=[average_precision_noninterpolated(self.binary[k][:,j],canonical) for j in range(20)]
        second=[average_precision_noninterpolated(self.binary[k][:,j],transformed) for j in range(20)]
        return {'max_user_ap_difference':float(np.max(np.abs(np.array(first)-second))),
                'n_unique_before':len(np.unique(canonical)),'n_unique_after':len(np.unique(transformed))}


def paired(first,second):
    a={r['video_id']:r for r in first['per_video']};b={r['video_id']:r for r in second['per_video']}
    if set(a)!=set(b):raise ValueError('Paired IDs differ')
    ids=sorted(a)
    return paired_bootstrap([a[k]['mean_ap']-b[k]['mean_ap'] for k in ids],
                            source_groups=[a[k]['source_group'] for k in ids],n_boot=10000,seed=42)


def rho(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(spearmanr(a,b).statistic)
