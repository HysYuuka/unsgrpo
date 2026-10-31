#!/usr/bin/env python3
"""Describe the existing mean aggregator's rank/tie effects; no new variants."""
import time
import numpy as np
from unsgrpo.round2_utils import *
from unsgrpo.round2_metrics import DevelopmentEvaluator
from unsgrpo.core import midrank_percentile
from unsgrpo.diagnostics import aggregate,rank_agreement


def main():
    start=time.monotonic();plan,by,train,dev,feat,pseudo=load_assets();ev=DevelopmentEvaluator(plan,by,dev)
    methods={}
    for method in ['S0','S+','G-CAL','G-RANK']:
        rows=[];cache=cached_windows(plan,method,by,dev)
        for k,g in cache.items():
            mean=np.round(aggregate(len(pseudo[k]['q']),g['ids'],g['values']),12)
            human=midrank_percentile(ev.clip_labels[k]);local_agree=[];mean_agree=[];ties=[];changes=[];tie_resolved=[]
            for ids,scores in zip(g['ids'],g['values']):
                scores=np.round(scores,12);after=mean[ids];q=human[ids];i,j=np.triu_indices(len(ids),1)
                before_sign=np.sign(scores[i]-scores[j]);after_sign=np.sign(after[i]-after[j])
                local_agree.append(rank_agreement(scores,q));mean_agree.append(rank_agreement(after,q))
                ties.append(float(np.mean(before_sign==0)));changes.append(float(np.mean(np.abs(before_sign-after_sign)/2)))
                tie_resolved.append(float(np.mean((before_sign==0)&(after_sign!=0))))
            _,count=np.unique(mean,return_counts=True)
            rows.append({'video_id':k,'window_pair_tie_fraction':float(np.mean(ties)),
                'whole_video_aggregate_pair_tie_fraction':float(np.sum(count*(count-1))/(len(mean)*(len(mean)-1))),
                'window_pairs_changed_by_existing_aggregation_ties_half':float(np.mean(changes)),
                'fraction_window_pairs_whose_tie_is_resolved':float(np.mean(tie_resolved)),
                'human_pair_agreement_before_aggregation':float(np.mean(local_agree)),
                'human_pair_agreement_after_existing_aggregation_same_pairs':float(np.mean(mean_agree))})
        methods[method]={'per_video':rows,'macro':{key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='video_id'}}
    write(ROOT/'postrun_analysis/aggregation_rank_tie_audit.json',{'status':'COMPLETED_DESCRIPTIVE_POST_RUN_AUDIT',
        'methods':methods,'new_prediction_or_aggregation_variants':0,'human_used_for_fitting':False,
        'definition':'Compare each original window order with the already-registered mean-aggregated scores on the same clip pairs. Local diagnostic uses mean-user importance midranks, not the V0 AP definition. No alternative aggregation rule is evaluated or selected.',
        'script':reference(__file__),'wall_seconds':time.monotonic()-start})
    for name,r in methods.items():print(name,r['macro'])


if __name__=='__main__':main()
