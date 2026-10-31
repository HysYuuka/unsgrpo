#!/usr/bin/env python3
"""Frozen-output calibration, rank, overlap and label-free offset diagnostics."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from unsgrpo.core import reward_parts
from unsgrpo.diagnostics import aggregate, offset_consensus
from unsgrpo.round2_utils import *
from unsgrpo.round2_metrics import DevelopmentEvaluator, paired, rho


def context_association(ids,values,features):
    observations=defaultdict(list)
    for ii,vv in zip(ids,values):
        for p,(i,s) in enumerate(zip(ii,vv)):
            others=[j for j in ii if j!=i]
            mean=features[others].mean(0)
            cosine=float(features[i]@mean/(np.linalg.norm(mean)+1e-30))
            observations[i].append((p/max(1,len(ii)-1),cosine,float(s)))
    xs=[];ys=[]
    for rows in observations.values():
        if len(rows)<2:continue
        a=np.asarray(rows);a-=a.mean(0);xs.extend(a[:,:2]);ys.extend(a[:,2])
    x,y=np.asarray(xs),np.asarray(ys)
    beta=np.linalg.solve(x.T@x+np.eye(2)*1e-8,x.T@y)
    residual=y-x@beta
    return {'position_coefficient':float(beta[0]),'neighbor_cosine_coefficient':float(beta[1]),
            'within_clip_R2':float(1-np.sum(residual**2)/np.sum(y**2)) if np.sum(y**2)>0 else None,
            'n_centered_observations_descriptive_only':len(y),
            'causal_position_vs_neighbor_separation':False}


def main():
    start=time.monotonic();plan,by,train,dev,feat,pseudo=load_assets();out=ROOT/'cached';out.mkdir()
    ev=DevelopmentEvaluator(plan,by,dev);results={};score_arrays={};overlap={};invariance={};decomposition={};cache={}
    original_eval=read(Path(plan['paths']['original_run'])/'evaluation_V0.json')
    registered_sensitivity=read('/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1/evaluation_sensitivity_v01/results/sensitivity.json')
    for method in plan['cached_predictions']['methods']:
        videos={r['video_id']:r for r in records(Path(plan['paths']['original_run'])/'inference'/method.replace('+','PLUS')/'videos.jsonl')}
        original={k:np.asarray(videos[k]['scores']) for k in dev}
        results[method]={'evaluation':ev.evaluate(original),'per_video':[],'kind':'CACHED_ORIGINAL'}
        if abs(results[method]['evaluation']['video_macro_ap']-registered_sensitivity['variants']['stable_round12']['runs'][method]['video_macro_ap'])>1e-12:
            raise ValueError(f'New development evaluator does not reproduce canonical original result: {method}')
        invariance[method]=[dict(video_id=k,**ev.affine_invariance(k,original[k])) for k in dev]
        for k in dev:
            q=np.asarray(pseudo[k]['q']);score_arrays[f'{method}__{k}']=original[k]
            results[method]['per_video'].append({'video_id':k,**pseudo_metrics(original[k],q),**dispersion(original[k])})
        if method=='B0':continue
        group=cached_windows(plan,method,by,dev);cache[method]=group
        if method in {'S0','S+','G-CAL','G-RANK'}:
            overlap[method]=[];decomposition[method]=[]
            for k in dev:
                g=group[k];base=aggregate(len(original[k]),g['ids'],g['values'])
                if not np.array_equal(base,original[k]):raise ValueError('Cached aggregation not exactly reproduced')
                overlap[method].append({'video_id':k,**overlap_statistics(g['ids'],g['values']),
                                        'context_association':context_association(g['ids'],g['values'],feat[k])})
                s=np.concatenate(g['values']);q=np.concatenate([np.asarray(pseudo[k]['q'])[ids] for ids in g['ids']])
                variance_s=float(np.var(s));variance_q=float(np.var(q));cov=float(np.mean((s-s.mean())*(q-q.mean())))
                mse=float(np.mean((s-q)**2));identity=variance_s+variance_q-2*cov+float((s.mean()-q.mean())**2)
                if abs(mse-identity)>1e-12:raise ValueError('MSE decomposition failed')
                decomposition[method].append({'video_id':k,'mse':mse,'prediction_variance':variance_s,
                    'teacher_variance':variance_q,'twice_covariance':2*cov,'mean_bias_squared':float((s.mean()-q.mean())**2),
                    'constant50_mse':float(np.mean((.5-q)**2)),'prediction_teacher_rho':rho(s,q)})
            for variant in ('affine','affine_quantized','offset_consensus'):
                name=method+'__'+variant;scores={};per=[];deltas=[]
                for k in dev:
                    g=group[k]
                    if variant=='offset_consensus':
                        values,bias=offset_consensus(g['ids'],g['values'],ridge=1.0)
                    else:
                        values=[.25*s+.375 for s in g['values']];bias=None
                        if variant=='affine_quantized':values=[np.round(s,2) for s in values]
                    pred=aggregate(len(pseudo[k]['q']),g['ids'],values);scores[k]=pred
                    r={'video_id':k,**pseudo_metrics(pred,pseudo[k]['q']),**dispersion(pred),
                       'overlap':overlap_statistics(g['ids'],values)}
                    if bias is not None:r['window_offsets']=bias.tolist()
                    per.append(r);score_arrays[f'{name}__{k}']=pred
                evaluation=ev.evaluate(scores)
                results[name]={'kind':'POSTPROCESSING_ONLY','evaluation':evaluation,'per_video':per,
                    'paired_vs_original':paired(evaluation,results[method]['evaluation']),
                    'teacher_or_human_used_for_offset_fit':False if variant=='offset_consensus' else None}
    constant={k:np.full(len(pseudo[k]['q']),.5) for k in dev}
    results['CONSTANT50']={'kind':'NO_MODEL_BASELINE','evaluation':ev.evaluate(constant),
        'per_video':[dict(video_id=k,**pseudo_metrics(constant[k],pseudo[k]['q']),**dispersion(constant[k])) for k in dev]}
    for k in dev:score_arrays[f'CONSTANT50__{k}']=constant[k]
    # Reproduce the window-weighted reward fields, and contrast constant/shrunk outputs.
    rewards={}
    for method,group in cache.items():
        parts=[];const=[];shrink=[]
        for k,g in group.items():
            for ids,record,s in zip(g['ids'],g['records'],g['values']):
                q=np.asarray(pseudo[k]['q'])[ids];p=reward_parts(record['response'],q);parts.append(p)
                const.append(1-float(np.mean((.5-q)**2)))
                if p.valid:shrink.append(1-float(np.mean((.25*s+.375-q)**2)))
        rewards[method]={'n_windows':len(parts),'legal_rate':float(np.mean([p.valid for p in parts])),
            'mean_total_including_illegal':float(np.mean([p.total for p in parts])),
            'calibration_among_legal':float(np.mean([p.calibration for p in parts if p.valid])),
            'ranking_among_legal':float(np.mean([p.ranking for p in parts if p.valid])),
            'constant50_calibration_all_windows':float(np.mean(const)),
            'shrunk_calibration_among_legal':float(np.mean(shrink))}
    before=np.mean([r['mean_range'] for r in overlap['S0']]);after=np.mean([r['overlap']['mean_range'] for r in results['S0__offset_consensus']['per_video']])
    fraction=float(1-after/before) if before>0 else 0.
    trigger=bool(before>=.05 and fraction>=.20)
    # Selection contains no performance/human scores and is fixed by the protocol.
    chosen=sorted(np.random.default_rng(42).choice(np.asarray(dev,dtype=object),4,replace=False).tolist())
    selected=[]
    for k in chosen:
        windows=cache['S0'][k]['records'];first=max(0,min(len(windows)-2,(len(windows)-1)//2))
        selected.extend(windows[first:first+2])
    write(out/'input_counterfactual_gate.json',{'run':trigger,'source':'Cached S0 shared-clip predictions only',
        'before_mean_range':float(before),'after_mean_range':float(after),'reduction_fraction':fraction,
        'human_or_teacher_scores_used':False,'chosen_video_ids':chosen,
        'selected_request_ids':[r['request_id'] for r in selected],
        'n_pairs':4,'checkpoint':'Frozen common merged S0'})
    np.savez_compressed(out/'scores.npz',**score_arrays)
    write(out/'results.json',{'status':'COMPLETED','evaluation_status':'DEVELOPMENT_ONLY','methods':results,
        'cross_window':overlap,'mse_decomposition':decomposition,'window_weighted_rewards':rewards,
        'monotone_after_canonical_mapping':invariance,
        'max_affine_canonical_user_AP_difference':max(r['max_user_ap_difference'] for v in invariance.values() for r in v),
        'gate':read(out/'input_counterfactual_gate.json'),'wall_seconds':time.monotonic()-start})


if __name__=='__main__':main()
