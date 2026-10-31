#!/usr/bin/env python3
"""Post-run independent metrics/aggregation verification; no model re-fitting."""
import argparse
import importlib.util
import time
import warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from unsgrpo.round2_utils import *


def main(out):
    started=time.monotonic();state=read(ROOT/'state.json')
    if state['status']!='COMPLETED_NUMERICAL_DIAGNOSTICS':raise ValueError('Wait for all fixed fits and endpoints')
    freeze=read(ROOT/'round2.freeze.json')
    for group in ('sources','inputs'):
        for p,digest in freeze[group].items():
            if sha(p)!=digest:raise ValueError(f'Changed frozen {group}: {p}')
    plan,by,train,dev,feat,pseudo=load_assets()
    path=Path('/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1/evaluation_sensitivity_v01/evaluate_sensitivity.py')
    spec=importlib.util.spec_from_file_location('original_registered_time_mapper',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    labels={}
    for k in dev:
        with np.load(Path(plan['paths']['labels'])/(k+'.npz'),allow_pickle=False) as data:labels[k]=(data['bounds'].copy(),data['ratings'].copy())
    maximum=0.;spearman_error=0.;n_video=0;n_user=0;independent_final={}
    def check(k,scores,expected):
        nonlocal maximum,spearman_error,n_video,n_user
        bounds,human=labels[k];mapped=np.round(mod.stable_interval_average(by[k]['bounds'],scores,bounds),12)
        binary=human>np.median(human,axis=0)[None,:]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ap=np.array([average_precision_score(binary[:,j],mapped) for j in range(20)])
            correlation=float(spearmanr(mapped,human.mean(1)).statistic) if np.ptp(mapped)>0 and np.ptp(human.mean(1))>0 else None
        err=float(np.max(np.abs(ap-expected['ap_by_annotator'])));maximum=max(maximum,err)
        if err>1e-12 or abs(ap.mean()-expected['mean_ap'])>1e-12:raise ValueError('AP implementation mismatch')
        old=expected['spearman']['value']
        if (old is None)!=(correlation is None):raise ValueError('Constant Spearman handling mismatch')
        if old is not None:
            spearman_error=max(spearman_error,abs(old-correlation))
            if abs(old-correlation)>1e-12:raise ValueError('Spearman mismatch')
        n_video+=1;n_user+=20
        return float(ap.mean())
    teacher=read(ROOT/'teachers/results.json');cached=read(ROOT/'cached/results.json');student=read(ROOT/'students/evaluation_results.json')
    for file,methods in [(ROOT/'teachers/q_scores.npz',{n:v['development_evaluation'] for n,v in teacher['variants'].items()}),
                         (ROOT/'cached/scores.npz',{n:v['evaluation'] for n,v in cached['methods'].items()})]:
        with np.load(file,allow_pickle=False) as predictions:
            for name,evaluation in methods.items():
                for expected in evaluation['per_video']:
                    k=expected['video_id'];check(k,predictions[f'{name}__{k}'],expected)
    pool=read(ROOT/'students/data_receipt.json')['development_rows']
    for name,run in student['runs'].items():
        receipt=read(ROOT/'students'/name/'receipt.json')
        if receipt['successful_optimizer_updates']!=256 or receipt['input_window_exposures']!=4096 or receipt['trainable_parameters']!=32897:
            raise ValueError('Small student budget/capacity mismatch')
        with np.load(ROOT/'students'/name/'predictions.npz',allow_pickle=False) as predictions:
            for step,point in run['steps'].items():
                pred=predictions['dev_'+step]
                for expected in point['evaluation']['per_video']:
                    k=expected['video_id'];terms={i:[] for i in range(len(by[k]['bounds']))}
                    for row,values in zip(pool,pred):
                        if row['video_id']==k:
                            for clip,value in zip(row['clip_ids'],values):terms[clip].append(float(value))
                    if any(not v for v in terms.values()):raise ValueError('Missing clips')
                    scores=np.array([sum(v)/len(v) for v in terms.values()])
                    value=check(k,scores,expected)
                    if step=='256':independent_final[(run['view'],run['objective'],run['seed'],k)]=value
    # Independently assemble matched source vectors, average seeds, then resample sources.
    contrasts={}
    for key,expected in student['paired_source_bootstrap_seed_averaged'].items():
        if key.startswith('global_minus_local'):
            objective=key.split('__',1)[1];a=('window_plus_video_statistics',objective);b=('window_only',objective)
        else:
            view=key.split('__',1)[1];a=(view,'rank_pair');b=(view,'mse_q')
        diff=np.array([np.mean([independent_final[(*a,seed,k)]-independent_final[(*b,seed,k)] for seed in [42,43,44]]) for k in sorted(dev)])
        draws=np.random.default_rng(42).integers(0,len(diff),(10000,len(diff)));means=diff[draws].mean(1)
        observed={'mean':float(diff.mean()),'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975))}
        if any(abs(observed[k]-expected[k])>1e-12 for k in observed):raise ValueError('Independent source bootstrap mismatch')
        contrasts[key]=observed
    rows=records(ROOT/'input_counterfactual/predictions.jsonl')
    if len(rows)!=24 or len({(r['condition'],r['request_id']) for r in rows})!=24:raise ValueError('Actual input counterfactual coverage changed')
    old={r['request_id']:r for r in records(Path(plan['paths']['original_run'])/'inference/S0/windows.jsonl')}
    for rid in {r['request_id'] for r in rows}:
        a,b,c=[next(r for r in rows if r['request_id']==rid and r['condition']==q) for q in 'ABC']
        if a['response']!=old[rid]['response']:raise ValueError('Exact frozen S0 replay not reproduced')
        if b['messages']!=c['messages']:raise ValueError('Matched B/C prompt changed')
        for key in ('input_ids_sha256','image_grid_thw','video_grid_thw','video_pixel_tensor_sha256'):
            if b['audit'][key]!=c['audit'][key]:raise ValueError('Counterfactual non-image variable changed')
    output={'status':'INDEPENDENT_ROUND2_CHECK_PASSED','created_utc':now(),'script':reference(__file__),
        'n_video_metric_instances':n_video,'n_annotator_AP_values':n_user,'max_AP_error':maximum,'max_Spearman_error':spearman_error,
        'bootstrap':contrasts,'same_source_pairs_not_30_independent_videos':True,
        'counterfactual_exact_A_replays':8,'B_C_only_second_image_contents_differ':True,
        'frozen_sources_and_inputs_unchanged':True,'training_or_Qwen_inference_repeated':False,
        'scope':'Independent sklearn/scipy metrics, explicit scalar clip aggregation and vectorized source bootstrap; original registered time mapper reused, no independent native-clock reconstruction.',
        'wall_seconds':time.monotonic()-started}
    write(out,output);print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
