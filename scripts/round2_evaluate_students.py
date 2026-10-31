#!/usr/bin/env python3
"""Evaluate all pre-fixed student endpoints/trajectory snapshots after fitting."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from unsgrpo.diagnostics import aggregate
from unsgrpo.round2_utils import *
from unsgrpo.round2_metrics import DevelopmentEvaluator,paired,rho


def average_seed_evaluation(runs):
    per=[]
    ids=sorted({r['video_id'] for r in runs[0]['final_evaluation']['per_video']})
    for k in ids:
        rows=[next(r for r in run['final_evaluation']['per_video'] if r['video_id']==k) for run in runs]
        per.append({'video_id':k,'source_group':rows[0]['source_group'],
                    'mean_ap':float(np.mean([r['mean_ap'] for r in rows]))})
    means=[r['final_evaluation']['video_macro_ap'] for r in runs]
    return {'evaluation_status':'DEVELOPMENT_ONLY_SEED_AVERAGED_SOURCE_UNITS','per_video':per,
        'video_macro_ap':float(np.mean(means)),'initialization_seed_macro_values':means,
        'seed_std_ddof1':float(np.std(means,ddof=1)),'seed_min':min(means),'seed_max':max(means),
        'n_source_groups':len(ids),'n_seeds':len(runs),'not_independent_sample_size_30':True}


def main():
    start=time.monotonic();plan,by,train,dev,feat,pseudo=load_assets();out=ROOT/'students'
    completed=read(out/'training_complete.json')
    if completed['status']!='COMPLETED_ALL_12_BEFORE_HUMAN_EVALUATION' or len(completed['runs'])!=12:
        raise ValueError('All prescribed fitting must finish before student human evaluation')
    data=read(out/'data_receipt.json');pool=data['development_rows'];ev=DevelopmentEvaluator(plan,by,dev)
    results={};final_scores={};timelines=[]
    for entry in completed['runs']:
        if sha(entry['receipt']['path'])!=entry['receipt']['sha256']:raise ValueError('Changed fitting receipt')
        receipt=read(entry['receipt']['path']);predfile=Path(receipt['predictions']['path'])
        if sha(predfile)!=receipt['predictions']['sha256']:raise ValueError('Changed stored predictions')
        if receipt['human_annotation_files_opened'] or receipt['successful_optimizer_updates']!=256 or receipt['input_window_exposures']!=4096:
            raise ValueError('Student label isolation or exposure budget failed')
        name=entry['name'];perstep={}
        with np.load(predfile,allow_pickle=False) as predictions:
            for step in plan['students']['fixed_saved_prediction_steps']:
                values=predictions[f'dev_{step}'];scores={};per=[]
                for k in dev:
                    positions=[i for i,r in enumerate(pool) if r['video_id']==k]
                    ids=[pool[i]['clip_ids'] for i in positions];vv=[values[i] for i in positions]
                    score=aggregate(len(pseudo[k]['q']),ids,vv);scores[k]=score
                    per.append({'video_id':k,**pseudo_metrics(score,pseudo[k]['q']),**dispersion(score),
                        'window_pair_agreement':float(np.mean([rank_agreement(v,np.asarray(pseudo[k]['q'])[ii]) for ii,v in zip(ids,vv)])),
                        'overlap':overlap_statistics(ids,vv)})
                    if step==256:final_scores[f'{name}__{k}']=score
                evaluation=ev.evaluate(scores)
                proxy=next(x for x in receipt['metrics_snapshots'] if x['step']==step)
                point={'step':step,'evaluation':evaluation,'per_video_diagnostics':per,
                       'window_weighted_train_q_MSE':proxy['train_q_MSE'],
                       'window_weighted_dev_q_MSE':proxy['development_q_MSE_no_human']}
                perstep[str(step)]=point
                timelines.append({'run':name,'seed':receipt['seed'],'view':receipt['view'],'objective':receipt['objective'],
                    'step':step,'train_q_MSE':proxy['train_q_MSE'],'development_q_MSE':proxy['development_q_MSE_no_human'],
                    'development_AP':evaluation['video_macro_ap'],'mean_overlap_range':float(np.mean([x['overlap']['mean_range'] for x in per])),
                    'mean_overlap_pair_disagreement':float(np.mean([x['overlap']['overlap_pair_order_disagreement'] for x in per]))})
        results[name]={'seed':receipt['seed'],'view':receipt['view'],'objective':receipt['objective'],
            'fitting_receipt':entry['receipt'],'final_evaluation':perstep['256']['evaluation'],
            'final_diagnostics':perstep['256']['per_video_diagnostics'],'steps':perstep,
            'training_wall_seconds':receipt['wall_seconds'],'optimizer_training_seconds':receipt['timing']['optimizer_training_seconds'],
            'final_endpoint':receipt['checkpoint'],'initialization_sha256':receipt['initial_parameter_sha256'],
            'trainable_parameters':receipt['trainable_parameters'],'source_window_exposures':4096}
    groups={}
    for view in plan['students']['views']:
        for objective in plan['students']['objectives']:
            group=[r for r in results.values() if r['view']==view and r['objective']==objective]
            groups[f'{view}__{objective}']=average_seed_evaluation(group)
    contrasts={}
    for objective in plan['students']['objectives']:
        contrasts[f'global_minus_local__{objective}']=paired(groups[f'window_plus_video_statistics__{objective}'],groups[f'window_only__{objective}'])
    for view in plan['students']['views']:
        contrasts[f'ranking_minus_regression__{view}']=paired(groups[f'{view}__rank_pair'],groups[f'{view}__mse_q'])
    trajectory_summary=[]
    for name,r in results.items():
        a,b=r['steps']['0'],r['steps']['256']
        trajectory_summary.append({'run':name,'objective':r['objective'],'view':r['view'],
            'train_q_MSE_change':b['window_weighted_train_q_MSE']-a['window_weighted_train_q_MSE'],
            'dev_q_MSE_change':b['window_weighted_dev_q_MSE']-a['window_weighted_dev_q_MSE'],
            'dev_AP_change':b['evaluation']['video_macro_ap']-a['evaluation']['video_macro_ap'],
            'same_data_fixed_endpoint_not_checkpoint_selected':True})
    np.savez_compressed(out/'final_video_scores.npz',**final_scores)
    write(out/'evaluation_results.json',{'status':'COMPLETED','evaluation_status':'DEVELOPMENT_ONLY','runs':results,
        'seed_averaged_groups':groups,'paired_source_bootstrap_seed_averaged':contrasts,
        'all_fixed_step_trajectories':timelines,'trajectory_changes':trajectory_summary,
        'final_q_MSE_vs_AP_spearman_descriptive':rho(
            [r['steps']['256']['window_weighted_dev_q_MSE'] for r in results.values()],
            [r['final_evaluation']['video_macro_ap'] for r in results.values()]),
        'fit_human_labels_used':False,'fair_same_Qwen_comparison':False,'wall_seconds':time.monotonic()-start})


if __name__=='__main__':main()
