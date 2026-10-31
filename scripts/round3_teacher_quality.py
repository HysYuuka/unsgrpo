#!/usr/bin/env python3
"""Isolated development-label process, after semantic and target cache freezes."""
import argparse
import csv
import numpy as np
from unsgrpo.round3_common import *
from unsgrpo.round2_metrics import DevelopmentEvaluator, rho
from unsgrpo.round2_utils import dispersion


def agreement(s,y,ii,jj):
    a=np.sign(s[ii]-s[jj]);b=np.sign(y[ii]-y[jj])
    # Both a human tie and a predicted tie receive 1/2; no preferred direction.
    value=np.where((a==0)|(b==0),.5,(a==b).astype(float))
    return float(value.mean()) if len(value) else None


def main(alpha):
    tag='alpha075' if alpha==.75 else 'alpha050';out=STORE/'teacher'
    if (REPORT/f'teacher_gate_{tag}.json').exists():raise FileExistsError('No repeated quality-driven selection')
    freeze=read(out/f'targets_{tag}.freeze.json')
    if sha(freeze['arrays']['path'])!=freeze['arrays']['sha256']:raise ValueError('Targets changed before quality check')
    plan,by,train,dev,feat,pseudo=assets();sealed=read(out/'semantic.freeze.json')
    qfile=np.load(freeze['arrays']['path'],allow_pickle=False)
    q={k:qfile['q__'+k] for k in by};c={k:qfile['c__'+k] for k in by}
    reference=np.load(out/'reference_targets.npz',allow_pickle=False)
    # This is the first human-label access in this script. Teacher fitting ended.
    ev=DevelopmentEvaluator(plan,by,dev)
    base=ev.evaluate({k:pseudo[k]['q'] for k in dev});new=ev.evaluate({k:q[k] for k in dev})
    excl=ev.evaluate({k:reference['q_excl__'+k] for k in dev})
    pools=read(WORK/'teacher_pair_pool.json')['pairs'];per=[]
    for k in dev:
        ii,jj=np.asarray(pools[k],dtype=int).T;y=ev.clip_labels[k]
        use=(c[k][ii]>=.75)&(c[k][jj]>=.75)&(np.abs(q[k][ii]-q[k][jj])>=.15)
        high_i,high_j=ii[use],jj[use]
        r={'video_id':k,'source_group':by[k]['source_group'],'alpha':alpha,
           'preselected_pairs':len(ii),'retained_pairs':int(use.sum()),'retained_pair_fraction':float(use.mean()),
           'high_confidence_clip_fraction':float(np.mean(c[k]>=.75)),
           'new_same_pairs_agreement':agreement(q[k],y,high_i,high_j),
           'old_same_pairs_agreement':agreement(np.asarray(pseudo[k]['q']),y,high_i,high_j),
           'new_all_pairs_agreement':agreement(q[k],y,ii,jj),
           'old_all_pairs_agreement':agreement(np.asarray(pseudo[k]['q']),y,ii,jj),
           'new_V0_AP':next(v['mean_ap'] for v in new['per_video'] if v['video_id']==k),
           'old_V0_AP':next(v['mean_ap'] for v in base['per_video'] if v['video_id']==k),
           'q_mean':float(q[k].mean()),'q_std':float(q[k].std()),'q_pair_tie_fraction':dispersion(q[k])['pair_tie_fraction'],
           'q_vs_human_clip_rho':rho(q[k],y)}
        per.append(r)
    windows=records(plan['paths']['train_pool']);counts=[]
    for r in windows:
        k=r['video_id'];ids=r['clip_ids'];qq=q[k][ids];cc=c[k][ids];ii,jj=np.triu_indices(len(ids),1)
        count=int(((cc[ii]>=.75)&(cc[jj]>=.75)&(np.abs(qq[ii]-qq[jj])>=.15)).sum())
        counts.append(count)
    available=[r for r in per if r['retained_pairs']>0]
    new_agree=float(np.mean([r['new_same_pairs_agreement'] for r in available])) if available else None
    old_agree=float(np.mean([r['old_same_pairs_agreement'] for r in available])) if available else None
    coverage=float(np.mean([r['high_confidence_clip_fraction'] for r in per]))
    train_coverage=float(np.mean(np.asarray(counts)>=3))
    delta=(new['video_macro_ap']-base['video_macro_ap'])*100
    tests={'format_precheck':sealed['early_stop'] is None,
        'high_confidence_pair_agreement_at_least_60pct':new_agree is not None and new_agree>=.60,
        'same_pairs_improvement_at_least_3pp':new_agree is not None and new_agree-old_agree>=.03,
        'high_confidence_clip_coverage_at_least_30pct':coverage>=.30,
        'at_least_8_development_videos_with_pairs':len(available)>=8,
        'half_training_pool_windows_have_at_least_3_pairs':train_coverage>=.50,
        'primary_direct_score_not_worse_than_1pp':delta>=-1.,
        'V0_direct_score_not_worse_than_1pp':delta>=-1.}
    result={'status':'TEACHER_GATE_PASSED' if all(tests.values()) else 'TEACHER_GATE_FAILED',
        'passed':all(tests.values()),'alpha':alpha,'time':timestamp(),'tests':tests,
        'quality_macro':new_agree,'old_on_same_pairs_macro':old_agree,
        'lift_pp':100*(new_agree-old_agree) if new_agree is not None else None,
        'high_confidence_clip_coverage_video_macro':coverage,
        'high_confidence_clip_coverage_pooled':float(np.concatenate([c[k]>=.75 for k in dev]).mean()),
        'available_development_videos':len(available),'training_pool_usable_windows_fraction':train_coverage,
        'training_pool_windows':len(windows),'training_unique_windows':len({r['request_id'] for r in windows}),
        'direct_primary_delta_pp':delta,'primary':'V0-stable-mapper-round12','public_top5':'NOT_READY',
        'T0':base,'T1':new,'q_excl_direct':excl,'per_video':per,
        'frozen_targets':reference_path(freeze['arrays']),
        'same_pairs_control':True,'multiple_alpha_results':'Fallback permitted only after primary fails',
        'interpretation':'Project admission rule, not a statistical significance test; ten DEVELOPMENT sources'}
    write(REPORT/f'teacher_gate_{tag}.json',result)
    with (REPORT/f'teacher_quality_{tag}.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(per[0]));writer.writeheader();writer.writerows(per)
    print(json.dumps({'status':result['status'],'alpha':alpha,'tests':tests,'new_agreement':new_agree,
        'old_on_same_pairs':old_agree,'V0_delta_pp':delta}),flush=True)


def reference_path(x):return dict(x)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--alpha',type=float,choices=[.75,.5],required=True);a=p.parse_args();main(a.alpha)
