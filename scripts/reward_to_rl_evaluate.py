#!/usr/bin/env python3
"""Isolated human-label evaluation: one-window replacement, then optional endpoints."""
import argparse
import csv
from collections import defaultdict
import numpy as np
from unsgrpo.reward_to_rl_common import *
from unsgrpo.core import parse_scores
from unsgrpo.data import aggregate_windows
from unsgrpo.round2_metrics import DevelopmentEvaluator


def describe_subset(rows):
    by=defaultdict(list)
    for r in rows:by[r['video_id']].append(r)
    per=[]
    for k,group in sorted(by.items()):
        entry={'video_id':k,'n_groups':len(group),'O':float(np.mean([r['O'] for r in group])),
            'oracle_minus_greedy':float(np.mean([r['oracle_minus_greedy'] for r in group])),
            'n_informative_groups':sum(r['H_range']>=.001 for r in group)}
        for name in ('R_old','R_pair'):
            a=[r[name]['discrimination_accuracy'] for r in group if r[name]['discrimination_accuracy'] is not None]
            entry[name]={'G':float(np.mean([r[name]['G'] for r in group])),
                'selected_minus_greedy':float(np.mean([r[name]['selected_minus_greedy'] for r in group])),
                'accuracy':float(np.mean(a)) if a else None,'groups_with_discrimination_pairs':len(a),
                'qualifying_candidate_pairs':sum(r[name]['qualifying_candidate_pairs'] for r in group)}
        per.append(entry)
    if not per:return {'n_sources':0,'per_video':[],'O':None,'oracle_minus_greedy':None,'R_old':None,'R_pair':None}
    oracle=float(np.mean([r['O'] for r in per]));result={'n_sources':len(per),'per_video':per,'O':oracle,
        'oracle_minus_greedy':float(np.mean([r['oracle_minus_greedy'] for r in per])),
        'n_informative_groups':sum(r['n_informative_groups'] for r in per),
        'sources_with_informative_groups':sum(r['n_informative_groups']>0 for r in per)}
    rng=np.random.default_rng(42);boot=rng.integers(0,len(per),(10000,len(per)))
    oo=np.array([r['O'] for r in per])[boot].mean(1)
    result['O_CI']=np.quantile(oo,[.025,.975]).tolist()
    og=np.array([r['oracle_minus_greedy'] for r in per])[boot].mean(1)
    result['oracle_minus_greedy_CI']=np.quantile(og,[.025,.975]).tolist()
    for name in ('R_old','R_pair'):
        G=float(np.mean([r[name]['G'] for r in per]));dis=[r[name]['accuracy'] for r in per if r[name]['accuracy'] is not None]
        vs=float(np.mean([r[name]['selected_minus_greedy'] for r in per]));acc=float(np.mean(dis)) if dis else None
        gg=np.array([r[name]['G'] for r in per])[boot].mean(1)
        vsboot=np.array([r[name]['selected_minus_greedy'] for r in per])[boot].mean(1)
        accarray=np.array([r[name]['accuracy'] if r[name]['accuracy'] is not None else np.nan for r in per])
        aboot=accarray[boot];counts=np.isfinite(aboot).sum(1)
        aval=np.nansum(aboot,axis=1)[counts>0]/counts[counts>0]
        result[name]={'G':G,'accuracy':acc,'G_over_O':G/oracle if oracle>0 else None,
            'selected_minus_greedy':vs,'G_CI':np.quantile(gg,[.025,.975]).tolist(),
            'selected_minus_greedy_CI':np.quantile(vsboot,[.025,.975]).tolist(),
            'accuracy_CI':np.quantile(aval,[.025,.975]).tolist() if len(aval) else None,
            'G_over_O_CI':np.quantile(gg[oo>0]/oo[oo>0],[.025,.975]).tolist() if np.any(oo>0) else None,
            'accuracy_sources':len(dis),'qualifying_candidate_pairs':sum(r[name]['qualifying_candidate_pairs'] for r in per),
            'base_admission':acc is not None and acc>=.6 and G>0 and oracle>0 and G/oracle>=.2 and vs>=0}
    return result


def screen():
    validate_freeze();sealed=read(STORE/'screen/candidates_and_rewards.freeze.json')
    for p,h in sealed['inputs'].items():
        if sha(p)!=h:raise ValueError('Candidates/rewards changed before label access')
    plan,by,train,dev,feat,pseudo=assets()
    candidates=records(STORE/'screen/candidates.jsonl');rewards=records(STORE/'screen/candidate_rewards.jsonl')
    rlookup={(r['request_id'],r['g']):r for r in rewards}
    windows=records(WORK/'dev_windows.jsonl');cached=records(ORIGINAL/'inference/S0/windows.jsonl')
    original={k:[r for r in cached if r['video_id']==k] for k in dev}
    # Only now instantiate the evaluation-only reader, after every reward is frozen.
    ev=DevelopmentEvaluator(plan,by,dev);base={};full_base={}
    for k in dev:
        full_base[k]=aggregate_windows(len(by[k]['bounds']),[(r['clip_ids'],r['response']) for r in original[k]])[0]
        base[k]=ev.score(k,full_base[k])
    groups=[];candidate_metrics=[]
    for row in windows:
        k=row['video_id'];rid=row['request_id'];rr=sorted([r for r in candidates if r['request_id']==rid],key=lambda r:r['g'])
        if len(rr)!=4:raise ValueError('Incomplete group')
        hh=[]
        for sample in rr:
            predictions=[(r['clip_ids'],sample['response'] if r['request_id']==rid else r['response']) for r in original[k]]
            scores,invalid=aggregate_windows(len(by[k]['bounds']),predictions);score=ev.score(k,scores);hh.append(score['mean_ap'])
            detail={'request_id':rid,'video_id':k,'g':sample['g'],'legal':sample['legal'],'H':score['mean_ap'],
                'H_by_annotator':score['ap_by_annotator'],'full_video_scores_sha256':__import__('hashlib').sha256(scores.tobytes()).hexdigest(),
                'greedy_AP':base[k]['mean_ap'],'invalid_window_fraction':invalid,
                'rewards':rlookup[(rid,sample['g'])]['rewards']}
            candidate_metrics.append(detail)
        h=np.array(hh);random=float(h.mean());oracle=float(h.max());baseline=base[k]['mean_ap']
        result={'request_id':rid,'video_id':k,'all_four_legal':all(r['legal'] for r in rr),'H':hh,
            'H_range':float(np.ptp(h)),'random_AP':random,'oracle_AP':oracle,'greedy_AP':baseline,
            'O':oracle-random,'oracle_minus_greedy':oracle-baseline}
        ii,jj=np.triu_indices(4,1);usable=np.abs(h[ii]-h[jj])>=.001
        for name in ('R_old','R_pair'):
            raw=np.array([rlookup[(rid,i)]['rewards'][name] for i in range(4)])
            key=np.round(raw,12);best=np.flatnonzero(key==key.max());selected=float(h[best].mean())
            credit=np.where(key[ii]==key[jj],.5,((key[ii]-key[jj])*(h[ii]-h[jj])>0).astype(float))
            result[name]={'reward_values':raw.tolist(),'best_candidate_indices':best.tolist(),'selected_AP':selected,
                'G':selected-random,'selected_minus_greedy':selected-baseline,
                'qualifying_candidate_pairs':int(usable.sum()),'discrimination_accuracy':float(credit[usable].mean()) if usable.any() else None}
        groups.append(result)
    legal=[r for r in groups if r['all_four_legal']];primary=describe_subset(legal);allgroups=describe_subset(groups)
    information=(primary['n_informative_groups']>=30 and primary['sources_with_informative_groups']>=8 and
        primary['O']>=.0005 and primary['oracle_minus_greedy']>0)
    chosen=None
    if information:
        old=primary['R_old'];new=primary['R_pair']
        if new['base_admission'] and (not old['base_admission'] or (new['accuracy']>=old['accuracy']+.03 and new['G']>=old['G'])):chosen='R_pair'
        elif old['base_admission']:chosen='R_old'
    decision='CANDIDATE_POOL_UNINFORMATIVE' if not information else 'REWARD_SELECTED_FOR_TRAINING' if chosen else 'REWARD_NOT_ALIGNED'
    result={'status':decision,'time':now(),'selected_reward':chosen,'development_selection_disclosed':True,
        'information_gate_passed':information,'n_groups':len(groups),'n_all_four_legal':len(legal),
        'primary_all_four_legal':primary,'secondary_all_groups':allgroups,'per_window':groups,
        'baseline_per_video':list(base.values()),'label_boundary':reference(STORE/'screen/candidates_and_rewards.freeze.json'),
        'units':'AP in [0,1]; 0.001 =0.1 percentage point; video macro, windows averaged within video',
        'bootstrap':'10000 paired source resamples seed42; windows are not independent',
        'oracle_used_for_training':False,'single_window_gains_are_not_additive':True}
    write(REPORT/'screen_result.json',result)
    with (STORE/'screen/single_window_full_video_AP.jsonl').open('x') as f:
        for r in candidate_metrics:f.write(__import__('json').dumps(r)+'\n')
    with (REPORT/'screen_per_window.csv').open('x') as f:
        fields=['video_id','request_id','all_four_legal','H_range','O','oracle_minus_greedy','R_old_G','R_pair_G','R_old_accuracy','R_pair_accuracy']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in groups:writer.writerow({**{k:r[k] for k in fields[:6]},'R_old_G':r['R_old']['G'],'R_pair_G':r['R_pair']['G'],
            'R_old_accuracy':r['R_old']['discrimination_accuracy'],'R_pair_accuracy':r['R_pair']['discrimination_accuracy']})
    print(__import__('json').dumps({'status':decision,'selected_reward':chosen,'primary':{k:v for k,v in primary.items() if k!='per_video'}}),flush=True)


def endpoints(seed):
    plan,by,train,dev,feat,pseudo=assets();sealed=read(STORE/f'predictions_seed{seed}.freeze.json')
    for p,h in sealed['files'].items():
        if sha(p)!=h:raise ValueError('Endpoint prediction changed')
    ev=DevelopmentEvaluator(plan,by,dev);methods={}
    for arm in ('G_plain','G_guard','RSFT'):
        rr=records(STORE/f'predictions/seed{seed}_{arm}/videos.jsonl')
        methods[arm]=ev.evaluate({r['video_id']:r['scores'] for r in rr})
    historical=read(PROJECT/'reports/round3_rl_method_v1/historical_baselines.json')
    refs={'S0':historical['S0']['evaluation'],'G-CAL':historical['G-CAL']['evaluation'],'RSFT':methods['RSFT']}
    best=max(refs,key=lambda k:refs[k]['video_macro_ap']);bestvalue=refs[best]['video_macro_ap'];passing=[]
    for arm in ('G_plain','G_guard'):
        old={r['video_id']:r['mean_ap'] for r in refs[best]['per_video']}
        positive=sum(r['mean_ap']>old[r['video_id']] for r in methods[arm]['per_video'])
        if methods[arm]['video_macro_ap']>=bestvalue+.01 and positive>=6:passing.append(arm)
    write(REPORT/f'endpoints_seed{seed}.json',{'status':'DEVELOPMENT_ENDPOINTS_EVALUATED','time':now(),
        'seed':seed,'methods':methods,'repeat_gate_passed':bool(passing),'passing_arms':passing,
        'repeat_reference':best,'reference_macro_AP':bestvalue,'human_checkpoint_selection':False,
        'scope':'continuation randomness from one shared S0, not full-pipeline seeds'})
    print('Endpoints evaluated '+str(seed)+'; repeat gate '+str(bool(passing)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['screen','endpoints']);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    screen() if a.phase=='screen' else endpoints(a.seed)
