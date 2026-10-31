#!/usr/bin/env python3
"""Isolated development utility reader and the single final endpoint decision."""
import argparse
from collections import Counter,defaultdict
import numpy as np
from unsgrpo.decisive_common import *
from unsgrpo.decisive_core import evidence,reward_gate
from reward_to_rl_evaluate import describe_subset


def reward_quality():
    validate();frozen=read(STORE/'teacher/dev_rewards.freeze.json')
    for p,h in frozen['files'].items():assert sha(p)==h
    inputs=read(WORK/'reused_inputs.json');rewardrows=records(STORE/'teacher/dev_candidate_rewards.jsonl');lookup={(r['request_id'],r['g']):r for r in rewardrows}
    # Cached H is evaluation information; first value read follows new reward freeze.
    human=records(inputs['old_full_video_H']);hmap={(r['request_id'],r['g']):r for r in human}
    rows=records(WORK/'dev_windows.jsonl');edges=records(STORE/'teacher/dev_edges.jsonl');groups=[];cover=[]
    for row in rows:
        rid=row['request_id'];hh=np.array([hmap[(rid,g)]['H'] for g in range(4)]);legal=all(lookup[(rid,g)]['legal'] for g in range(4));greedy=hmap[(rid,0)]['greedy_AP']
        r={'request_id':rid,'video_id':row['video_id'],'all_four_legal':legal,'H':hh.tolist(),'H_range':float(np.ptp(hh)),
            'O':float(hh.max()-hh.mean()),'oracle_minus_greedy':float(hh.max()-greedy)}
        ii,jj=np.triu_indices(4,1);use=np.abs(hh[ii]-hh[jj])>=.001
        for label,name in [('R_old','old_reward'),('R_pair','new_reward')]:
            raw=np.array([lookup[(rid,g)][name] for g in range(4)]);key=np.round(raw,12);best=np.flatnonzero(key==key.max());selected=float(hh[best].mean())
            credit=np.where(key[ii]==key[jj],.5,((key[ii]-key[jj])*(hh[ii]-hh[jj])>0).astype(float))
            r[label]={'reward_values':raw.tolist(),'G':selected-float(hh.mean()),'selected_minus_greedy':selected-greedy,
                'discrimination_accuracy':float(credit[use].mean()) if use.any() else None,'qualifying_candidate_pairs':int(use.sum()),'best_candidates':best.tolist()}
        groups.append(r);ev=evidence(row['clip_ids'],[e for e in edges if e['request_id']==rid]);cover.append(dict(ev,video_id=row['video_id'],request_id=rid))
    primary=describe_subset([r for r in groups if r['all_four_legal']]);secondary=describe_subset(groups)
    for stats in (primary,secondary):
        for key in ('R_old','R_pair'):stats[key].pop('base_admission',None)
    counts=Counter(r['video_id'] for r in cover if r['sufficient']);coverage={'sufficient_windows':sum(r['sufficient'] for r in cover),'sources_with_two':sum(n>=2 for n in counts.values()),'source_counts':dict(counts),'per_window':cover}
    positive=sum(v['R_pair']['G']>0 for v in primary['per_video']);passed,tests=reward_gate(primary['R_pair'],primary['R_old'],coverage,positive)
    result={'status':'REWARD_READY' if passed else 'NO_GO_REWARD','passed':passed,'time':now(),'tests':tests,'coverage_all_60':coverage,
        'primary_59_all_legal':primary,'secondary_all_60':secondary,'positive_sources_G_new':positive,'per_window':groups,
        'accuracy_is_descriptive_not_60pct_admission':True,'units':'AP fractions; multiply100 for percentage points',
        'teacher_new_key':'R_pair = fixed 8B ABCD teacher, four disjoint pairs; distinct from previous 2B teacher',
        'old_reward_recomputed_on_identical_candidates':True,'candidate_H_reused_after_hash_check':reference(inputs['old_full_video_H']),
        'no_old_reward_training_fallback':True,'label_boundary':reference(STORE/'teacher/dev_rewards.freeze.json')}
    write(REPORT/'reward_quality.json',result);print(result['status']+' '+str(tests),flush=True)
    write(WORK/'reward_admission.json',{'status':result['status'],'passed':passed,'time':now(),
        'evaluation_receipt':reference(REPORT/'reward_quality.json'),'purpose':'Boolean control for training; contains no human AP vectors or labels'})


def final_endpoints():
    validate();frozen=read(STORE/'endpoints_and_predictions.freeze.json')
    for p,h in frozen['files'].items():assert sha(p)==h
    from unsgrpo.round2_metrics import DevelopmentEvaluator
    from unsgrpo.evaluation import paired_bootstrap
    plan,by,train,dev,features,pseudo=assets();ev=DevelopmentEvaluator(plan,by,dev);methods={}
    for method in ('P-DPO-80','P-DPO-long','G-plain','G-guard'):
        rr=records(STORE/'predictions'/method/'videos.jsonl');methods[method]=ev.evaluate({r['video_id']:r['scores'] for r in rr})
    for method,slug in [('S0','S0'),('G-CAL','G-CAL')]:
        rr=records(ORIGINAL/'inference'/slug/'videos.jsonl');methods[method]=ev.evaluate({r['video_id']:r['scores'] for r in rr})
    dpo='P-DPO-80' if methods['P-DPO-80']['video_macro_ap']>=methods['P-DPO-long']['video_macro_ap'] else 'P-DPO-long'
    tests={};contrasts={}
    for method in ('G-plain','G-guard'):
        aa={r['video_id']:r['mean_ap'] for r in methods[method]['per_video']};bb={r['video_id']:r['mean_ap'] for r in methods[dpo]['per_video']};ids=sorted(aa);delta=np.array([aa[k]-bb[k] for k in ids])
        tests[method]={'gain_S0_at_least_1pp':methods[method]['video_macro_ap']-methods['S0']['video_macro_ap']>=.01,
            'gain_stronger_DPO_at_least_half_pp':delta.mean()>=.005,'not_below_GCAL':methods[method]['video_macro_ap']>=methods['G-CAL']['video_macro_ap'],
            'at_least_6_sources_improve':int((delta>0).sum())>=6,'all_leave_one_source_out_gains_positive':all(np.delete(delta,i).mean()>0 for i in range(len(delta)))}
        for reference_name in ('P-DPO-80','P-DPO-long','S0','G-CAL'):
            ref={r['video_id']:r['mean_ap'] for r in methods[reference_name]['per_video']};values=[aa[k]-ref[k] for k in ids]
            contrasts[method+'__minus__'+reference_name]=paired_bootstrap(values,source_groups=ids,n_boot=10000,seed=42)
    for first,second in [('G-guard','G-plain'),('P-DPO-long','P-DPO-80')]:
        aa={r['video_id']:r['mean_ap'] for r in methods[first]['per_video']};bb={r['video_id']:r['mean_ap'] for r in methods[second]['per_video']};ids=sorted(aa)
        contrasts[first+'__minus__'+second]=paired_bootstrap([aa[k]-bb[k] for k in ids],source_groups=ids,n_boot=10000,seed=42)
    guard=all(tests['G-guard'].values());plain=all(tests['G-plain'].values())
    if guard:decision='GO_GUARDED_RL'
    elif plain:decision='GO_STANDARD_RL_ONLY'
    else:
        b={r['video_id']:r['mean_ap'] for r in methods['S0']['per_video']}
        teacher_signal=methods[dpo]['video_macro_ap']-methods['S0']['video_macro_ap']>=.01 and sum(r['mean_ap']>b[r['video_id']] for r in methods[dpo]['per_video'])>=6
        decision='TEACHER_ONLY_SIGNAL' if teacher_signal else 'NO_GO_RL_INCREMENT'
    write(REPORT/'final_evaluation.json',{'status':decision,'time':now(),'methods':methods,'stronger_whole_macro_DPO':dpo,'continuation_tests':tests,
        'all_raw_GRPO_vs_each_DPO_comparisons':contrasts,'single_continuation_seed':42,'source_bootstrap_does_not_include_training_randomness':True,
        'selected_DPO_is_preregistered_conservative_resource_rule_not_selection_corrected_significance':True})
    print(decision,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['reward_quality','final_endpoints']);a=p.parse_args();globals()[a.phase]()
