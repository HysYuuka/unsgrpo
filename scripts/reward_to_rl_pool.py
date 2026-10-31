#!/usr/bin/env python3
"""Label-free frozen training-pool eligibility and reward-selected SFT answers."""
import argparse
from collections import defaultdict,Counter
import json
import numpy as np
from unsgrpo.reward_to_rl_common import *
from unsgrpo.reward_to_rl import *


def write_rows(path,rr):
    with path.open('x') as f:
        for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')


def eligible(force_old=False):
    validate_freeze();screen=read(REPORT/'screen_result.json')
    if screen['status']!='REWARD_SELECTED_FOR_TRAINING':raise ValueError('No admitted reward')
    selected='R_old' if force_old else screen['selected_reward']
    if force_old and not screen['primary_all_four_legal']['R_old']['base_admission']:raise ValueError('Old reward never passed admission')
    rows=records(WORK/'train_windows.initial.jsonl');out=STORE/'training_pool';out.mkdir(parents=True,exist_ok=True)
    edges=records(out/'teacher_edges.jsonl') if selected=='R_pair' else []
    edge_map={r['request_id']:[e for e in edges if e['request_id']==r['request_id']] for r in rows}
    good=[];coverage=[]
    for row in rows:
        use=[e for e in edge_map[row['request_id']] if e['kept']]
        covered={i for e in use for i in (e['clip_i'],e['clip_j'])}
        valid=selected=='R_old' or (len(use)>=3 and len(covered)>=4)
        if valid:good.append(row)
        coverage.append({'request_id':row['request_id'],'video_id':row['video_id'],'kept_edges':len(use),
            'covered_clips':len(covered),'eligible':valid})
    sources=len({r['video_id'] for r in good});passed=len(good)>=80 and sources>=32
    write(out/('coverage_R_old.json' if selected=='R_old' else 'coverage_R_pair.json'),{
        'reward':selected,'n_windows':len(good),'n_sources':sources,'passed':passed,'per_window':coverage})
    if not passed:
        if selected=='R_pair' and screen['primary_all_four_legal']['R_old']['base_admission']:
            selected='R_old';good=rows;sources=40;force_old=True
        else:
            write(REPORT/'train_pool_status.json',{'status':'TRAIN_SIGNAL_TOO_SPARSE','windows':len(good),'sources':sources});return
    write_rows(WORK/'train_windows.eligible.jsonl',good)
    write(out/'reward_choice.json',{'status':'ELIGIBLE_POOL_FROZEN','selected_reward':selected,
        'screen_selected_reward':screen['selected_reward'],'single_predeclared_old_reward_fallback_used':force_old,
        'n_windows':len(good),'n_sources':sources,'human_labels_read':False})


def finalize():
    out=STORE/'training_pool';choice=read(out/'reward_choice.json');selected=choice['selected_reward']
    rows=records(WORK/'train_windows.eligible.jsonl');samples=records(out/'candidates.jsonl')
    edges=records(out/'teacher_edges.jsonl') if selected=='R_pair' else []
    edge_map={r['request_id']:[e for e in edges if e['request_id']==r['request_id']] for r in rows}
    rng=np.random.default_rng(1618);targets=[];good=[];removed=[]
    for row in rows:
        rr=sorted([r for r in samples if r['request_id']==row['request_id']],key=lambda r:r['g'])
        if len(rr)!=4:raise ValueError('Offline pool incomplete; do not use partial candidates')
        legal=[]
        for sample in rr:
            value=rewards(sample['response'],row['pseudo_scores'],row['clip_ids'],edge_map[row['request_id']])[selected]
            append(out/'offline_candidate_rewards.jsonl',{'request_id':row['request_id'],'g':sample['g'],
                'selected_reward':selected,'reward':value,'legal':sample['legal']})
            if sample['legal']:legal.append((sample,value))
        if not legal:removed.append(row['request_id']);continue
        maxvalue=max(round(v,12) for r,v in legal);ties=[(r,v) for r,v in legal if round(v,12)==maxvalue]
        sample,value=ties[int(rng.integers(len(ties)))];u=reliability(selected,row['clip_ids'],edge_map[row['request_id']])
        targets.append({'request_id':row['request_id'],'video_id':row['video_id'],'response':sample['response'],
            'token_ids':sample['token_ids'],'prompt_token_ids':sample['prompt_token_ids'],'candidate_g':sample['g'],
            'reward':value,'u':u,'reward_name':selected,'tie_count':len(ties),'no_human_selection':True})
        good.append(row)
    by=defaultdict(list)
    for row in good:by[row['video_id']].append(row['request_id'])
    if len(good)<80 or len(by)<32:
        write(REPORT/'train_pool_status.json',{'status':'TRAIN_SIGNAL_TOO_SPARSE','post_legality_windows':len(good),'post_legality_sources':len(by),'removed':removed});return
    ids=sorted(by);rng=np.random.default_rng(42);schedule=[]
    while len(schedule)<320:
        for k in rng.permutation(ids):
            schedule.append(str(rng.choice(by[k])))
            if len(schedule)==320:break
    assert max(Counter(next(row['video_id'] for row in good if row['request_id']==rid) for rid in schedule).values())-min(Counter(next(row['video_id'] for row in good if row['request_id']==rid) for rid in schedule).values())<=1
    write_rows(WORK/'train_windows.final.jsonl',good);write_rows(out/'RSFT_targets.jsonl',targets)
    write(WORK/'training_schedule.json',{'seed':42,'request_ids':schedule,'exposures':320,
        'source_exposure_counts':dict(Counter(next(row['video_id'] for row in good if row['request_id']==rid) for rid in schedule))})
    sourcepaths=list((PROJECT/'src/unsgrpo').glob('reward_to_rl*.py'))+list((PROJECT/'scripts').glob('reward_to_rl*.py'))
    write(WORK/'training.freeze.json',{'status':'FROZEN_BEFORE_REAL_UPDATES','time':now(),'selected_reward':selected,
        'sources':{str(p):sha(p) for p in sourcepaths},'inputs':{str(p):sha(p) for p in [WORK/'train_windows.final.jsonl',WORK/'training_schedule.json',
            out/'RSFT_targets.jsonl',out/'candidates.jsonl',out/'offline_candidate_rewards.jsonl',out/'reward_choice.json']+
            ([out/'teacher_edges.jsonl'] if selected=='R_pair' else [])},'n_windows':len(good),'n_sources':len(by),
        'removed_no_legal_candidate':removed,'same_teacher_same_pool_same_exposures_all_arms':True,'human_train_labels_read':False})
    write(REPORT/'train_pool_status.json',{'status':'TRAIN_POOL_READY','selected_reward':selected,'windows':len(good),'sources':len(by),
        'removed_no_legal_candidate':removed,'fixed_exposures':320})
    print(f'Frozen {len(good)} training windows / {len(by)} sources; reward {selected}; 320 matched exposures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['eligible','finalize']);p.add_argument('--force-old',action='store_true');a=p.parse_args()
    eligible(a.force_old) if a.phase=='eligible' else finalize()
