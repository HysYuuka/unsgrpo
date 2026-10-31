#!/usr/bin/env python3
"""No human labels: frozen rewards, eligible windows, S0 candidates, DPO pairs."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from unsgrpo.decisive_common import *
from unsgrpo.decisive_core import reward,evidence
from unsgrpo.reward_to_rl_model import load_engine,generate,prompt_encoded,encoding_audit
from unsgrpo.core import parse_scores,reward_parts


def dump_rows(path,rr):
    with path.open('x') as f:
        for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')


def dev_rewards():
    validate();inputs=read(WORK/'reused_inputs.json');rows=records(WORK/'dev_windows.jsonl');samples=records(inputs['candidates'])
    edges=records(STORE/'teacher/dev_edges.jsonl');plan,by,train,dev,features,pseudo=assets()
    result=[]
    for r in samples:
        ee=[e for e in edges if e['request_id']==r['request_id']];q=np.asarray(pseudo[r['video_id']]['q'])[r['clip_ids']]
        result.append({'request_id':r['request_id'],'video_id':r['video_id'],'g':r['g'],'legal':r['legal'],
            'new_reward':reward(r['response'],r['clip_ids'],ee),'old_reward':reward_parts(r['response'],q,.8).total})
    dump_rows(STORE/'teacher/dev_candidate_rewards.jsonl',result)
    paths=[STORE/'teacher/dev_edges.jsonl',STORE/'teacher/dev_candidate_rewards.jsonl',Path(inputs['candidates'])]
    write(STORE/'teacher/dev_rewards.freeze.json',{'status':'FROZEN_BEFORE_NEW_DEVELOPMENT_UTILITY_READ','time':now(),
        'files':{str(p):sha(p) for p in paths},'n_candidates':len(result),'human_annotation_values_read':False})


def training_candidates():
    validate();quality=read(WORK/'reward_admission.json')
    if not quality['passed']:raise ValueError('No old-reward training fallback')
    rows=records(WORK/'train_windows.initial.jsonl');edges=records(STORE/'teacher/train_edges.jsonl');good=[];coverage=[]
    for row in rows:
        ee=[r for r in edges if r['request_id']==row['request_id']];ev=evidence(row['clip_ids'],ee)
        coverage.append(dict(ev,request_id=row['request_id'],video_id=row['video_id']))
        if ev['sufficient']:good.append(row)
    write(REPORT/'train_teacher_coverage.json',{'nominal_windows':len(rows),'sufficient_windows':len(good),'sources':len({r['video_id'] for r in good}),'per_window':coverage})
    if len(good)<80 or len({r['video_id'] for r in good})<32:
        write(REPORT/'train_signal.json',{'status':'NO_GO_TRAIN_SIGNAL','reason':'directional evidence coverage before sampling','windows':len(good),'sources':len({r['video_id'] for r in good})});return
    out=STORE/'pool';out.mkdir(exist_ok=True);dump_rows(WORK/'train_windows.eligible.jsonl',good)
    with gpu_session('S0_train_candidates','student'):
        import torch
        engine=load_engine(Path(read(PROJECT/'reports/common_s0.json')['merged_model']));torch.manual_seed(31416);torch.cuda.manual_seed_all(31416);np.random.seed(31416)
        for n,row in enumerate(good):
            enc=prompt_encoded(engine,row);audit=encoding_audit(enc)
            if audit['processed_video_frames']!=32:raise ValueError('Changed student media')
            for g in range(4):
                r=generate(engine,row)
                if r['prompt_token_ids']!=enc['input_ids']:raise ValueError('Student prompt mismatch')
                try:parse_scores(r['response'],8);legal=True
                except (ValueError,TypeError):legal=False
                append(out/'candidates.jsonl',dict(r,g=g,request_id=row['request_id'],video_id=row['video_id'],legal=legal,encoding=audit))
            if (n+1)%20==0:print(f'{n+1}/{len(good)} training windows sampled',flush=True)
        write(out/'candidate_receipt.json',{'status':'FROZEN_S0_OFFLINE_TRAIN_CANDIDATES','seed':31416,'n_windows':len(good),'n_answers':4*len(good),
            'file':reference(out/'candidates.jsonl'),'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'human_labels_read':False})


def freeze_pool():
    validate();rows=records(WORK/'train_windows.eligible.jsonl');samples=records(STORE/'pool/candidates.jsonl');edges=records(STORE/'teacher/train_edges.jsonl')
    rng=np.random.default_rng(1618);good=[];pairs=[];rejected=[];all_rewards=[]
    for row in rows:
        rr=sorted([r for r in samples if r['request_id']==row['request_id']],key=lambda r:r['g']);ee=[r for r in edges if r['request_id']==row['request_id']]
        if len(rr)!=4:raise ValueError('Do not train from partial candidate cache')
        scored=[(r,reward(r['response'],row['clip_ids'],ee)) for r in rr]
        all_rewards += [{'request_id':row['request_id'],'g':r['g'],'legal':r['legal'],'reward':v} for r,v in scored]
        legal=[(r,v) for r,v in scored if r['legal']]
        if len(legal)<2:rejected.append({'request_id':row['request_id'],'reason':'fewer_than_two_legal'});continue
        hi=max(round(v,12) for r,v in legal);lo=min(round(v,12) for r,v in legal)
        if hi-lo<.10:rejected.append({'request_id':row['request_id'],'reason':'reward_gap_below_0.10'});continue
        top=[(r,v) for r,v in legal if round(v,12)==hi];bottom=[(r,v) for r,v in legal if round(v,12)==lo]
        pos,pv=top[int(rng.integers(len(top)))];neg,nv=bottom[int(rng.integers(len(bottom)))];ev=evidence(row['clip_ids'],ee)
        good.append(row);pairs.append({'request_id':row['request_id'],'video_id':row['video_id'],'chosen':pos,'rejected':neg,
            'chosen_reward':pv,'rejected_reward':nv,'reward_gap':pv-nv,'u':ev['u'],'no_human_preference':True})
    dump_rows(STORE/'pool/candidate_rewards.jsonl',all_rewards)
    if len(good)<80 or len({r['video_id'] for r in good})<32:
        write(REPORT/'train_signal.json',{'status':'NO_GO_TRAIN_SIGNAL','reason':'insufficient legal best/worst reward-gap coverage','windows':len(good),'sources':len({r['video_id'] for r in good}),'excluded':rejected});return
    dump_rows(WORK/'train_windows.final.jsonl',good);dump_rows(STORE/'pool/preferences.jsonl',pairs)
    by=defaultdict(list)
    for r in good:by[r['video_id']].append(r['request_id'])
    rng=np.random.default_rng(42);schedule=[]
    while len(schedule)<1280:
        for k in rng.permutation(sorted(by)):
            schedule.append(str(rng.choice(by[k])))
            if len(schedule)==1280:break
    write(WORK/'schedule.json',{'seed':42,'request_ids':schedule,'main_prefix':320,'long_exposures':1280,
        'main_source_counts':dict(Counter(next(r['video_id'] for r in good if r['request_id']==rid) for rid in schedule[:320])),
        'all_source_counts':dict(Counter(next(r['video_id'] for r in good if r['request_id']==rid) for rid in schedule))})
    sources=list((PROJECT/'src/unsgrpo').glob('decisive_*.py'))+list((PROJECT/'scripts').glob('decisive_*.py'))
    paths=[WORK/'train_windows.final.jsonl',WORK/'schedule.json',STORE/'pool/preferences.jsonl',STORE/'pool/candidates.jsonl',STORE/'pool/candidate_rewards.jsonl',STORE/'teacher/train_edges.jsonl']
    write(WORK/'training.freeze.json',{'status':'FROZEN_BEFORE_ANY_UPDATES','time':now(),'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in paths},
        'n_windows':len(good),'n_sources':len(by),'excluded':rejected,'actual_u_min':min(r['u'] for r in pairs),'actual_u_max':max(r['u'] for r in pairs)})
    write(REPORT/'train_signal.json',{'status':'TRAIN_SIGNAL_READY','windows':len(good),'sources':len(by),'excluded':rejected,'human_labels_read':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dev_rewards','training_candidates','freeze_pool']);a=p.parse_args();globals()[a.phase]()
