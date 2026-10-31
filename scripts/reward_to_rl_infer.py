#!/usr/bin/env python3
"""Complete endpoint inference, preserving every illegal window in aggregation."""
import argparse
from pathlib import Path
import numpy as np
from unsgrpo.reward_to_rl_common import *
from unsgrpo.reward_to_rl_model import *
from unsgrpo.core import parse_scores,reward_parts
from unsgrpo.data import aggregate_windows
from unsgrpo.round2_utils import overlap_statistics


def infer(seed,arm):
    validate_freeze();validate_freeze('training.freeze.json')
    receipt=read(STORE/f'training/seed{seed}_{arm}/receipt.json')
    if receipt['status']!='COMPLETED_ACTUAL_TRAINING' or receipt['completed_optimizer_steps']!=80:raise ValueError('No accepted final80 endpoint')
    adapter=Path(receipt['formal_endpoint'])
    for name,digest in receipt['endpoint_files'].items():
        if sha(adapter/name)!=digest:raise ValueError('Endpoint changed')
    plan,by,train,dev,feat,pseudo=assets();rows=records(plan['paths']['eval_pool'])
    out=STORE/f'predictions/seed{seed}_{arm}';out.mkdir(parents=True,exist_ok=False);records_out=[]
    with gpu_session(f'inference_seed{seed}_{arm}',cap=max(1,21600-gpu_used()-30)):
        engine=load_engine(Path(read(PROJECT/'reports/common_s0.json')['merged_model']),adapter)
        for n,row in enumerate(rows):
            result=generate(engine,row,greedy=True)
            try:parse_scores(result['response'],len(row['clip_ids']));legal=True
            except (ValueError,TypeError):legal=False
            r=dict(result,request_id=row['request_id'],video_id=row['video_id'],clip_ids=row['clip_ids'],
                media_signature=row['media_signature'],legal=legal,
                original_q0_reward=reward_parts(result['response'],np.asarray(pseudo[row['video_id']]['q'])[row['clip_ids']]).total)
            append(out/'windows.jsonl',r);records_out.append(r)
            if (n+1)%50==0:print(f'{seed}/{arm}: {n+1}/{len(rows)} full evaluation windows',flush=True)
        for k in dev:
            rr=[r for r in records_out if r['video_id']==k]
            scores,invalid=aggregate_windows(len(by[k]['bounds']),[(r['clip_ids'],r['response']) for r in rr])
            append(out/'videos.jsonl',{'video_id':k,'bounds':by[k]['bounds'],'scores':scores.tolist(),'invalid_window_fraction':invalid})
        import torch
        write(out/'receipt.json',{'status':'COMPLETE_FROZEN_GREEDY_PREDICTIONS','time':now(),'seed':seed,'arm':arm,
            'n_windows':len(rows),'n_sources':len(dev),'legal_fraction':sum(r['legal'] for r in records_out)/len(records_out),
            'training_receipt':reference(STORE/f'training/seed{seed}_{arm}/receipt.json'),
            'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'no_human_labels_read':True,
            'files':{str(p):sha(p) for p in (out/'windows.jsonl',out/'videos.jsonl')}})


def freeze(seed):
    paths=[]
    for arm in ('G_plain','G_guard','RSFT'):
        out=STORE/f'predictions/seed{seed}_{arm}';r=read(out/'receipt.json')
        if r['status']!='COMPLETE_FROZEN_GREEDY_PREDICTIONS':raise ValueError('No partial endpoint comparison')
        paths += [out/'windows.jsonl',out/'videos.jsonl',out/'receipt.json']
    write(STORE/f'predictions_seed{seed}.freeze.json',{'status':'ALL_THREE_PREDICTIONS_FROZEN_BEFORE_UNIFIED_EVALUATION','time':now(),
        'files':{str(p):sha(p) for p in paths},'seed':seed})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['G_plain','G_guard','RSFT']);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    freeze(a.seed) if a.freeze else infer(a.seed,a.arm)
