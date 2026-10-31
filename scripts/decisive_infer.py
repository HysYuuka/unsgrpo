#!/usr/bin/env python3
"""All four endpoints must exist before any complete development inference."""
import argparse
from unsgrpo.decisive_common import *
from unsgrpo.reward_to_rl_model import load_engine,generate
from unsgrpo.core import parse_scores
from unsgrpo.data import aggregate_windows

METHODS={'P-DPO-80':('P-DPO',80),'P-DPO-long':('P-DPO',320),'G-plain':('G-plain',80),'G-guard':('G-guard',80)}


def endpoints():
    files={};endpoints={}
    for name,(arm,step) in METHODS.items():
        receipt_path=STORE/'training'/arm/f'receipt_{step}.json';r=read(receipt_path)
        if r['status']!='COMPLETED_ACTUAL_TRAINING' or r['completed_optimizer_steps']!=(240 if step==320 else 80):raise ValueError('All four complete endpoints required')
        endpoint=Path(r['endpoint']);endpoints[name]=str(endpoint);files[str(receipt_path)]=sha(receipt_path)
        for p,h in r['endpoint_files'].items():
            if sha(endpoint/p)!=h:raise ValueError('Endpoint identity changed')
            files[str(endpoint/p)]=h
    return files,endpoints


def infer(method):
    validate();validate('training.freeze.json');files,paths=endpoints()
    plan,by,train,dev,feat,pseudo=assets();rows=records(plan['paths']['eval_pool']);assert len(rows)==346
    out=STORE/'predictions'/method;out.mkdir(parents=True,exist_ok=False);pred=[]
    with gpu_session('inference_'+method,'evaluation'):
        import torch
        engine=load_engine(Path(read(PROJECT/'reports/common_s0.json')['merged_model']),Path(paths[method]))
        for n,row in enumerate(rows):
            result=generate(engine,row,greedy=True)
            try:parse_scores(result['response'],len(row['clip_ids']));legal=True
            except (ValueError,TypeError):legal=False
            r=dict(result,request_id=row['request_id'],video_id=row['video_id'],clip_ids=row['clip_ids'],media_signature=row['media_signature'],legal=legal)
            append(out/'windows.jsonl',r);pred.append(r)
            if (n+1)%50==0:print(f'{method}: {n+1}/346 complete inference windows',flush=True)
        for k in dev:
            rr=[r for r in pred if r['video_id']==k];scores,invalid=aggregate_windows(len(by[k]['bounds']),[(r['clip_ids'],r['response']) for r in rr])
            append(out/'videos.jsonl',{'video_id':k,'scores':scores.tolist(),'bounds':by[k]['bounds'],'invalid_window_fraction':invalid})
        write(out/'receipt.json',{'status':'COMPLETE_FROZEN_GREEDY_PREDICTIONS','method':method,'time':now(),'n_windows':len(rows),'n_sources':len(dev),
            'legal_fraction':sum(r['legal'] for r in pred)/len(pred),'endpoint':paths[method],'no_human_labels_read':True,
            'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'files':{str(p):sha(p) for p in (out/'windows.jsonl',out/'videos.jsonl')}})


def freeze():
    validate();files,paths=endpoints()
    for method in METHODS:
        out=STORE/'predictions'/method;r=read(out/'receipt.json')
        if r['status']!='COMPLETE_FROZEN_GREEDY_PREDICTIONS':raise ValueError('Partial endpoint evaluation forbidden')
        files[str(out/'receipt.json')]=sha(out/'receipt.json')
        for p,h in r['files'].items():
            assert sha(p)==h;files[p]=h
    write(STORE/'endpoints_and_predictions.freeze.json',{'status':'ALL_FOUR_COMPLETE_BEFORE_NEW_ENDPOINT_HUMAN_AP','time':now(),'files':files,'endpoints':paths})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=list(METHODS));p.add_argument('--freeze',action='store_true');a=p.parse_args()
    freeze() if a.freeze else infer(a.method)
