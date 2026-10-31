#!/usr/bin/env python3
"""Bounded label-free GPU screening and reusable training-cache operations."""
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from unsgrpo.reward_to_rl_common import *
from unsgrpo.reward_to_rl import *
from unsgrpo.reward_to_rl_model import *
from unsgrpo.core import parse_scores
from unsgrpo.data import canonical_hash


def teacher_row(by,vid,i,j):
    row=by[vid]
    msg=[{'role':'system','content':'Judge only visible evidence. Answer one letter only: A, B, or C.'},
        {'role':'user','content':'<image>\nChronological overview of the whole video.\n'+
         f'Clip A, original time [{row["bounds"][i][0]:.6f}, {row["bounds"][i][1]:.6f}) seconds, four actual frames:\n<video>\n'+
         f'Clip B, original time [{row["bounds"][j][0]:.6f}, {row["bounds"][j][1]:.6f}) seconds, four actual frames:\n<video>\n'+QUESTION}]
    return {'messages':msg,'images':[row['overview']],'videos':[row['frame_paths'][i],row['frame_paths'][j]]}


def pair_judgment(engine,by,vid,i,j,*,context_audit=False):
    rr=[]
    for a,b in [(i,j),(j,i)]:
        row=teacher_row(by,vid,a,b)
        for path in row['images']+[p for vv in row['videos'] for p in vv]:
            if sha(path)!=by[vid]['media_files_sha256'][path]:raise ValueError('Changed actual teacher frames')
        t=time.monotonic();score,detail=conditional_letters(engine,row,audit_context=context_audit)
        if detail['encoding']['processed_video_frames']!=8:raise ValueError('Teacher did not consume 2 x4 frames')
        rr.append({'clip_A':a,'clip_B':b,'log_probabilities_ABC':score,'detail':detail,
            'prompt_hash':canonical_hash(row['messages']),'media_paths':row['images']+row['videos'],
            'seconds':time.monotonic()-t})
    return dict(align_preference(rr[0]['log_probabilities_ABC'],rr[1]['log_probabilities_ABC']),
        clip_i=i,clip_j=j,views=rr)


def interface():
    validate_freeze();plan,by,train,dev,feat,pseudo=assets();out=STORE/'screen'
    with gpu_session('screen_teacher_interface',screen=True):
        engine=load_engine(Path('/root/autodl-fs/unsgrpo/models/Qwen3-VL-2B-Instruct'))
        total=0
        for k in sorted(train)[:8]:
            ids=read(WORK/'fixed_edge_choices.json')
            row=next(r for r in records(WORK/'train_windows.initial.jsonl') if r['video_id']==k)
            i,j=row['clip_ids'][0],row['clip_ids'][1]
            for a,b in ((i,i),(i,j),(row['clip_ids'][2],row['clip_ids'][6])):
                record=pair_judgment(engine,by,k,a,b,context_audit=True)
                append(out/'interface.jsonl',dict(record,video_id=k,self_pair=a==b));total+=1
        import torch
        write(out/'interface_receipt.json',{'status':'ACTUAL_CONDITIONAL_POSITION_MEDIA_NUMERIC_CHECK_PASSED',
            'pairs':total,'self_pairs':8,'time':now(),'forward_calls':sum(v['detail']['forward_calls'] for r in records(out/'interface.jsonl') for v in r['views']),
            'single_token_in_all_contexts':all(v['detail']['single_token_same_position'] for r in records(out/'interface.jsonl') for v in r['views']),
            'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'human_labels_read':False})


def candidates(scope):
    validate_freeze();plan,by,train,dev,feat,pseudo=assets()
    rows=records(WORK/'dev_windows.jsonl') if scope=='dev' else records(WORK/'train_windows.eligible.jsonl')
    out=STORE/('screen' if scope=='dev' else 'training_pool');out.mkdir(parents=True,exist_ok=True)
    dest=out/'candidates.jsonl'
    if dest.exists():raise FileExistsError('No candidate regeneration')
    with gpu_session('screen_S0_candidates' if scope=='dev' else 'train_S0_offline_candidates',screen=scope=='dev'):
        import torch
        seed=31415 if scope=='dev' else 31416
        engine=load_engine(Path(read(PROJECT/'reports/common_s0.json')['merged_model']))
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        for n,row in enumerate(rows):
            enc=prompt_encoded(engine,row);audit=encoding_audit(enc)
            if audit['processed_video_frames']!=4*len(row['clip_ids']):raise ValueError('Changed policy frame count')
            if len(enc['input_ids'])+128>8192:raise ValueError('Policy input would truncate')
            for g in range(4):
                before=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest();t=time.monotonic()
                result=generate(engine,row)
                if result['prompt_token_ids']!=list(enc['input_ids']):raise ValueError('Generation prompt differs from audited template')
                try:score=parse_scores(result['response'],len(row['clip_ids']));legal=True
                except (ValueError,TypeError):score=np.zeros(len(row['clip_ids']));legal=False
                append(dest,dict(result,request_id=row['request_id'],video_id=row['video_id'],g=g,
                    clip_ids=row['clip_ids'],media_signature=row['media_signature'],prompt_hash=canonical_hash(row['messages']),
                    encoding=audit,rng_before_sha256=before,sampling_seed=seed,legal=legal,scores=score.tolist(),
                    seconds=time.monotonic()-t))
            if (n+1)%10==0:print(f'{scope}: {n+1}/{len(rows)} windows sampled',flush=True)
        rr=records(dest);legal_groups=sum(all(r['legal'] for r in rr[i:i+4]) for i in range(0,len(rr),4))
        write(out/'candidates_receipt.json',{'status':'CANDIDATES_FROZEN_NO_LABEL_READS','time':now(),'scope':scope,
            'n_windows':len(rows),'n_answers':len(rr),'all_four_legal_groups':legal_groups,
            'all_four_legal_fraction':legal_groups/len(rows),'file':reference(dest),
            'sampling_seed':seed,'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'total_prompt_tokens':sum(r['usage_prompt_tokens'] for r in rr),'total_generated_tokens':sum(r['usage_completion_tokens'] for r in rr),
            'policy':'actual common BF16 S0; no updating rollout reused'})


def teacher(scope):
    validate_freeze();plan,by,train,dev,feat,pseudo=assets();edges=read(WORK/'fixed_edge_choices.json')
    rows=records(WORK/'dev_windows.jsonl') if scope=='dev' else records(WORK/'train_windows.initial.jsonl')
    out=STORE/('screen' if scope=='dev' else 'training_pool');out.mkdir(parents=True,exist_ok=True)
    dest=out/'teacher_edges.jsonl'
    if dest.exists():raise FileExistsError('No teacher regeneration')
    if scope=='dev':
        r=read(out/'candidates_receipt.json')
        if r['all_four_legal_fraction']<.8:raise ValueError('POLICY_OUTPUT_NOT_READY; do not generate developer teacher')
    with gpu_session('screen_teacher_pairs' if scope=='dev' else 'training_teacher_pairs',screen=scope=='dev'):
        engine=load_engine(Path('/root/autodl-fs/unsgrpo/models/Qwen3-VL-2B-Instruct'))
        for n,row in enumerate(rows):
            for i,j in edges[row['request_id']]:
                result=pair_judgment(engine,by,row['video_id'],i,j,context_audit=False)
                append(dest,dict(result,request_id=row['request_id'],video_id=row['video_id']))
            if (n+1)%10==0:print(f'{scope}: {n+1}/{len(rows)} windows teacher frozen',flush=True)
        import torch
        rr=records(dest)
        write(out/'teacher_receipt.json',{'status':'FROZEN_LABEL_FREE_PREFERENCE_EDGES','time':now(),'scope':scope,
            'n_windows':len(rows),'n_edges':len(rr),'n_kept':sum(r['kept'] for r in rr),
            'forward_calls':sum(v['detail']['forward_calls'] for r in rr for v in r['views']),
            'file':reference(dest),'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'abstention_counts':{reason:sum(r['abstention_reason']==reason for r in rr) for reason in sorted({r['abstention_reason'] for r in rr if r['abstention_reason'] is not None})},
            'human_labels_read':False,'policy_answers_seen_by_teacher':False})


def freeze_rewards():
    validate_freeze();plan,by,train,dev,feat,pseudo=assets();out=STORE/'screen'
    candidates=records(out/'candidates.jsonl');edges=records(out/'teacher_edges.jsonl')
    edge_map={r['request_id']:[e for e in edges if e['request_id']==r['request_id']] for r in records(WORK/'dev_windows.jsonl')}
    for r in candidates:
        q=np.asarray(pseudo[r['video_id']]['q'])[r['clip_ids']]
        rr=rewards(r['response'],q,r['clip_ids'],edge_map[r['request_id']])
        append(out/'candidate_rewards.jsonl',{'request_id':r['request_id'],'g':r['g'],'rewards':rr,
            'selection_reward_keys':{k:round(v,12) for k,v in rr.items()},'legal':r['legal']})
    write(out/'candidates_and_rewards.freeze.json',{'status':'FROZEN_BEFORE_NEW_HUMAN_LABEL_ACCESS','time':now(),
        'inputs':{str(p):sha(p) for p in [out/'candidates.jsonl',out/'teacher_edges.jsonl',out/'candidate_rewards.jsonl',WORK/'dev_windows.jsonl']},
        'n_candidates':len(candidates),'teacher_edges_independent_of_policy_answers':True,'human_labels_read':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['interface','candidates','teacher','freeze_rewards']);p.add_argument('--scope',choices=['dev','train'],default='dev');a=p.parse_args()
    if a.phase=='interface':interface()
    elif a.phase=='freeze_rewards':freeze_rewards()
    elif a.phase=='candidates':candidates(a.scope)
    else:teacher(a.scope)
