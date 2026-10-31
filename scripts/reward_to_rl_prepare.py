#!/usr/bin/env python3
"""Freeze label-free window choices, baseline identity and all screening rules."""
import copy
import json
import shutil
from pathlib import Path
import numpy as np
from unsgrpo.reward_to_rl_common import *
from unsgrpo.reward_to_rl import uniform_indices,selected_edges,QUESTION
from unsgrpo.data import build_window_record,canonical_hash,audit_window_record,window_indices,aggregate_windows


def main():
    if (WORK/'screen_protocol.freeze.json').exists():raise FileExistsError('Already frozen')
    plan,by,train,dev,feat,pseudo=assets()
    if '5 passed' not in (WORK/'logs/cpu_contracts.log').read_text():raise ValueError('CPU contracts not passed')
    frozen_predictions=read(ORIGINAL/'predictions.freeze.json')
    if frozen_predictions['status']!='FROZEN':raise ValueError('Baseline index not frozen')
    for entry in (frozen_predictions['runs']['S0'],frozen_predictions['runs']['S0']['receipt']):
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed frozen S0 baseline artifact')
    old_receipt=read(ORIGINAL/'inference/S0/receipt.json')
    if old_receipt['common_s0_sha256']!=sha(PROJECT/'reports/common_s0.json'):raise ValueError('Baseline used another common S0')
    previous=read(R2/'round2.freeze.json')
    window_path=str(ORIGINAL/'inference/S0/windows.jsonl')
    if sha(window_path)!=previous['inputs'][window_path]:raise ValueError('Baseline window cache changed since round two')
    expected=records(plan['paths']['eval_pool']);greedy=records(ORIGINAL/'inference/S0/windows.jsonl')
    lookup={r['request_id']:r for r in greedy}
    if len(lookup)!=len(greedy) or set(lookup)!={r['request_id'] for r in expected}:raise ValueError('Incomplete/duplicate cached greedy predictions')
    selected=[];trainrows=[]
    for k in dev:
        group=[r for r in expected if r['video_id']==k]
        if [r['clip_ids'] for r in group]!=window_indices(len(by[k]['bounds'])):raise ValueError('Window definition differs')
        for r in group:
            old=lookup[r['request_id']]
            for field in ('video_id','clip_ids','media_signature'):
                if old[field]!=r[field]:raise ValueError('Greedy cache identity mismatch')
        baseline=next(r for r in records(ORIGINAL/'inference/S0/videos.jsonl') if r['video_id']==k)
        rebuilt=aggregate_windows(len(by[k]['bounds']),[(lookup[r['request_id']]['clip_ids'],lookup[r['request_id']]['response']) for r in group])[0]
        if not np.allclose(rebuilt,baseline['scores'],rtol=0,atol=1e-15):raise ValueError('Greedy window aggregation does not reproduce frozen video scores')
        for j in uniform_indices(len(group),6):
            row=copy.deepcopy(group[j]);audit_window_record(row,check_media_files=True)
            selected.append(row)
    for k in train:
        row=by[k];windows=window_indices(len(row['bounds']))
        for j in uniform_indices(len(windows),4):
            trainrows.append(build_window_record(video_id=k,source_group=row['source_group'],split='train',
                duration=row['duration'],bounds=row['bounds'],frame_paths=row['frame_paths'],sampled_pts=row['sampled_pts'],
                overview_path=row['overview'],clip_ids=windows[j],pseudo_scores=pseudo[k]['q'],task='grpo'))
    if len(selected)!=60 or len(trainrows)!=160:raise ValueError('Unexpected actual pool size; require explicit denominator amendment before calls')
    for name,rr in [('dev_windows.jsonl',selected),('train_windows.initial.jsonl',trainrows)]:
        with (WORK/name).open('x') as f:
            for r in rr:f.write(json.dumps(r)+'\n')
    pairs={r['request_id']:selected_edges(r['video_id'],r['request_id'],r['clip_ids']) for r in selected+trainrows}
    write(WORK/'fixed_edge_choices.json',pairs)
    common=read(PROJECT/'reports/common_s0.json');model=Path(common['merged_model'])
    for name,digest in common['merged_model_files'].items():
        if sha(model/name)!=digest:raise ValueError(f'Common S0 changed: {name}')
    asset=read(PROJECT/'manifests/Qwen3-VL-2B-Instruct.asset.json');base=Path('/root/autodl-fs/unsgrpo/models/Qwen3-VL-2B-Instruct')
    for item in asset['files']:
        if sha(base/item['file'])!=item['sha256']:raise ValueError('Frozen original teacher changed')
    prior=read(PROJECT/'reports/freeze.json')
    prior_project={p:h for p,h in prior['files'].items() if p.startswith(str(PROJECT)+'/')}
    for p,h in prior_project.items():
        if sha(p)!=h:raise ValueError('Previous science source changed')
    spec={'status':'FROZEN_BEFORE_GPU_AND_HUMAN_DIAGNOSTICS','time':now(),
        'plan':reference(PROJECT/'reports/NEXT_REWARD_AND_RL_EXPERIMENT_20261029.md'),
        'primary':'V0 stable mapper then round12; all ten sources DEVELOPMENT','public_top5':'NOT_READY; no new alignment attempt',
        'policy':reference(PROJECT/'reports/common_s0.json'),'teacher':reference(PROJECT/'manifests/Qwen3-VL-2B-Instruct.asset.json'),
        'train_sources':train,'development_sources':dev,'dev_windows':60,'dev_candidates_per_window':4,
        'policy_sampling':{'temperature':1,'top_p':1,'top_k':0,'cap':128,'repetition_penalty':1,'seed':31415,
            'order':'manifest dev-ID order, increasing selected-window index, then candidate 0..3; sequential calls'},
        'teacher_question':QUESTION,'edge_permutation':'default_rng(int.from_bytes(sha256(2718|video_id|request_id)[:8],big)); 8 ring +4 opposite',
        'answer_candidates':['A','B','C'],'conditional_p':'p1=P(A|A,B), p2=P(B|A,B) in swapped view; C from ABC normalization',
        'abstention':['C >= max(A,B) in either view','tied A/B or opposite actual winners','w < 0.2'],
        'weight':'(1-t)*(1-abs(p1-p2))*2*abs((p1+p2)/2-0.5)',
        'candidate_legality_gate':{'all_four_legal_groups_fraction_min':.8},
        'information_gate':{'H_range_min':.001,'groups_min':30,'source_coverage_min':8,'mean_O_min':.0005,'oracle_minus_greedy_positive':True},
        'reward_gate':{'pair_H_gap_min':.001,'accuracy_min':.6,'G_positive':True,'G_over_O_min':.2,'selected_minus_greedy_min':0},
        'reward_selection':'R_pair if alone passes; if both pass need +.03 accuracy and G >= R_old; otherwise R_old if passes; else stop',
        'reward_selection_ties':'round(raw reward,12), average H over exact canonical maxima; never use H to break reward ties',
        'training_pool':{'initial_windows':160,'initial_sources':40,'pair_min_edges':3,'pair_min_clips':4,'final_min_windows':80,'final_min_sources':32,
            'offline_candidates_per_window':4,'offline_sampling_seed':31416,'RSFT_tie_seed':1618,'schedule_seed':42,
            'schedule':'320 exposures, source counts differ by <=1; randomized source cycles and replacement window draws; reused for every arm/seed'},
        'training':{'arms':['G_plain','G_guard','RSFT'],'seeds':[42],'conditional_extra_seed':43,'steps':80,'prompts_per_update':4,
            'G':4,'lr':1e-5,'weight_decay':0,'warmup':.05,'cosine':True,'clip_grad':1,'beta':.04,'epsilon':.2,
            'LoRA_rank':16,'LoRA_alpha':32,'LoRA_dropout':0,'visual_and_aligner_frozen':True,
            'reference':'same fixed S0 via disabled fresh language LoRA','loss':'sequence completion mean then fixed batch mean',
            'trainer_backend':'Auditable PyTorch optimizer loop using original Swift/Qwen encode, generate and forward path; actual raw generated tokens, no fabricated EOS',
            'guard_std_floor':.05,'advantage_epsilon':.0001,'guard_pair_u':'mean kept w * distinct covered clips /8','guard_old_u':1,
            'RSFT_loss':'u * sequence mean completion CE; no prompt loss',
            'technical_updates':4,'technical_adapter_reused':False,'checkpoint_selection':'only final80; rolling40 recovery, no development selection'},
        'repeat_gate':{'RL_over_max_S0_RSFT_GCAL_pp':1,'positive_sources_min':6,'repeat_all_three_arms':True},
        'budget':{'screen_GPU_seconds':3600,'total_GPU_seconds':21600,'engineering_seconds':14400,'elapsed_after_first_GPU_seconds':86400,
            'software_fault_repairs_max':1,'new_persistent_bytes':15*1024**3,'new_temporary_bytes':5*1024**3},
        'labels':'Only isolated development evaluation after candidates+teacher+rewards freeze. No train human annotation values.',
        'Git_sync':False,'subagents_used':False,'prior_frozen_project_sources_verified':len(prior_project)}
    write(REPORT/'protocol.json',spec)
    sources=list((PROJECT/'src/unsgrpo').glob('reward_to_rl*.py'))+list((PROJECT/'scripts').glob('reward_to_rl*.py'))+[PROJECT/'tests/test_reward_to_rl.py']
    sources += [PROJECT/'src/unsgrpo'/n for n in ('core.py','data.py','diagnostics.py','evaluation.py','round2_utils.py','round2_metrics.py')]
    sources += [PROJECT/'scripts'/n for n in ('verify_checkpoint.py','training_telemetry.py')]
    sources += [PROJECT/'scripts/reward_to_rl_detached.sh']
    inputs=[PROJECT/'reports/NEXT_REWARD_AND_RL_EXPERIMENT_20261029.md',REPORT/'protocol.json',WORK/'dev_windows.jsonl',WORK/'train_windows.initial.jsonl',WORK/'fixed_edge_choices.json',
        PROJECT/'reports/common_s0.json',PROJECT/'manifests/Qwen3-VL-2B-Instruct.asset.json',ORIGINAL/'predictions.freeze.json',
        ORIGINAL/'inference/S0/windows.jsonl',ORIGINAL/'inference/S0/videos.jsonl',ORIGINAL/'inference/S0/receipt.json',
        Path(plan['paths']['prepared']),Path(plan['paths']['pseudo']),Path(plan['paths']['eval_pool']),R2/'protocol.json']
    for p in sources:
        dst=WORK/'screen_source_snapshot'/p.relative_to(PROJECT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    write(WORK/'screen_protocol.freeze.json',{'status':'FROZEN','time':now(),'sources':{str(p):sha(p) for p in sources},
        'inputs':{str(p):sha(p) for p in inputs},'common_S0_model_files_SHA_checked':True,'teacher_model_files_SHA_checked':True,
        'original_prediction_freeze_identity':reference(ORIGINAL/'predictions.freeze.json'),'previous_closed_round3':reference(PROJECT/'reports/round3_rl_method_v1/CLOSED.json')})
    print('Frozen: 60 development windows, 240 new S0 samples, 12 fixed edges/window; no human label reads.',flush=True)


if __name__=='__main__':main()
