#!/usr/bin/env python3
"""Input-only preparation and fixed metric/protocol, before any new model call."""
import json
from pathlib import Path
import shutil
import time
import platform
from importlib.metadata import version
import numpy as np
from unsgrpo.decisive_common import *
from unsgrpo.decisive_core import pairs,QUESTION


def main():
    if (WORK/'protocol.freeze.json').exists():raise FileExistsError('Already frozen')
    start=time.monotonic();cpu_start=time.process_time();old_closed=read(OLD_REPORT/'CLOSED.json');assert sha(old_closed['artifact_index']['path'])==old_closed['artifact_index']['sha256']
    index=read(old_closed['artifact_index']['path'])
    def find(suffix,location):
        found=[r for r in index['files'] if r['location']==location and r['path'].endswith(suffix)]
        if len(found)!=1:raise ValueError(f'Nonunique old artifact: {suffix}')
        item=found[0]
        if sha(item['path'])!=item['sha256']:raise ValueError('Old indexed input changed')
        return item
    candidate_freeze=find('/screen/candidates_and_rewards.freeze.json','persistent')
    frozen=read(candidate_freeze['path'])
    resolved={Path(p).name:p for p in frozen['inputs']}
    for p,h in frozen['inputs'].items():assert sha(p)==h
    inputs={'candidates':resolved['candidates.jsonl'],'old_rewards':resolved['candidate_rewards.jsonl'],
        'old_candidate_rewards_freeze':candidate_freeze['path'],
        'old_full_video_H':find('/screen/single_window_full_video_AP.jsonl','persistent')['path']}
    for filename,n in [('dev_windows.jsonl',60),('train_windows.initial.jsonl',160)]:
        item=find('/'+filename,'working');rr=records(item['path']);assert len(rr)==n
        shutil.copy2(item['path'],WORK/filename);inputs[filename]=str(WORK/filename)
    candidates=records(inputs['candidates']);assert len(candidates)==240
    assert sum(all(r['legal'] for r in candidates[i:i+4]) for i in range(0,240,4))==59
    write(WORK/'reused_inputs.json',inputs)
    plan,by,train,dev,features,pseudo=assets();devrows=records(WORK/'dev_windows.jsonl');trainrows=records(WORK/'train_windows.initial.jsonl')
    write(WORK/'pair_choices.json',{r['request_id']:pairs(r['video_id'],r['request_id'],r['clip_ids']) for r in devrows+trainrows})
    rng=np.random.default_rng(2718);chosen=rng.choice(sorted(train),8,replace=False).tolist();tech=[]
    for k in chosen:
        choices=[r for r in trainrows if r['video_id']==k];row=choices[int(rng.integers(len(choices)))];ids=row['clip_ids'];i=int(rng.choice(ids));a,b=map(int,rng.choice(ids,2,replace=False))
        tech += [{'video_id':k,'clip_i':i,'clip_j':i,'same_clip':True},{'video_id':k,'clip_i':a,'clip_j':b,'same_clip':False}]
    write(WORK/'technical_pairs.json',{'seed':2718,'records':tech})
    # Reuse only the existing metric evidence. No UMT clip-axis asset has appeared.
    metric_start=time.monotonic();old_metric=read(PROJECT/'reports/round3_rl_method_v1/metric_choice.json')
    for row in old_metric['native_time_axis_acceptance']:
        for field in ('native_clock','converted_label'):
            assert sha(row[field]['path'])==row[field]['sha256']
    metric={'status':'V0_PRIMARY_FROZEN_PUBLIC_TOP5_NOT_READY','time':now(),
        'primary':'Stable V0 / overlap integral / round12 exact ties','public_top5':'NOT_READY',
        'existing_UMT_implementation_evidence':old_metric['implementation_acceptance'],
        'reason':'Existing assets still lack audited UMT annotation/feature clip-time correspondence; no new download or metric invention.',
        'engineering_wall_seconds':time.monotonic()-metric_start,'new_human_annotation_values_read':False}
    write(REPORT/'metric_alignment.json',metric)
    common=read(PROJECT/'reports/common_s0.json')
    for name,h in common['merged_model_files'].items():assert sha(Path(common['merged_model'])/name)==h
    asset=read(STORE/'teacher_model/asset_receipt.json')
    if asset['status']!='COMPLETED_VERIFIED_OFFICIAL_BF16_ASSET':raise ValueError('RESOURCE_NOT_READY: incomplete teacher')
    for r in asset['files']:
        p=STORE/'teacher_model'/r['file']
        if p.stat().st_size!=r['bytes']:raise ValueError('Teacher asset size changed')
    if '7 passed' not in (WORK/'logs/contracts.log').read_text():raise ValueError('New contracts not passed')
    protocol={'status':'FROZEN_BEFORE_FIRST_NEW_MODEL_CALL','time':now(),'plan':reference(PROJECT/'reports/NEXT_DECISIVE_RL_EXPERIMENT_20261029.md'),
        'namespace':'decisive_rl_v2','model_teacher':reference(STORE/'teacher_model/asset_receipt.json'),
        'common_S0':reference(PROJECT/'reports/common_s0.json'),'reused_inputs':inputs,'seed':42,
        'input':'Original 16-cell overview, 8 clips x4 frames; policy JSON 8 integers 0..100; original mean aggregation, illegal windows zero included',
        'teacher_question':QUESTION,'teacher_candidates':['A','B','C','D'],'teacher_dtype':'BF16, batch1, no quantization, last-position logits',
        'technical':{'seed':2718,'sources':chosen,'nominal_views':32,'same_clip_C_in_both_min':6,'raw_mass_at_least_half_fraction_min':.9,
            'top_rule':'unique argmax; C tied with D is insufficient for correct tie','view_cache':'exact prompt, overview/frame bytes, teacher revision and processor identity'},
        'edge_rules':{'permutation_seed':2718,'pairs_per_window':4,'nonoverlapping':True,'mass_min':.5,'margin_min':.1,
            'same_actual_direction_both_views':True,'weight':'(1-TV(p,q_aligned))*mean_probability_margin',
            'tie':'both unique C; archive only','unknown':'D, order conflict, low mass/margin or ambiguity; never a negative label'},
        'reward':'Kept-edge weighted order agreement 1/.5/0, invalid JSON -1, legal/no edge .5',
        'development_gate':{'sufficient_windows_min':36,'sources_with_at_least_two_sufficient_windows_min':8,
            'directional_edges_per_window_min':2,'distinct_clips_min':4,'G_new_pp_min':.15,'G_new_minus_old_pp_min':.03,
            'selected_minus_greedy_pp_min':0,'sources_positive_G_min':6,'accuracy_60pct_gate':False},
        'train_pool':{'nominal_windows':160,'min_windows':80,'min_sources':32,'G':4,'sampling_seed':31416,
            'preference_reward_gap_min':.1,'tie_seed':1618,'no_resampling_when_gap_insufficient':True,
            'u':'mean directional weight * directional clip coverage /8','schedule_seed':42,
            'exposures_main':320,'exposures_DPO_long':1280,'source_balancing':'randomized source cycles; window draws with replacement; common 320-prefix'},
        'optimization':{'seed':42,'main_steps':80,'DPO_long_steps':320,'prompts_per_step':4,'lr':1e-5,'warmup_steps':4,'actual_step1_to4_learning_rates':[2.5e-6,5e-6,7.5e-6,1e-5],'after_warmup':'constant',
            'weight_decay':0,'clip_grad_norm':1,'LoRA':{'rank':16,'alpha':32,'dropout':0,'trainable':'language linear layers only'},
            'DPO_beta':.1,'DPO_logp':'SUM over all actual generated completion tokens including actual EOS; no prompt',
            'GRPO_G':4,'std':'sample','epsilon_adv':.0001,'std_floor_guard':.05,'beta_KL':.04,'PPO_clip':.2,
            'generation':{'temperature':1,'top_p':1,'top_k':0,'repetition_penalty':1,'cap':128},
            'reference':'Frozen original common S0 by disabling all fresh LoRA','all_zero_u_batch':'skip optimizer and scheduler; never change momentum',
            'technical_checks':'Separate discarded four-step DPO and four-step guard adapters; eight optimizer calls total, charged to student technical budget',
            'endpoints':['P-DPO-80','P-DPO-long-320','G-plain-80','G-guard-80'],'recovery_only':[40],
            'DPO_continuation':'restore adapter/optimizer/scheduler/RNG after80, continue81..320; no endpoint AP read until all four frozen'},
        'decision':{'GRPO_minus_S0_pp_min':1,'GRPO_minus_stronger_whole_macro_DPO_pp_min':.5,'not_below_GCAL':True,
            'positive_sources_vs_DPO_min':6,'each_leave_one_source_difference_positive':True,'bootstrap_seed':42,'bootstrap_n':10000,
            'teacher_only_signal_rule':'Both RL fail; stronger DPO beats S0 by at least1pp and at least6/10 sources improve; project label, not significance'},
        'budgets':{'GPU_total_seconds':28800,'GPU_buckets':LIMITS,'new_store_bytes':40*1024**3,'new_work_bytes':5*1024**3,
            'local_free_min':10*1024**3,'file_free_min':20*1024**3,'engineering_repairs_max':1,
            'engineering_preparation_elapsed_limit':5400,'metric_CPU_limit':3600,'evaluation_reserve_GPU_seconds':3600},
        'human_labels':'No training human values; teacher/candidate rewards frozen before isolated dev evaluator reads cached H/ratings',
        'known_89_conflicts':'final fixed-case review only; never used for inputs, question or admission','Git_sync':False,'subagents_used':False}
    protocol['failure_case_reporting_rules']={'interface':'First3 failed same-clip pairs sorted by source/clip',
        'reward':'Lowest single-window G, max1/source, ties requestID, first3',
        'final':'Worst3 source AP differences versus stronger whole-macro DPO for each RL; report positive values if fewer than3 failures'}
    protocol['runtime_versions']={'python':platform.python_version(),**{k:version(k) for k in ('torch','transformers','ms-swift','peft','numpy','scikit-learn')}}
    write(REPORT/'protocol.actual.json',protocol)
    sources=list((PROJECT/'src/unsgrpo').glob('decisive_*.py'))+list((PROJECT/'scripts').glob('decisive_*'))+[PROJECT/'tests/test_decisive_rl.py']
    sources += [PROJECT/'src/unsgrpo'/name for name in ('core.py','data.py','diagnostics.py','evaluation.py','round2_utils.py','round2_metrics.py','reward_to_rl.py','reward_to_rl_model.py','reward_to_rl_common.py')]
    sources += [PROJECT/'scripts'/name for name in ('verify_checkpoint.py','training_telemetry.py','reward_to_rl_train.py','reward_to_rl_evaluate.py','ranged_download.py')]
    paths=[PROJECT/'reports/NEXT_DECISIVE_RL_EXPERIMENT_20261029.md',PROJECT/'reports/common_s0.json',PROJECT/'reports/input_audit.json',REPORT/'protocol.actual.json',REPORT/'metric_alignment.json',
        WORK/'reused_inputs.json',WORK/'pair_choices.json',WORK/'technical_pairs.json',WORK/'dev_windows.jsonl',WORK/'train_windows.initial.jsonl',WORK/'logs/contracts.log',WORK/'local_preflight.json',
        STORE/'teacher_model/asset_receipt.json',R2/'protocol.json',PROJECT/'reports/round3_rl_method_v1/historical_baselines.json',
        PROJECT/'reports/round3_rl_method_v1/metric_choice.json',Path(plan['paths']['labels'])/'conversion_receipt.json',
        Path(plan['paths']['prepared']),Path(plan['paths']['pseudo']),Path(plan['paths']['train_pool']),Path(plan['paths']['eval_pool'])]+[Path(v) for v in inputs.values()]
    for p in sources:
        if not p.is_file():continue
        dest=WORK/'source_snapshot'/p.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(read(WORK/'initial_state.json')['started_utc'])).total_seconds()
    if elapsed>5400:raise RuntimeError('RESOURCE_NOT_READY: preparation exceeded90min elapsed bound')
    write(WORK/'protocol.freeze.json',{'status':'FROZEN','time':now(),'sources':{str(p):sha(p) for p in sources if p.is_file()},
        'inputs':{str(p):sha(p) for p in paths},'old_closed_job':old_closed,'preparation_elapsed_seconds':elapsed,
        'prepare_wall_seconds':time.monotonic()-start,'prepare_CPU_process_seconds':time.process_time()-cpu_start,
        'engineering_active_seconds_conservative_upper_bound':elapsed,
        'engineering_bound_note':'Entire preparation elapsed, including parallel asset acquisition, is a conservative upper bound; download and engineering overlap and must not be summed.',
        'no_new_human_annotation_values_read':True})
    print('Frozen official 8B + identical student/input + DPO/GRPO limits before first model call',flush=True)


if __name__=='__main__':main()
