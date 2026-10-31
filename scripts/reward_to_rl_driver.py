#!/usr/bin/env python3
"""Finite offline controller: admission -> same-job training -> all endpoints."""
from datetime import datetime,timezone
import os
import signal
import subprocess
import sys
import time
import traceback
from unsgrpo.reward_to_rl_common import *


def main():
    validate_freeze()
    if (WORK/'state.json').exists():raise FileExistsError('No automatic rerun of a completed/failed namespace')
    state={'status':'RUNNING','time':now(),'phases':{},'active_phase':None,'seed43_started':False,
        'automatic_scientific_retries':0,'software_repairs_used':int((WORK/'software_fix1.json').exists()),'GPU_seconds_max':21600,
        'continued_after_documented_software_fix':reference(WORK/'software_fix1.json') if (WORK/'software_fix1.json').exists() else None}
    env=os.environ.copy();env.update(PYTHONPATH=str(PROJECT/'src')+os.pathsep+str(PROJECT/'scripts'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1')
    def save():write(WORK/'state.json',state,fresh=False)
    def run(name,script,args=(),gpu=False,allow_failure=False):
        check_storage();first=WORK/'gpu_sessions.jsonl'
        if first.exists():
            starts=[r for r in records(first) if r['event']=='start']
            if starts and (datetime.now(timezone.utc)-datetime.fromisoformat(starts[0]['time']['utc'])).total_seconds()>86400:raise TimeoutError('BUDGET_LIMITED: 24h decision deadline')
        state['active_phase']=name;state['phases'][name]={'status':'RUNNING','time':now()};save()
        command=[str(PYTHON),str(PROJECT/'scripts'/script),*args];child=env.copy();child['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
        before=gpu_used();t=time.monotonic()
        with (WORK/'logs'/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(command,env=child,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['phases'][name].update(command=command,pid=proc.pid);save()
            try:code=proc.wait(timeout=max(30,21600-before+15) if gpu else 3600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        elapsed=time.monotonic()-t
        if gpu:
            # Conservative occupancy includes interpreter/model teardown and errors
            # before session entry; avoids undercounting a worker's GPU lifetime.
            extra=max(0.,elapsed-(gpu_used()-before))
            append(WORK/'gpu_sessions.jsonl',{'event':'end','phase':name+'_process_overhead_upper_bound','session':name+'_parent',
                'seconds':extra,'success':code==0,'time':now(),'accounting':'parent wall minus measured child session; conservative'})
        state['phases'][name].update(status='COMPLETED' if code==0 else 'FAILED',ended=now(),wall_seconds=elapsed,exit_code=code);save()
        print(name+' '+state['phases'][name]['status']+f'; cumulative conservative GPU {gpu_used():.1f}s',flush=True)
        if code and not allow_failure:raise RuntimeError(f'{name} failed ({code}); inspect exact log, no automatic parameter/restart change')
        return code
    def finish(status):
        state.update(status=status,active_phase=None,ended=now(),GPU_seconds=gpu_used());save()
        run('report','reward_to_rl_report.py');state['active_phase']=None;save()
    try:
        run('interface','reward_to_rl_screen.py',('interface',),gpu=True)
        run('dev_candidates','reward_to_rl_screen.py',('candidates','--scope','dev'),gpu=True)
        if read(STORE/'screen/candidates_receipt.json')['all_four_legal_fraction']<.8:
            finish('POLICY_OUTPUT_NOT_READY');return
        run('dev_teacher','reward_to_rl_screen.py',('teacher','--scope','dev'),gpu=True)
        run('dev_reward_freeze','reward_to_rl_screen.py',('freeze_rewards',))
        run('screen_evaluation','reward_to_rl_evaluate.py',('screen',))
        choice=read(REPORT/'screen_result.json')
        if choice['status']!='REWARD_SELECTED_FOR_TRAINING':finish(choice['status']);return
        force_old=False
        if choice['selected_reward']=='R_pair':
            # Incomplete teacher cache is never used for training. Only the
            # predeclared admitted-old-reward fallback may follow this failure.
            code=run('train_teacher','reward_to_rl_screen.py',('teacher','--scope','train'),gpu=True,allow_failure=True)
            if code:
                text=(WORK/'logs/train_teacher.log').read_text()
                if 'BUDGET_LIMITED' not in text and 'TimeoutError' not in text:raise RuntimeError('Teacher cache software failure; not eligible for scientific old-reward fallback')
                if choice['primary_all_four_legal']['R_old']['base_admission']:force_old=True
                else:finish('BUDGET_LIMITED');return
        run('eligible_training_pool','reward_to_rl_pool.py',('eligible',*(['--force-old'] if force_old else [])))
        if (REPORT/'train_pool_status.json').exists() and read(REPORT/'train_pool_status.json')['status']=='TRAIN_SIGNAL_TOO_SPARSE':finish('TRAIN_SIGNAL_TOO_SPARSE');return
        run('train_offline_candidates','reward_to_rl_screen.py',('candidates','--scope','train'),gpu=True)
        run('finalize_training_pool','reward_to_rl_pool.py',('finalize',))
        if read(REPORT/'train_pool_status.json')['status']!='TRAIN_POOL_READY':finish('TRAIN_SIGNAL_TOO_SPARSE');return
        run('technical4','reward_to_rl_train.py',('--arm','G_guard','--seed','42','--technical'),gpu=True)
        tech=read(STORE/'training/technical4/receipt.json')
        predicted=tech['mean_step_seconds']*80*2.1+3*read(ORIGINAL/'inference/S0/receipt.json')['wall_seconds']
        write(WORK/'formal_budget_estimate.json',{'time':now(),'GPU_seconds_used':gpu_used(),'remaining':21600-gpu_used(),
            'seed42_train_and_infer_estimate_seconds':predicted,'basis':'actual four-step guarded timing, factor2.1 for two GRPO and cheaper RSFT plus three historical full S0 inference durations'})
        if predicted>21600-gpu_used()-300:finish('BUDGET_LIMITED_BEFORE_COMPLETE_MATRIX');return
        for seed in (42,43):
            if seed==43:
                previous=read(REPORT/'endpoints_seed42.json')
                if not previous['repeat_gate_passed']:break
                phases=state['phases'];cost=sum(v['wall_seconds'] for k,v in phases.items() if k.startswith('train42_') or k.startswith('infer42_'))*1.1
                if cost>21600-gpu_used()-300:
                    state['repeat_not_started_reason']='Insufficient remaining measured budget for all three arms and full evaluation';finish('BUDGET_LIMITED_REPEAT_NOT_STARTED');return
                state['seed43_started']=True;save()
            for arm in ('G_plain','G_guard','RSFT'):
                run(f'train{seed}_{arm}','reward_to_rl_train.py',('--arm',arm,'--seed',str(seed)),gpu=True)
            for arm in ('G_plain','G_guard','RSFT'):
                run(f'infer{seed}_{arm}','reward_to_rl_infer.py',('--arm',arm,'--seed',str(seed)),gpu=True)
            run(f'freeze_predictions{seed}','reward_to_rl_infer.py',('--seed',str(seed),'--freeze'))
            run(f'evaluate_endpoints{seed}','reward_to_rl_evaluate.py',('endpoints','--seed',str(seed)))
        finish('COMPLETED_PRESPECIFIED_EXPERIMENT')
    except BaseException as exc:
        state.update(status='BUDGET_LIMITED' if isinstance(exc,TimeoutError) or gpu_used()>=21000 else 'IMPLEMENTATION_OR_RUNTIME_BLOCKED',
            error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),ended=now(),GPU_seconds=gpu_used());save()
        try:run('failure_report','reward_to_rl_report.py')
        except BaseException:pass
        raise


if __name__=='__main__':main()
