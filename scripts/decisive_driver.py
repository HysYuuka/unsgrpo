#!/usr/bin/env python3
"""Finite detached pipeline, explicit gates, no network/model/config fallback."""
import os
import signal
import subprocess
import time
import traceback
from unsgrpo.decisive_common import *


def child(script,*args,bucket=None):
    storage_check();before=used(bucket) if bucket else 0.;start=time.monotonic()
    cap=min(28800-used(),LIMITS[bucket]-used(bucket))-15 if bucket else 1200
    if cap<=30:raise TimeoutError('RESOURCE_NOT_READY: insufficient phase budget')
    phase='_'.join([script]+list(args)).replace('/','_');log=WORK/'logs'/(phase+'.log')
    command=[str(PYTHON),str(PROJECT/'scripts'/(script+'.py'))]+list(args)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if bucket else ''
    append(WORK/'commands.jsonl',{'time':now(),'command':command,'bucket':bucket,'timeout':cap,'log':str(log)})
    with log.open('x') as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            code=process.wait(timeout=cap)
            if code:raise RuntimeError(f'INCOMPLETE: {phase} exited {code}; see {log}')
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise TimeoutError('RESOURCE_NOT_READY: phase deadline '+phase)
        finally:
            if bucket:
                elapsed=time.monotonic()-start;recorded=used(bucket)-before
                append(WORK/'GPU_ledger.jsonl',{'event':'end','phase':phase+'_parent_overhead','bucket':bucket,'seconds':max(0.,elapsed-recorded),
                    'time':now(),'success':process.returncode==0,'note':'Process startup, loading imports and teardown not already charged by child GPU session'})
    print('Completed '+phase,flush=True)


def throughput():
    dpo=read(STORE/'training/technical_P-DPO/technical_receipt.json');guard=read(STORE/'training/technical_G-guard/technical_receipt.json')
    # Conservative startup + reference/hash audits + checkpoint I/O allowance per process.
    train_est=1.2*(dpo['mean_step_seconds']*320+guard['mean_step_seconds']*160)+600
    old=read(ORIGINAL/'inference/S0/receipt.json')
    seconds=old.get('seconds',old.get('wall_seconds'))
    if seconds is None:
        # Existing first-round inference finished within its declared shared one-hour reserve.
        # Use fixed full evaluation bucket rather than assuming a faster throughput.
        eval_est=3600-used('evaluation')
    else:eval_est=1.2*float(seconds)*4+160
    good=train_est<=LIMITS['training']-used('training') and eval_est<=LIMITS['evaluation']-used('evaluation') and train_est+eval_est<=28800-used()
    write(REPORT/'throughput_reservation.json',{'status':'READY' if good else 'RESOURCE_NOT_READY','time':now(),'training_estimate_seconds':train_est,
        'evaluation_reservation_seconds':eval_est,'remaining_training_seconds':LIMITS['training']-used('training'),'total_remaining_seconds':28800-used(),
        'DPO_mean_actual_step_seconds':dpo['mean_step_seconds'],'GRPO_mean_actual_step_seconds':guard['mean_step_seconds'],
        'rule':'1.2x measured technical throughput plus600s formal startup/I/O; four complete evaluations reserved',
        'reference_receipt':reference(ORIGINAL/'inference/S0/receipt.json')})
    if not good:raise TimeoutError('RESOURCE_NOT_READY: measured throughput cannot reserve all four endpoints and evaluation')


def pipeline():
    child('decisive_teacher','technical',bucket='teacher');r=read(STORE/'teacher/technical_receipt.json')
    if not r['passed']:return 'NO_GO_TEACHER_INTERFACE'
    child('decisive_teacher','dev',bucket='teacher');child('decisive_pool','dev_rewards');child('decisive_evaluate','reward_quality')
    if not read(REPORT/'reward_quality.json')['passed']:return 'NO_GO_REWARD'
    child('decisive_teacher','train',bucket='teacher');child('decisive_pool','training_candidates',bucket='student')
    if (REPORT/'train_signal.json').exists() and read(REPORT/'train_signal.json')['status']!='TRAIN_SIGNAL_READY':return 'NO_GO_TRAIN_SIGNAL'
    child('decisive_pool','freeze_pool')
    if read(REPORT/'train_signal.json')['status']!='TRAIN_SIGNAL_READY':return 'NO_GO_TRAIN_SIGNAL'
    child('decisive_train','--arm','P-DPO','--technical',bucket='student');child('decisive_train','--arm','G-guard','--technical',bucket='student');throughput()
    for arm in ('P-DPO','G-plain','G-guard'):child('decisive_train','--arm',arm,bucket='training')
    child('decisive_train','--arm','P-DPO','--resume-long',bucket='training')
    from decisive_infer import endpoints
    files,paths=endpoints();write(STORE/'endpoints.freeze.json',{'status':'ALL_FOUR_COMPLETE_BEFORE_INFERENCE','time':now(),'files':files,'endpoints':paths})
    for method in ('P-DPO-80','G-plain','G-guard','P-DPO-long'):child('decisive_infer','--method',method,bucket='evaluation')
    child('decisive_infer','--freeze');child('decisive_evaluate','final_endpoints')
    return read(REPORT/'final_evaluation.json')['status']


def main():
    validate();state={'status':'RUNNING','started':now(),'pid':os.getpid(),'network_required_after_start':False};write(WORK/'state.json',state)
    try:status=pipeline();error=None
    except BaseException as exc:
        traceback.print_exc();error=f'{type(exc).__name__}: {exc}'
        status='NO_GO_TRAIN_SIGNAL' if 'NO_GO_TRAIN_SIGNAL' in str(exc) else 'RESOURCE_NOT_READY' if isinstance(exc,TimeoutError) or 'RESOURCE_NOT_READY' in str(exc) or 'out of memory' in str(exc).lower() else 'INCOMPLETE'
    state.update(status=status,ended=now(),error=error,GPU_seconds=used());write(WORK/'state.json',state,fresh=False)
    child('decisive_report');child('decisive_audit');child('decisive_close');print('FINAL '+status,flush=True)


if __name__=='__main__':main()
