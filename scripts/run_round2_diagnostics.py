#!/usr/bin/env python3
"""Freeze and execute the bounded CPU/optional frozen-input diagnostic round."""
import argparse
import fcntl
import importlib.metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unsgrpo.round2_utils import *

PHASES=[('teacher_directions','round2_teachers.py'),('cached_consistency','round2_cached.py'),
        ('frozen_input_probe','round2_input_counterfactual.py'),('small_supervised_fitting','round2_train_students.py'),
        ('small_supervised_evaluation','round2_evaluate_students.py'),('numeric_report','round2_report.py')]


def files_under(root):
    return [p for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts]


def prepare():
    if (ROOT/'round2.freeze.json').exists():raise FileExistsError('Diagnostic protocol is already frozen')
    plan,by,train,dev,feat,pseudo=load_assets()
    if read(ROOT/'processor_preflight.json')['status']!='COMPLETED_ACTUAL_PROCESSOR_AUDIT':raise ValueError('Actual two-image preflight missing')
    testlog=(ROOT/'preflight_tests.log').read_text()
    if '16 passed' not in testlog:raise ValueError('Local diagnostic contract checks did not pass')
    # Preserve and verify first-round science; new diagnostic sources are separate.
    first=read(PROJECT/'reports/freeze.json');checked_original={}
    for name,digest in first['files'].items():
        p=Path(name)
        if p.is_relative_to(PROJECT):
            if sha(p)!=digest:raise ValueError(f'First-round frozen project file changed: {p}')
            checked_original[str(p)]=digest
    inputs=[Path(plan['paths'][k]) for k in ('prepared','pseudo','cluster_model','train_pool','eval_pool')]
    for r in by.values():inputs += [Path(r['features_path']),Path(r['frame_bounds_path'])]
    original=Path(plan['paths']['original_run'])
    inputs += [original/'predictions.freeze.json',Path(plan['paths']['labels'])/'conversion_receipt.json',
               PROJECT/'reports/common_s0.json',PROJECT/'reports/ROUND1_DECISION_AND_FAILURE_REVIEW_20261028.md']
    for method in ('B0','B1','S0','SPLUS','G-CAL','G-RANK'):
        base=original/'inference'/method;inputs += [base/'videos.jsonl',base/'receipt.json']
        if method!='B0':inputs.append(base/'windows.jsonl')
    conversion=read(Path(plan['paths']['labels'])/'conversion_receipt.json')
    for k in dev:
        p=Path(plan['paths']['labels'])/(k+'.npz')
        if sha(p)!=next(r['converted_sha256'] for r in conversion['videos'] if r['video_id']==k):raise ValueError('Changed converted development labels')
        inputs.append(p)
    sources=[PROJECT/'src/unsgrpo'/name for name in ('core.py','data.py','evaluation.py','diagnostics.py','round2_utils.py','round2_metrics.py')]
    sources += [PROJECT/'scripts'/name for name in [s for _,s in PHASES]+['run_round2_diagnostics.py','archive_round2_diagnostics.py','run_round2_detached.sh']]
    sources += [PROJECT/'configs/round2_diagnostic_protocol_v1.json',PROJECT/'tests/test_diagnostics.py',
                PROJECT/'tests/test_round2_diagnostic_contracts.py',ROOT/'protocol.json',ROOT/'processor_preflight.json',ROOT/'preflight_tests.log']
    snapshot=ROOT/'source_snapshot';snapshot.mkdir()
    for p in sources:
        name=p.relative_to(PROJECT) if p.is_relative_to(PROJECT) else Path('run_preflight')/p.name
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    devmanifest=[]
    for k,r in by.items():
        row=dict(r);row['original_split']=row['split'];row['split']='train' if k in train else 'dev';devmanifest.append(row)
    (ROOT/'development_manifest.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in devmanifest))
    freeze={'status':'FROZEN_BEFORE_NEW_DIAGNOSTIC_FITS','created_utc':now(),'sources':{str(p.resolve()):sha(p) for p in sources},
        'inputs':{str(p.resolve()):sha(p) for p in inputs},'original_frozen_sources_verified_unchanged':checked_original,
        'source_snapshot':str(snapshot),'plan':reference(ROOT/'protocol.json'),'development_sources':dev,
        'training_sources':train,'human_annotation_hashing_only_during_prepare':True,'training_human_values_read':False,
        'new_method_result_used_to_choose_parameters':False,'subagents_used':False,
        'environment':{k:importlib.metadata.version(k) for k in ('numpy','scipy','torch','scikit-learn','umap-learn','ms-swift','transformers')},
        'git_commit':None,'version_basis':'Per-file SHA256, original repository has no commit; no Git mutation'}
    write(ROOT/'round2.freeze.json',freeze)
    print('FROZEN: 40 train / 10 DEVELOPMENT sources; 9 teachers, 12 fixed students, <=24 frozen inference calls',flush=True)


def validate(freeze):
    for group in ('sources','inputs'):
        for p,digest in freeze[group].items():
            if sha(p)!=digest:raise ValueError(f'Frozen diagnostic {group} changed: {p}')


def execute():
    freeze=read(ROOT/'round2.freeze.json');validate(freeze);plan=read(ROOT/'protocol.json')
    if (ROOT/'state.json').exists():raise FileExistsError('No automatic retry or hidden reuse; investigate existing state')
    started=time.monotonic();cpu_used=0.;initial_bytes=sum(p.stat().st_size for p in files_under(ROOT))
    state={'status':'RUNNING','started_utc':now(),'pid':os.getpid(),'freeze':reference(ROOT/'round2.freeze.json'),
        'evaluation_status':'DEVELOPMENT_ONLY','phases':{n:{'status':'PLANNED'} for n,_ in PHASES},
        'initial_work_bytes':initial_bytes,'automatic_training_retry':False,'subagents_used':False}
    def save():write(ROOT/'state.json',state,fresh=False)
    env=os.environ.copy();env.update(PYTHONPATH=str(PROJECT/'src')+os.pathsep+str(PROJECT/'scripts'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1')
    (ROOT/'logs').mkdir();save()
    active=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'User or host interrupted diagnostic signal {signum}; no implicit retry')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    with (ROOT/'.driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for name,script in PHASES:
                validate(freeze);phase=state['phases'][name];t=time.monotonic()
                phase.update(status='RUNNING',started_utc=now());state['active_phase']=name;save()
                command=[sys.executable,str(PROJECT/'scripts'/script)];phase['command']=command
                phase['log']=str(ROOT/'logs'/(name+'.log'));phase_env=env.copy()
                phase_env['CUDA_VISIBLE_DEVICES']='0' if name=='frozen_input_probe' else ''
                timeout=(plan['limits']['optional_Qwen_inference_max_seconds']+30 if name=='frozen_input_probe'
                         else max(1,plan['limits']['cpu_compute_max_seconds']-cpu_used))
                gpu_lock=None
                if name=='frozen_input_probe':
                    gpu_lock=(ROOT.parent/'.round1_gpu_driver.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    with open(phase['log'],'x') as log:
                        active=subprocess.Popen(command,cwd=PROJECT,env=phase_env,stdin=subprocess.DEVNULL,
                                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        phase['pid']=active.pid;save();code=active.wait(timeout=timeout)
                    if code!=0:raise RuntimeError(f'{name} exited {code}; see {phase["log"]}')
                finally:
                    if active is not None and active.poll() is None:
                        os.killpg(active.pid,signal.SIGTERM)
                        try:active.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                    active=None
                    if gpu_lock is not None:gpu_lock.close()
                phase.update(status='COMPLETED',ended_utc=now(),wall_seconds=time.monotonic()-t)
                if name!='frozen_input_probe':cpu_used+=phase['wall_seconds']
                save();print(f'{name} COMPLETED {phase["wall_seconds"]:.2f}s',flush=True)
                current=sum(p.stat().st_size for p in files_under(ROOT))
                if current-initial_bytes>plan['limits']['new_storage_max_bytes']:raise RuntimeError('New output byte budget exceeded')
            state.update(status='COMPLETED_NUMERICAL_DIAGNOSTICS',ended_utc=now(),active_phase=None,
                cpu_stage_wall_seconds=cpu_used,total_execution_wall_seconds=time.monotonic()-started,
                final_work_bytes=sum(p.stat().st_size for p in files_under(ROOT)),
                new_work_bytes=sum(p.stat().st_size for p in files_under(ROOT))-initial_bytes,
                report=reference(PROJECT/'reports/round2_diagnostic_v1/RESULTS.md'),
                summary=reference(PROJECT/'reports/round2_diagnostic_v1/summary.json'))
            save()
            write(PROJECT/'reports/round2_diagnostic_v1/EXECUTION_RECEIPT.json',{
                'status':state['status'],'freeze':state['freeze'],'state':reference(ROOT/'state.json'),
                'cpu_stage_wall_seconds':cpu_used,'total_execution_wall_seconds':state['total_execution_wall_seconds'],
                'work_bytes_added':state['new_work_bytes'],'frozen_input_stage_wall_seconds':state['phases']['frozen_input_probe']['wall_seconds'],
                'GRPO_or_Qwen_training':False,'evaluation_status':'DEVELOPMENT_ONLY','subagents_used':False})
        except BaseException as e:
            state.update(status='FAILED_REQUIRES_INVESTIGATION',ended_utc=now(),error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
            if state.get('active_phase'):state['phases'][state['active_phase']]['status']='FAILED'
            save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true');a=p.parse_args()
    prepare() if a.prepare else execute()
