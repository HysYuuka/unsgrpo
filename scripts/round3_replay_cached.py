#!/usr/bin/env python3
"""Reproduce frozen targets and admission decisions without another model call."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from unsgrpo import round3_common as common

ORIGINAL_WORK=common.WORK
ORIGINAL_STORE=common.STORE
ORIGINAL_REPORT=common.REPORT


def verify_original():
    frozen=common.read(ORIGINAL_WORK/'front_end.freeze.json')
    for group in ('sources','inputs'):
        for path,digest in frozen[group].items():
            if common.sha(path)!=digest:raise ValueError(f'Original {group} changed: {path}')
    seal=common.read(ORIGINAL_STORE/'teacher/semantic.freeze.json')
    for key in ('semantic_call_file','semantic_clip_file','reference_targets'):
        r=seal[key]
        if common.sha(r['path'])!=r['sha256']:raise ValueError('Semantic cache changed')


def worker(root,phase,alpha):
    verify_original();common.WORK=root/'work';common.STORE=root/'store';common.REPORT=root/'reports'
    path=common.PROJECT/'scripts'/(phase+'.py')
    spec=importlib.util.spec_from_file_location('round3_cache_replay_phase',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main(alpha)


def main(args):
    verify_original()
    if args.check_only:print('Original source and input hashes verified; no model calls.');return
    root=args.out_root.resolve()
    if root.parent!=ORIGINAL_WORK.parent or not root.name.startswith('round3_') or root==ORIGINAL_WORK:
        raise ValueError('Choose a fresh round3_ directory directly under work/runs')
    if args.phase:worker(root,args.phase,args.alpha);return
    if root.exists():raise FileExistsError('Cache replay never overwrites an existing directory')
    for sub in ('work','store/teacher','reports','logs'):(root/sub).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ORIGINAL_WORK/'teacher_pair_pool.json',root/'work/teacher_pair_pool.json')
    shutil.copy2(ORIGINAL_REPORT/'metric_choice.json',root/'reports/metric_choice.json')
    for name in ('semantic.freeze.json','semantic_calls.jsonl','semantic_clips.jsonl','reference_targets.npz'):
        shutil.copy2(ORIGINAL_STORE/'teacher'/name,root/'store/teacher'/name)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONPATH=str(common.PROJECT/'src')+os.pathsep+str(common.PROJECT/'scripts'))
    runs=[]
    for alpha,tag in ((.75,'alpha075'),(.5,'alpha050')):
        if alpha==.5 and common.read(root/'reports/teacher_gate_alpha075.json')['passed']:break
        for phase in ('round3_materialize_targets','round3_teacher_quality'):
            command=[sys.executable,__file__,'--out-root',str(root),'--phase',phase,'--alpha',str(alpha)]
            with (root/'logs'/(phase+'_'+tag+'.log')).open('x') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            runs.append(command)
        actual=np.load(root/f'store/teacher/targets_{tag}.npz',allow_pickle=False)
        original=np.load(ORIGINAL_STORE/f'teacher/targets_{tag}.npz',allow_pickle=False)
        assert set(actual.files)==set(original.files)
        assert all(np.array_equal(actual[k],original[k]) for k in actual.files)
        assert (root/f'reports/teacher_quality_{tag}.csv').read_bytes()==(ORIGINAL_REPORT/f'teacher_quality_{tag}.csv').read_bytes()
    common.write(root/'receipt.json',{'status':'CACHED_TARGETS_AND_QUALITY_REPLAY_PASSED','time':common.timestamp(),
        'original_scientific_freeze':common.reference(ORIGINAL_WORK/'front_end.freeze.json'),
        'helper':common.reference(__file__),'commands':runs,'q_c_arrays_equal':True,
        'teacher_quality_CSV_bytes_equal':True,'model_calls_added':0,'training_added':0,
        'namespace_rebinding_only':True,'scope':'Cached target/quality reproduction, not a full GPU teacher replay'})
    print(str(root/'receipt.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out-root',type=Path);p.add_argument('--check-only',action='store_true')
    p.add_argument('--phase',choices=['round3_materialize_targets','round3_teacher_quality']);p.add_argument('--alpha',type=float,choices=[.75,.5])
    a=p.parse_args()
    if not a.check_only and a.out_root is None:p.error('--out-root is required')
    main(a)
