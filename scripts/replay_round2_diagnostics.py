#!/usr/bin/env python3
"""Replay unchanged round-two stages into a fresh output namespace.

Only output destinations are rebound. Scientific sources and original inputs
must still match the recorded freeze. Each real stage uses a fresh subprocess.
This helper was added after the original run; it did not produce its results.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unsgrpo import round2_utils as common

ORIGINAL=common.ROOT
PHASES=['round2_teachers','round2_cached','round2_input_counterfactual',
        'round2_train_students','round2_evaluate_students','round2_report']


def verify_original():
    frozen=common.read(ORIGINAL/'round2.freeze.json')
    for group in ('sources','inputs'):
        for path,digest in frozen[group].items():
            if common.sha(path)!=digest:raise ValueError(f'Original {group} changed: {path}')
    return frozen


def bind_stage(root,phase):
    verify_original()
    original_loader=common.load_assets
    common.ROOT=root
    def load_from_replay(output_root=root):
        return original_loader(output_root)
    common.load_assets=load_from_replay
    path=common.PROJECT/'scripts'/(phase+'.py')
    spec=importlib.util.spec_from_file_location('frozen_round2_replay_stage',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if phase=='round2_report':
        # Its only PROJECT use is the report destination, never fitting/evaluation.
        module.PROJECT=root/'report_sink'
    return module


def worker(root,phase):
    module=bind_stage(root,phase)
    module.main()


def main(args):
    frozen=verify_original()
    if args.check_only:
        print('Original scientific sources/inputs match; fresh-root replay is available; no fitting executed.');return
    root=args.run_root.resolve()
    if root.parent!=ORIGINAL.parent or root==ORIGINAL:raise ValueError('Choose a new direct child of the work/runs directory')
    if args.phase:
        if not (root/'replay_origin.json').exists():raise ValueError('Replay namespace not initialized')
        worker(root,args.phase);return
    if root.exists():raise FileExistsError('Replay never overwrites existing outputs')
    root.mkdir();(root/'logs').mkdir();(root/'report_sink/reports/round2_diagnostic_v1').mkdir(parents=True)
    (root/'protocol.json').write_bytes((ORIGINAL/'protocol.json').read_bytes())
    (root/'round2.freeze.json').write_bytes((ORIGINAL/'round2.freeze.json').read_bytes())
    common.write(root/'replay_origin.json',{'status':'UNCHANGED_METHOD_FRESH_NAMESPACE_REGISTERED',
        'created_utc':common.now(),'original_freeze':common.reference(ORIGINAL/'round2.freeze.json'),
        'helper':common.reference(__file__),'new_output_root':str(root),'scientific_source_bytes_changed':False,
        'overrides':'round2_utils.ROOT and load_assets output-root default; report PROJECT only to fresh report_sink. Protocol parameters and asset paths unchanged.'})
    state={'status':'RUNNING','phases':{},'started_utc':common.now()};cpu=0.
    env=os.environ.copy();env.update(PYTHONPATH=str(common.PROJECT/'src')+os.pathsep+str(common.PROJECT/'scripts'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1')
    try:
        for phase in PHASES:
            t=time.monotonic();gpu=phase=='round2_input_counterfactual';env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
            command=[sys.executable,__file__,'--run-root',str(root),'--phase',phase]
            with (root/'logs'/(phase+'.log')).open('x') as log:
                code=subprocess.run(command,env=env,cwd=common.PROJECT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,timeout=630 if gpu else max(1,1800-cpu)).returncode
            if code:raise RuntimeError(f'Phase {phase} failed ({code}); no retries')
            seconds=time.monotonic()-t
            if not gpu:cpu+=seconds
            state['phases'][phase]={'status':'COMPLETED','wall_seconds':seconds,'command':command}
            common.write(root/'replay_state.json',state,fresh=False)
        state.update(status='COMPLETED',ended_utc=common.now(),report=str(root/'report_sink/reports/round2_diagnostic_v1/RESULTS.md'))
        common.write(root/'replay_state.json',state,fresh=False)
        print(state['report'])
    except BaseException as e:
        state.update(status='FAILED_NO_RETRY',error=str(e),traceback=traceback.format_exc());common.write(root/'replay_state.json',state,fresh=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path)
    p.add_argument('--phase',choices=PHASES);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if not args.check_only and args.run_root is None:p.error('--run-root is required for real replay')
    main(args)
