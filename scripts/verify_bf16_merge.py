#!/usr/bin/env python3
"""Verify the actual BF16 common-S0 tensor merge, preserving its measured gap.

The fixed .03/.25 logp thresholds remain diagnostics. They are never relabelled
as passed. Acceptance requires every merged weight to match BF16(base+2BA)
bit-for-bit, every other tensor to remain bit-for-bit unchanged, and a completed
real-GPU fixed-answer comparison with the same inputs. This script runs on CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from diagnose_merge import diagnose,require,sha

ACCEPTANCE_BASIS='verified_bf16_tensor_merge_common_s0'


def logp_gap(snapshot,compared):
    require(snapshot.get('status')=='COMPLETED' and snapshot.get('mode')=='snapshot'
            and snapshot.get('gpu_checkpoint_verification') is True,
            'Require an actual completed GPU snapshot, not source/CPU evidence')
    require(compared.get('mode')=='compare-merged','Require the actual merged-model GPU comparison')
    first=np.asarray(snapshot['fixed_answer']['token_logps'],dtype=np.float64)
    second=np.asarray(compared['fixed_answer']['token_logps'],dtype=np.float64)
    require(first.ndim==1 and len(first)>0 and first.shape==second.shape
            and np.isfinite(first).all() and np.isfinite(second).all(),
            'Fixed-answer probability vectors are missing/non-finite')
    delta=np.abs(first-second)
    expected_pass=bool(delta.mean()<=.03 and delta.max()<=.25)
    numeric=compared['merge_logp_comparison']
    require(numeric['mean_abs_tolerance']==.03 and numeric['max_abs_tolerance']==.25
            and numeric['passed']==expected_pass
            and abs(float(numeric['mean_abs'])-float(delta.mean()))<1e-12
            and abs(float(numeric['max_abs'])-float(delta.max()))<1e-12,
            'Previously recorded gap/initial thresholds do not reproduce numerically')
    if expected_pass:
        require(compared.get('status')=='COMPLETED' and compared.get('gpu_checkpoint_verification') is True,
                'A within-tolerance comparison must have completed its real-GPU check')
    else:
        require(compared.get('status')=='FAILED'
                and compared.get('error','').startswith('RuntimeError: Merge exceeded preregistered BF16 logp tolerance'),
                'Only a recorded initial logp-tolerance failure is a gap diagnostic; other failures block acceptance')
    stages=compared.get('stages',{})
    require({'load_model','fixed_answer_forward','greedy_generation'}<=set(stages)
            and all(stages[key]['wall_seconds']>0 and stages[key]['peak_cuda_allocated_bytes']>0
                    for key in ('load_model','fixed_answer_forward','greedy_generation')),
            'Merged comparison lacks real GPU forward/generation evidence')
    return {'mean_abs':float(delta.mean()),'max_abs':float(delta.max()),
            'initial_mean_abs_tolerance':.03,'initial_max_abs_tolerance':.25,
            'within_initial_tolerance':expected_pass,'original_compare_status':compared['status'],
            'original_failed_report_modified':False,
            'greedy_equal':bool(snapshot['greedy']['response']==compared['greedy']['response'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','adapter','merged','snapshot','compare','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    require(args.threads>=1,'Positive CPU thread count required')
    require(not args.out.exists(),'Use a fresh immutable verification report')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),
            'acceptance_basis':ACCEPTANCE_BASIS,'merge_validation_passed':False,
            'gpu_used':False,'gpu_training_verified':False,'model_files_modified':False,
            'premerge_functional_equivalence_claimed':False,
            'definition':'Common S0 is the actual BF16 quantized merged endpoint; S0 inference and all continuation bases use exactly this endpoint.'}
    started=time.monotonic()
    try:
        snapshot=json.loads(args.snapshot.read_text());compared=json.loads(args.compare.read_text())
        gap=logp_gap(snapshot,compared)
        report['logp_gap']=gap
        report['merge_logp_within_initial_tolerance']=gap['within_initial_tolerance']
        diagnose(args,report)
        require(report['merge_formula_verified_all_tensors'] is True
                and report['aggregate']['different_from_exact_bf16_elements']==0
                and all(item['bitwise_unchanged'] is True for item in report['non_lora_tensors']),
                'Exact tensor merge or non-LoRA preservation failed')
        report.update(status='COMPLETED',merge_validation_passed=True,
                      source_files_sha256={Path(__file__).name:sha(Path(__file__)),
                          'diagnose_merge.py':sha(Path(__file__).with_name('diagnose_merge.py'))})
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,ended_utc=datetime.now(timezone.utc).isoformat())
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'acceptance_basis':ACCEPTANCE_BASIS,
                      'merge_validation_passed':report['merge_validation_passed'],
                      'merge_logp_within_initial_tolerance':report['merge_logp_within_initial_tolerance'],
                      'n_model_tensors':report['n_model_tensors'],'out':str(args.out.resolve())}))


if __name__=='__main__':main()
