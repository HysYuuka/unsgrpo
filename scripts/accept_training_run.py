#!/usr/bin/env python3
"""Accept a completed formal arm from actual runtime ledgers and saved weights.

A successful trainer exit is insufficient. This script recomputes every update,
exposure, rollout and token count, validates the frozen protocol and input audit,
checks common-S0 ancestry, and reconstructs the final LoRA parameter hash from
checkpoint safetensors before promoting the run receipt. Smoke runs never pass.
The immutable trainer-exit receipt is preserved. No exact resume is claimed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from freeze_round1 import (command_flags, one, read_json, read_rows, require,
                           sha, validate_run)

PROJECT = Path(__file__).resolve().parents[1]
METHODS = {
    'S0': {'stage':'sft0', 'mode':'sft', 'updates':256, 'prompts':16, 'rollouts':0},
    'S+': {'stage':'sft-continue', 'mode':'sft', 'updates':256, 'prompts':4, 'rollouts':0},
    'G-CAL': {'stage':'grpo', 'mode':'grpo', 'updates':256, 'prompts':4, 'rollouts':4096, 'rank_weight':0.},
    'G-RANK': {'stage':'grpo', 'mode':'grpo', 'updates':256, 'prompts':4, 'rollouts':4096, 'rank_weight':.8},
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def expected_method(receipt, method):
    require(method in METHODS, 'Unknown formal training method')
    spec = METHODS[method]
    require(receipt.get('mode') == 'formal', 'Technical/smoke runs cannot be promoted to formal experiments')
    require(receipt.get('stage') == spec['stage'] and receipt.get('method_id') == method,
            'Requested method differs from the actual recorded formal command')
    require(receipt.get('status') == 'TRAINER_EXITED_SUCCESS_REQUIRES_AUDIT' and receipt.get('exit_code') == 0,
            'Require the original successful trainer exit, before acceptance promotion')
    require(receipt.get('gpu_training_verified') is False,
            'Run must not already claim GPU acceptance before artifact recomputation')
    return spec


def checked_hash_table(root, files, evidence_files):
    root = Path(root).resolve()
    require(root.is_dir() and files, 'Missing registered model hash table')
    for name, info in files.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Model hash path escapes root or is missing')
        expected = info.get('sha256') if isinstance(info, dict) else info
        require(isinstance(expected, str) and sha(path) == expected, f'Registered model file changed: {path}')
        if isinstance(info, dict) and 'bytes' in info:
            require(path.stat().st_size == info['bytes'], 'Registered model file size changed')
        evidence_files.add(path)


def audit_consumption(run, audit, mode, evidence_files):
    rows_path = Path(audit['rows_path']).resolve()
    require(sha(rows_path) == audit['rows_sha256'], 'Input audit row ledger changed')
    require(run['begin']['audit_rows_sha256'] == audit['rows_sha256'],
            'Trainer used a different full-input audit row ledger')
    rows = read_rows(rows_path)
    require(len(rows) == audit['n_rows_encoded'], 'Input audit row count differs')
    by_id = {}
    for row in rows:
        if row['task'] == mode:
            by_id.setdefault(row['dataset_request_id'], row)
    require(by_id, 'No audited rows exist for this training objective')
    for consumed in run['consumed_rows']:
        rid = consumed['dataset_request_id']
        require(rid in by_id, 'Consumed window is absent from the frozen full-input audit')
        registered = by_id[rid]
        require(consumed['clip_ids'] == registered['clip_ids']
                and consumed['prompt_tokens'] == registered['prompt_tokens'],
                'Consumed clip order or prompt length differs from real input audit')
        require(consumed['media_tensor_sha256'] in
                {registered['media_tensor_sha256'], registered.get('media_tensor_bf16_sha256')},
                'Actual consumed media differs from audited processor tensors')
        if mode == 'sft':
            require(consumed.get('assistant_only_labels_verified') is True
                    and consumed['input_ids_sha256'] == registered['input_ids_sha256']
                    and consumed['encoded_tokens'] == registered['encoded_tokens']
                    and consumed['loss_tokens'] == registered['supervised_tokens'],
                    'SFT consumed tokens/assistant-only labels differ from the full-input audit')
    evidence_files.add(rows_path)
    return {'rows_path':str(rows_path), 'rows_sha256':audit['rows_sha256'],
            'n_consumed_rows_bound_to_audit':len(run['consumed_rows'])}


def final_endpoint(checkpoint, output_dir, evidence_files):
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    require(checkpoint.is_relative_to(output_dir) and checkpoint.name == 'checkpoint-256',
            'Formal endpoint must be the actual final checkpoint-256 under this run output')
    require((checkpoint/'adapter_model.safetensors').is_file()
            and (checkpoint/'adapter_config.json').is_file(), 'Final endpoint lacks real LoRA weights/config')
    state = read_json(checkpoint/'trainer_state.json')
    require(state['global_step'] == 256, 'Final trainer state is not exactly update 256')
    config = read_json(checkpoint/'adapter_config.json')
    require(config.get('r') == 16 and config.get('lora_alpha') == 32 and config.get('lora_dropout') == 0,
            'Saved endpoint LoRA specification differs from the frozen rank/alpha/dropout')
    files = {}
    for path in sorted(checkpoint.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(checkpoint))] = sha(path)
            evidence_files.add(path.resolve())
    require(files.get('adapter_model.safetensors'), 'No saved endpoint weight hash')
    return {'path':str(checkpoint), 'files':files}


def accept(run_dir, method, freeze_path, out_path, common_path=None, *, allow_formal_zero_signal=False):
    run_dir, freeze_path, out_path = map(lambda p:Path(p).resolve(), (run_dir, freeze_path, out_path))
    require(not out_path.exists(), 'Use a fresh acceptance receipt path; preserve failed evidence')
    report = {'status':'RUNNING', 'method_id':method,
              'started_utc':datetime.now(timezone.utc).isoformat(),
              'gpu_training_verified':False, 'human_highlight_labels_read':False,
              'exact_resume_verified':False, 'run_dir':str(run_dir)}
    started = time.monotonic()
    atomic_json(out_path, report)
    receipt_path = run_dir/'receipt.json'
    evidence_files = {freeze_path, Path(__file__).resolve()}
    try:
        original_receipt_sha = sha(receipt_path)
        receipt = read_json(receipt_path)
        spec = expected_method(receipt, method)
        require(not allow_formal_zero_signal or spec['mode']=='grpo',
                'Zero-signal negative-outcome acceptance is restricted to explicitly opted-in formal GRPO')
        require(Path(receipt.get('freeze_path','')).resolve() == freeze_path
                and receipt.get('freeze_sha256') == sha(freeze_path),
                'Current protocol is not the frozen protocol recorded before this actual training run')
        freeze = read_json(freeze_path)
        require(freeze.get('status') == 'FROZEN' and freeze.get('human_annotations_read_for_training') is False,
                'Protocol is not frozen with the declared no-human-label boundary')
        frozen_files = {}
        for name, digest in freeze['files'].items():
            path = Path(name)
            if not path.is_absolute(): path = PROJECT/path
            path = path.resolve()
            require(path.is_file() and sha(path) == digest, f'Frozen evidence file changed: {path}')
            frozen_files[str(path)] = digest
        require(str(Path(__file__).resolve()) in frozen_files,
                'Formal acceptance implementation was not included in the original protocol freeze')
        technical_path = Path(freeze['technical_acceptance']).resolve()
        technical = read_json(technical_path)
        require(technical.get('status') == 'COMPLETED' and technical.get('gpu_training_verified') is True
                and technical.get('checks') and all(value is True for value in technical['checks'].values()),
                'The original real-GPU technical acceptance is incomplete')
        evidence_files.add(technical_path)
        frozen_arm = freeze['matrix'][method]
        require((frozen_arm['updates'], frozen_arm['prompt_exposures_per_update'], frozen_arm['group_size'], frozen_arm['seed'])
                == (spec['updates'],spec['prompts'],4 if spec['mode']=='grpo' else 1,42),
                'Frozen matrix differs from the original first-round budget')
        flags = command_flags(receipt['command'])
        require(int(one(flags,'--seed')) == int(one(flags,'--data_seed')) == 42, 'Actual training seeds differ from freeze')
        protocol = freeze['protocol']
        expected_lr = protocol['optimization']['sft0_lr' if method=='S0' else 'continuation_lr']
        require(float(one(flags,'--learning_rate')) == expected_lr
                and float(one(flags,'--weight_decay')) == protocol['optimization']['weight_decay']
                and float(one(flags,'--max_grad_norm')) == protocol['optimization']['max_grad_norm'],
                'Actual optimizer settings differ from the frozen method')
        audit_path = Path(receipt.get('input_audit_path','')).resolve()
        require(str(audit_path) in frozen_files and receipt.get('input_audit_sha256') == sha(audit_path),
                'Training input audit differs from its run-start/frozen identity')
        audit = read_json(audit_path)
        require(audit.get('status') == 'COMPLETED' and audit.get('all_rows_encoded') is True and not audit.get('errors'),
                'Actual full-input processor/template audit is incomplete')
        for key, value in audit['visual_environment'].items():
            require(value is not None and receipt['env'].get(key) == value, 'Training visual budget differs from real input audit')
        evidence_files.add(audit_path)
        dataset = Path(one(flags,'--dataset')).resolve()
        require(receipt.get('training_dataset_path') == str(dataset)
                and receipt.get('training_dataset_sha256') == sha(dataset)
                and audit['input_sha256'].get(str(dataset)) == sha(dataset)
                and frozen_files.get(str(dataset)) == sha(dataset),
                'Actual training dataset changed or is outside the frozen audited pool')
        run = validate_run(run_dir, spec['mode'], spec['updates'], spec['prompts'], evidence_files,
                           allow_formal_zero_signal=allow_formal_zero_signal)
        counts = run['counts_recomputed']
        require(counts['successful_optimizer_updates'] == 256
                and counts['prompt_exposures_in_successful_updates'] == 256*spec['prompts'],
                'Recomputed update/exposure budget differs from the formal method')
        report['input_audit_binding'] = audit_consumption(run, audit, spec['mode'], evidence_files)
        common_hash = None
        if method == 'S0':
            require(Path(run['model_path']).resolve() == Path(audit['model_path']).resolve(),
                    'Formal S0 did not start from the audited original Qwen base')
        else:
            common_path = Path(common_path or PROJECT/'reports/common_s0.json').resolve()
            common_hash = sha(common_path)
            require(receipt.get('parent_common_s0_sha256') == common_hash
                    and Path(receipt.get('parent_common_s0_path','')).resolve() == common_path,
                    'Common S0 differs from the parent recorded at training start')
            common = read_json(common_path)
            require(common.get('status') == 'COMPLETED' and common.get('merge_validation_passed') is True
                    and common.get('acceptance_basis')=='verified_bf16_tensor_merge_common_s0'
                    and Path(common['merged_model']).resolve() == Path(run['model_path']).resolve(),
                    'Continuation did not use the registered verified merged common S0')
            checked_hash_table(common['merged_model'], common['merged_model_files'], evidence_files)
            evidence_files.add(common_path)
            report['common_s0_sha256'] = common_hash
        if spec['mode'] == 'grpo':
            require(counts.get('rollout_completions_generated') == 4096
                    and counts.get('rollout_completions_consumed_in_successful_updates') == 4096,
                    'GRPO actual generated/consumed rollout budget is not exactly 4096')
            expected_weight = spec['rank_weight']
            require(float(receipt['env']['HL_RANK_WEIGHT']) == frozen_arm['rank_weight'] == expected_weight,
                    'Actual GRPO reward branch differs from frozen method')
            require(one(flags,'--advantage_estimator') == 'grpo' and one(flags,'--kl_in_reward') == 'false'
                    and float(one(flags,'--beta')) == protocol['grpo']['beta']
                    and float(one(flags,'--epsilon')) == protocol['grpo']['epsilon']
                    and one(flags,'--remove_unused_columns') == 'false',
                    'Actual GRPO advantage/KL/reward metadata settings differ from frozen method')
            reward_paths = list(run_dir.glob('rollouts.rank*.jsonl'))
            require(len(reward_paths) == 1, 'Missing/duplicate single-GPU reward ledger')
            rewards = read_rows(reward_paths[0])
            require(len(rewards) == 4096 and all(float(row['rank_weight']) == expected_weight for row in rewards),
                    'Actual reward rows or branch weights differ from frozen GRPO method')
        endpoint = final_endpoint(run['checkpoint'], receipt['output_dir'], evidence_files)
        negative=run.get('negative_outcome')
        changed_updates=run['parameter_changed_updates']
        if negative is None:
            require(run['begin']['initial_trainable_sha256'] != run['end']['final_trainable_sha256']
                    and run['end'].get('trainable_changed_since_begin') is True,
                    'Final LoRA did not change from its fresh initial adapter')
        else:
            require(allow_formal_zero_signal and spec['mode']=='grpo'
                    and negative=='ALL_GROUP_ADVANTAGES_ZERO_NO_POLICY_UPDATE'
                    and changed_updates==0
                    and run['begin']['initial_trainable_sha256']==run['end']['final_trainable_sha256']
                    and run['end'].get('trainable_changed_since_begin') is False,
                    'Unchanged parameters lack strictly verified, opted-in formal GRPO zero-signal evidence')
        report.update(negative_outcome=negative,parameter_changed_updates=changed_updates,
                      allow_formal_zero_signal=allow_formal_zero_signal)
        require(sha(receipt_path) == original_receipt_sha, 'Run receipt changed during artifact validation')
        # Preserve the exact original trainer outcome as an immutable separate file.
        preserved = run_dir/'trainer_exit_receipt.json'
        if preserved.exists():
            require(sha(preserved)==original_receipt_sha, 'Previously preserved trainer-exit receipt differs; never overwrite it')
        else:
            with preserved.open('xb') as stream:
                stream.write(receipt_path.read_bytes())
        evidence_files.discard(receipt_path)
        evidence_files.add(preserved)
        evidence_hashes = {str(path):sha(path) for path in sorted(evidence_files)}
        report.update(status='COMPLETED', gpu_training_verified=True,
                      freeze_path=str(freeze_path), freeze_sha256=sha(freeze_path),
                      trainer_exit_receipt_path=str(preserved), trainer_exit_receipt_sha256=sha(preserved),
                      counts_recomputed=counts, endpoint=endpoint,
                      evidence_files_sha256=evidence_hashes,
                      frozen_files_revalidated=len(frozen_files),
                      validator_source_sha256=sha(Path(__file__)),
                      runtime_versions=run['versions'], stage_costs=run['stage_costs'],
                      n_successful_optimizer_updates=256,
                      n_prompt_exposures=256*spec['prompts'], n_rollout_completions=spec['rollouts'])
        report['wall_seconds'] = time.monotonic()-started
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(out_path, report)
        promoted = dict(receipt)
        promoted.update(status='COMPLETED', gpu_training_verified=True, method_id=method,
                        trainer_exit_status=receipt.get('trainer_exit_status',receipt['status']),
                        trainer_exit_receipt_path=str(preserved), trainer_exit_receipt_sha256=sha(preserved),
                        acceptance_path=str(out_path), acceptance_sha256=sha(out_path),
                        endpoint=endpoint, n_successful_optimizer_updates=256,
                        n_prompt_exposures=256*spec['prompts'], n_rollout_completions=spec['rollouts'],
                        exact_resume_verified=False,negative_outcome=negative,
                        parameter_changed_updates=changed_updates,allow_formal_zero_signal=allow_formal_zero_signal)
        if common_hash is not None: promoted['common_s0_sha256']=common_hash
        atomic_json(receipt_path, promoted)
        return report
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        # Keep the accepted report hash stable after the run receipt binds it.
        if report['status']!='COMPLETED': atomic_json(out_path, report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--method',choices=tuple(METHODS),required=True)
    parser.add_argument('--freeze',type=Path,default=PROJECT/'reports/freeze.json')
    parser.add_argument('--common-s0',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--allow-formal-zero-signal',action='store_true',
                        help='Only formal GRPO: accept an independently proved all-zero-signal negative outcome')
    args=parser.parse_args()
    report=accept(args.run_dir,args.method,args.freeze,args.out or args.run_dir/'acceptance.json',args.common_s0,
                  allow_formal_zero_signal=args.allow_formal_zero_signal)
    print(json.dumps({'status':report['status'],'method_id':report['method_id'],
                      'counts':report['counts_recomputed'],'endpoint':report['endpoint']['path']}))


if __name__=='__main__':main()
