#!/usr/bin/env python3
"""Real-GPU fixed-answer checkpoint and S0 reference integrity checks.

snapshot records pre-merge token probabilities and greedy output. compare-merged
uses the fixed, preregistered BF16 thresholds (mean absolute difference <= .03,
maximum <= .25). reference-update performs one assistant-only CE optimizer step
solely to test fresh LoRA/reference integrity; it is NOT a GRPO training receipt.
No evaluation labels or test examples are accepted.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from unsgrpo.core import parse_scores, reward_parts
from unsgrpo.data import audit_window_record, canonical_hash

MERGE_MEAN_ABS_TOLERANCE = 0.03
MERGE_MAX_ABS_TOLERANCE = 0.25


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_file_hashes(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f'Local model/adapter directory required: {root}')
    suffixes = {'.json', '.safetensors', '.bin', '.model', '.txt'}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in suffixes)
    if not paths:
        raise ValueError(f'No model/processor files found: {root}')
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': file_sha256(p)} for p in paths}


def select_train_row(path: Path, row_index: int = 0, request_id: str | None = None) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if request_id is not None:
        rows = [row for row in rows if row.get('dataset_request_id', row.get('request_id')) == request_id]
        row_index = 0
    if not rows or not 0 <= row_index < len(rows):
        raise ValueError('No matching fixed training example')
    row = rows[row_index]
    audit_window_record(row, check_media_files=True)
    if row['task'] != 'sft' or row['split'] != 'train':
        raise ValueError('Checkpoint verification accepts only pseudo-labelled train SFT records')
    parse_scores(row['messages'][-1]['content'], len(row['clip_ids']))
    return row


def binding(path: Path, row: dict) -> dict:
    return {'inputs_path': str(path.resolve()), 'inputs_sha256': file_sha256(path),
            'request_id': row['request_id'], 'dataset_request_id': row.get('dataset_request_id', row['request_id']),
            'pool_instance_id': row.get('pool_instance_id'), 'media_signature': row['media_signature'],
            'clip_ids': row['clip_ids'], 'assistant_answer': row['messages'][-1]['content'],
            'row_sha256': canonical_hash(row)}


def difference(before, after, *, mean_tolerance=MERGE_MEAN_ABS_TOLERANCE,
               max_tolerance=MERGE_MAX_ABS_TOLERANCE) -> dict:
    a, b = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if a.ndim != 1 or not len(a) or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Fixed-answer token probabilities must be finite aligned vectors')
    delta = np.abs(a - b)
    return {'mean_abs': float(delta.mean()), 'max_abs': float(delta.max()),
            'mean_abs_tolerance': mean_tolerance, 'max_abs_tolerance': max_tolerance,
            'exact_equal': bool(np.array_equal(a, b)),
            'passed': bool(delta.mean() <= mean_tolerance and delta.max() <= max_tolerance)}


@contextmanager
def measured(report: dict, name: str):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        report.setdefault('stages', {})[name] = {
            'wall_seconds': time.monotonic() - started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        }


def load_engine(model_path: Path, adapter_path: Path | None = None):
    import torch
    from swift.infer_engine import TransformersEngine
    if not model_path.is_dir() or (adapter_path is not None and not adapter_path.is_dir()):
        raise ValueError('Use complete local model and adapter paths')
    engine = TransformersEngine(str(model_path.resolve()),
                                adapters=[str(adapter_path.resolve())] if adapter_path else None,
                                model_type='qwen3_vl', torch_dtype=torch.bfloat16,
                                attn_impl='sdpa', device_map='cuda:0', use_hf=True, max_batch_size=1)
    engine.template.max_length = 8192
    engine.template.truncation_strategy = 'raise'
    engine.model.config.use_cache = False
    return engine


def prepare_answer_batch(engine, row: dict):
    """Follow the real TransformersEngine encode/collate/pre_forward_hook path."""
    import torch
    from swift.utils import to_device
    template = engine.template
    template.model = engine.model
    template.set_mode('train')
    payload = {key: copy.deepcopy(row[key]) for key in ('messages', 'images', 'videos')}
    encoded = template.encode(payload)
    if len(encoded['input_ids']) > 8192:
        raise ValueError('Fixed answer exceeded strict length budget')
    unmasked = [i for i,label in enumerate(encoded['labels']) if label!=-100]
    actual_targets = [encoded['labels'][i] for i in unmasked]
    expected_targets = template.tokenizer.encode(row['messages'][-1]['content'],add_special_tokens=False)+template._encode_context_list(template.template_meta.suffix)[0]
    if not unmasked or actual_targets!=expected_targets or unmasked!=list(range(unmasked[0],len(encoded['input_ids']))):
        raise ValueError('Actual template labels are not exactly the assistant answer and EOS suffix')
    collated = template.data_collator([encoded])
    # template.encode uses inference_mode for preprocessing. Cloning outside that
    # context ensures saved tensors are ordinary tensors during the gradient test.
    def ordinary(value):
        if isinstance(value, torch.Tensor): return value.clone()
        if isinstance(value, dict): return {key: ordinary(item) for key, item in value.items()}
        if isinstance(value, list): return [ordinary(item) for item in value]
        if isinstance(value, tuple): return tuple(ordinary(item) for item in value)
        return value
    collated = to_device(ordinary(collated), engine.model.device)
    _, collated = template.pre_forward_hook(engine.model, None, collated)
    labels = collated.pop('labels')
    if not torch.any(labels[:, 1:] != -100):
        raise ValueError('No shifted assistant answer tokens')
    for key in ('loss_scale', '_extra_kwargs'):
        collated.pop(key, None)
    return collated, labels


def answer_logps(model, batch: dict, labels, *, require_grad=False):
    import torch
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        outputs = model(**batch, use_cache=False, return_dict=True)
        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
            raise ValueError('Model logits and assistant labels are not aligned')
        positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
        targets = labels[0, 1:][positions]
        selected = logits[0, positions].float()
        logps = torch.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
        if not torch.isfinite(logps).all():
            raise FloatingPointError('Non-finite fixed-answer log probabilities')
        if require_grad and not logps.requires_grad:
            raise RuntimeError('Current-policy log probabilities lost their gradient')
    return logps, targets.detach().cpu().tolist(), positions.detach().cpu().tolist()


def forward_record(engine, row, report, name):
    engine.model.eval()
    with measured(report, name):
        batch, labels = prepare_answer_batch(engine, row)
        logps, token_ids, positions = answer_logps(engine.model, batch, labels)
    return {'token_logps': logps.detach().cpu().tolist(), 'answer_token_ids': token_ids,
            'prediction_positions': positions, 'supervised_tokens_including_eos': len(token_ids),
            'input_ids_sha256': hashlib.sha256(json.dumps(batch['input_ids'][0].tolist(), separators=(',', ':')).encode()).hexdigest()}


def greedy_record(engine, row, report, name):
    from swift.infer_engine import InferRequest, RequestConfig
    with measured(report, name):
        request = InferRequest(messages=copy.deepcopy(row['messages'][:2]),
                               images=row['images'], videos=row['videos'])
        config = RequestConfig(max_tokens=128, temperature=0, top_p=1, top_k=0,
                               repetition_penalty=1, return_details=True)
        response = engine.infer([request], config)[0]
        answer = response.choices[0].message.content
    parts = reward_parts(answer, row['pseudo_scores'])
    usage = response.usage
    return {'response': answer, 'finish_reason': response.choices[0].finish_reason,
            'valid_json': parts.valid, 'proxy_reward': parts.total,
            'prompt_tokens': getattr(usage, 'prompt_tokens', None),
            'completion_tokens': getattr(usage, 'completion_tokens', None)}


def parameter_fingerprints(model, *, lora_only=False, exclude_lora=False) -> dict:
    import torch
    result = {}
    for name, parameter in sorted(model.named_parameters()):
        is_lora = 'lora_' in name
        if lora_only and not is_lora or exclude_lora and is_lora:
            continue
        value = parameter.detach().cpu().contiguous()
        digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        result[name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': digest,
                        'requires_grad': parameter.requires_grad}
    return result


def add_fresh_language_lora(engine):
    import torch
    from peft import LoraConfig, get_peft_model
    if any('lora_' in name for name, _ in engine.model.named_parameters()):
        raise ValueError('Reference-update must start from the merged S0 with no existing adapter')
    prefixes = [name for name, _ in engine.model.named_modules() if name.endswith('language_model')]
    if len(prefixes) != 1:
        raise ValueError(f'Cannot unambiguously identify Qwen language tower: {prefixes}')
    prefix = prefixes[0] + '.'
    targets = [name for name, module in engine.model.named_modules()
               if name.startswith(prefix) and isinstance(module, torch.nn.Linear)]
    if not targets or any(any(tag in name for tag in ('visual', 'vision', 'merger', 'aligner')) for name in targets):
        raise ValueError('LoRA targets are missing or include frozen visual/aligner components')
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0., bias='none', task_type='CAUSAL_LM',
                        target_modules=targets, init_lora_weights=True)
    model = get_peft_model(engine.model, config)
    engine.model = engine.engine = model
    engine.template.model = model
    model.config.use_cache = False
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any('lora_' not in name or any(tag in name for tag in ('visual', 'vision', 'merger', 'aligner'))
                            for name in trainable):
        raise RuntimeError('Unexpected trainable base, visual, aligner or non-LoRA parameters')
    return model, targets, trainable


def reference_update(args, row, report):
    import torch
    if args.adapter is not None:
        raise ValueError('reference-update requires a plain merged S0 directory, not an existing adapter')
    if args.adapter_out is None:
        raise ValueError('reference-update requires explicit --adapter-out outside the repository')
    destination = args.adapter_out.resolve()
    if destination.exists():
        raise FileExistsError('Fresh test adapter output directory must not exist')
    if destination == PROJECT or PROJECT in destination.parents:
        raise ValueError('Adapter weights must be outside the code repository')
    with measured(report, 'load_merged_s0'):
        engine = load_engine(args.model)
    report['merged_base_before_adapter'] = forward_record(engine, row, report, 'merged_base_fixed_answer')
    model, targets, trainable = add_fresh_language_lora(engine)
    report['lora_spec'] = {'rank': 16, 'alpha': 32, 'dropout': 0., 'bias': 'none',
                           'target_module_names': targets, 'trainable_parameter_names': trainable,
                           'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    report['trainable_parameters_before'] = parameter_fingerprints(model, lora_only=True)
    report['frozen_base_parameters_before'] = parameter_fingerprints(model, exclude_lora=True)
    initial = forward_record(engine, row, report, 'new_adapter_initial_answer')
    report['new_adapter_initial'] = initial
    initial_difference = difference(report['merged_base_before_adapter']['token_logps'], initial['token_logps'],
                                    mean_tolerance=1e-6, max_tolerance=1e-6)
    report['new_adapter_initial_equivalence'] = initial_difference
    if not initial_difference['passed']:
        raise RuntimeError('New zero-B LoRA does not reproduce the initial merged S0 function')
    model.eval()
    with model.disable_adapter():
        reference_before = forward_record(engine, row, report, 'disabled_reference_before')
    report['reference_before'] = reference_before
    if not difference(report['merged_base_before_adapter']['token_logps'], reference_before['token_logps'],
                      mean_tolerance=0., max_tolerance=0.)['passed']:
        raise RuntimeError('Disabled new adapter does not exactly recover the merged S0 reference')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.)
    model.train()
    with measured(report, 'forward_and_backward'):
        batch, labels = prepare_answer_batch(engine, row)
        current, token_ids, positions = answer_logps(model, batch, labels, require_grad=True)
        loss = -current.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f'Missing/non-finite fresh LoRA gradient: {name}')
                gradients[name] = {'norm': float(parameter.grad.float().norm()),
                                   'max_abs': float(parameter.grad.float().abs().max())}
            elif parameter.grad is not None:
                raise RuntimeError(f'Frozen parameter received a gradient: {name}')
        if not any(item['norm'] > 0 for item in gradients.values()):
            raise RuntimeError('All fresh LoRA gradients are zero')
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError('Non-finite clipped gradient norm')
    with measured(report, 'optimizer_step'):
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    report['integrity_test_update'] = {
        'objective': 'assistant_only_cross_entropy_for_reference_integrity_test; not GRPO',
        'loss': float(loss.detach()), 'n_optimizer_updates': 1, 'learning_rate': 1e-5,
        'grad_norm_before_clip': float(grad_norm), 'per_parameter_gradients': gradients,
        'legal_pseudo_answer_proxy_reward': reward_parts(row['messages'][-1]['content'], row['pseudo_scores']).total,
        'n_answer_tokens_including_eos': len(token_ids),
    }
    report['trainable_parameters_after'] = parameter_fingerprints(model, lora_only=True)
    before = report['trainable_parameters_before']
    after = report['trainable_parameters_after']
    changed = [name for name in before if before[name]['sha256'] != after[name]['sha256']]
    if not changed:
        raise RuntimeError('Optimizer step did not change any trainable LoRA parameter')
    report['changed_lora_parameters'] = changed
    report['frozen_base_parameters_after'] = parameter_fingerprints(model, exclude_lora=True)
    if report['frozen_base_parameters_before'] != report['frozen_base_parameters_after']:
        raise RuntimeError('Frozen merged-S0 parameters changed during the new-adapter update')
    with model.disable_adapter():
        reference_after = forward_record(engine, row, report, 'disabled_reference_after')
    report['reference_after'] = reference_after
    report['reference_invariance'] = difference(reference_before['token_logps'], reference_after['token_logps'],
                                               mean_tolerance=0., max_tolerance=0.)
    if not report['reference_invariance']['exact_equal']:
        raise RuntimeError('Fixed S0 reference token log probabilities drifted after adapter update')
    updated = forward_record(engine, row, report, 'updated_adapter_answer')
    report['updated_adapter'] = updated
    report['updated_greedy'] = greedy_record(engine, row, report, 'updated_adapter_greedy')
    with measured(report, 'save_adapter'):
        model.save_pretrained(str(destination), safe_serialization=True)
        engine.template.processor.save_pretrained(str(destination))
    report['saved_adapter_path'] = str(destination)
    report['saved_adapter_file_hashes'] = model_file_hashes(destination)
    # Reload from disk after releasing the first engine, rather than reusing its
    # updated in-memory adapter and calling that a save/reload verification.
    del current, loss, batch, labels, optimizer, model, engine
    gc.collect()
    torch.cuda.empty_cache()
    with measured(report, 'reload_adapter_from_disk'):
        reloaded = load_engine(args.model, destination)
    reloaded_answer = forward_record(reloaded, row, report, 'reloaded_adapter_answer')
    report['reloaded_adapter'] = reloaded_answer
    if updated['answer_token_ids'] != reloaded_answer['answer_token_ids']:
        raise RuntimeError('Reload changed fixed answer token IDs')
    report['save_reload_equivalence'] = difference(updated['token_logps'], reloaded_answer['token_logps'],
                                                  mean_tolerance=1e-6, max_tolerance=1e-6)
    if not report['save_reload_equivalence']['passed']:
        raise RuntimeError('Saved/reloaded adapter changed fixed-answer probabilities')
    report['reloaded_greedy'] = greedy_record(reloaded, row, report, 'reloaded_adapter_greedy')
    report['save_reload_greedy_equal'] = report['updated_greedy']['response'] == report['reloaded_greedy']['response']
    if not report['save_reload_greedy_equal']:
        raise RuntimeError('Saved/reloaded adapter changed greedy output')
    report['gpu_reference_update_verified'] = True
    report['gpu_grpo_training_verified'] = False
    report['exact_optimizer_resume_verified'] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('snapshot', 'compare-merged', 'reference-update'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--row-index', type=int, default=0)
    parser.add_argument('--request-id')
    parser.add_argument('--snapshot', type=Path, help='Required pre-merge snapshot for compare-merged')
    parser.add_argument('--adapter-out', type=Path, help='New saved adapter outside repository for reference-update')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new checkpoint-verification report path')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'mode': args.mode, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'gpu_checkpoint_verification': False, 'gpu_grpo_training_verified': False,
              'human_highlight_labels_read': False,
              'merge_tolerances_preregistered': {'mean_abs': MERGE_MEAN_ABS_TOLERANCE, 'max_abs': MERGE_MAX_ABS_TOLERANCE}}
    started = time.monotonic()
    def save():
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    save()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This verification requires the real single GPU; CPU/source inspection cannot pass')
        if version('ms-swift') != '4.6.0':
            raise RuntimeError('Expected audited ms-swift==4.6.0')
        report['versions'] = {name: version(name) for name in ('ms-swift', 'torch', 'transformers', 'peft')}
        report['gpu_name'] = torch.cuda.get_device_name()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        row = select_train_row(args.inputs, args.row_index, args.request_id)
        report['fixed_training_example'] = binding(args.inputs, row)
        report['model_path'] = str(args.model.resolve())
        report['model_file_hashes'] = model_file_hashes(args.model.resolve())
        if args.adapter is not None:
            report['adapter_path'] = str(args.adapter.resolve())
            report['adapter_file_hashes'] = model_file_hashes(args.adapter.resolve())
        if args.mode == 'reference-update':
            reference_update(args, row, report)
        else:
            if args.mode == 'compare-merged':
                if args.snapshot is None or args.adapter is not None:
                    raise ValueError('compare-merged requires --snapshot and a plain merged --model')
                before = json.loads(args.snapshot.read_text())
                if before.get('status') != 'COMPLETED' or before.get('mode') != 'snapshot' or before.get('gpu_checkpoint_verification') is not True:
                    raise ValueError('Comparison requires a completed real-GPU snapshot')
                if before['fixed_training_example'] != report['fixed_training_example']:
                    raise ValueError('Merged comparison must use the exact same fixed training file and row')
                report['snapshot_path'] = str(args.snapshot.resolve())
                report['snapshot_sha256'] = file_sha256(args.snapshot)
            with measured(report, 'load_model'):
                engine = load_engine(args.model, args.adapter)
            report['fixed_answer'] = forward_record(engine, row, report, 'fixed_answer_forward')
            report['greedy'] = greedy_record(engine, row, report, 'greedy_generation')
            report['adapter_parameter_hashes'] = parameter_fingerprints(engine.model, lora_only=True)
            if args.mode == 'compare-merged':
                if before['fixed_answer']['answer_token_ids'] != report['fixed_answer']['answer_token_ids']:
                    raise RuntimeError('Token identities changed between SFT adapter and merged S0')
                report['merge_logp_comparison'] = difference(before['fixed_answer']['token_logps'], report['fixed_answer']['token_logps'])
                report['greedy_equal_before_after_merge'] = before['greedy']['response'] == report['greedy']['response']
                if not report['merge_logp_comparison']['passed']:
                    raise RuntimeError('Merge exceeded preregistered BF16 logp tolerance; retain failure and investigate')
        report.update(status='COMPLETED', gpu_checkpoint_verification=True)
    except BaseException as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
