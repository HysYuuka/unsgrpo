#!/usr/bin/env python3
"""Full processor/template audit without loading Qwen model weights.

The source-only mode is useful while dependencies/assets download. Its status is
SOURCE_INSPECTED, which cannot satisfy the training gate. A real complete audit
encodes every row in every input file with explicit truncation_strategy='raise'.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from unsgrpo.data import audit_window_record, canonical_hash


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_processor_file_hashes(root: Path) -> dict:
    """Hash local processor/tokenizer/template/config bytes, excluding weights.

    Keep raw hashes so export-induced serialization or config changes are visible;
    callers must verify legitimate serialization differences explicitly rather
    than silently pretending different files are byte-identical.
    """
    prefixes = ('processor', 'preprocessor', 'image_processor', 'video_processor',
                'feature_extractor', 'tokenizer', 'vocab', 'merges', 'chat_template',
                'special_tokens', 'added_tokens')
    excluded = ('.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.partial')
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name.endswith(excluded):
            continue
        name = path.name
        if name.endswith('config.json') or name.startswith(prefixes):
            files.append(path)
    if not files:
        raise ValueError('No local processor/tokenizer/template/config assets found')
    return {str(path.relative_to(root)): sha256(path) for path in files}


def model_processor_semantic_hashes(root: Path) -> dict:
    """Canonical JSON hashes, omitting only declared serialization metadata.

    Dtype is omitted because the registered runtime independently requires BF16.
    Processor classes, image sizes, temporal settings and tokenizer values are
    all retained. A changed semantic hash requires an explicit full encode check.
    """
    ignored = {'_name_or_path', 'transformers_version', 'torch_dtype', 'dtype'}
    def clean(value):
        if isinstance(value, dict):
            return {key:clean(item) for key,item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    raw = model_processor_file_hashes(root)
    result = {}
    for name, digest in raw.items():
        path = root/name
        if path.suffix == '.json':
            content = clean(json.loads(path.read_text()))
            result[name] = hashlib.sha256(json.dumps(content, sort_keys=True,
                ensure_ascii=False, separators=(',',':'),allow_nan=False).encode()).hexdigest()
        else:
            result[name] = digest
    return result


def source_evidence(root: Path, source_kind: str) -> dict:
    """Inspect and execute the framework's exact pure advantage function on CPU."""
    import torch
    paths = {
        'advantages': root / 'rl_core/advantage.py',
        'trainer_std': root / 'rlhf_trainers/grpo_trainer.py',
        'grpo_truncation': root / 'arguments/rlhf_args.py',
        'template_truncation': root / 'template/base.py',
        'qwen_template': root / 'template/templates/qwen.py',
        'rollout_identity': root / 'rlhf_trainers/rollout_mixin.py',
    }
    evidence = {'source_kind': source_kind, 'sources': {}}
    for name, path in paths.items():
        text = path.read_text()
        patterns = {
            'advantages': ('grouped.std(dim=1)', 'std + 1e-4'),
            'trainer_std': ('grouped_rewards.std(dim=1)',),
            'grpo_truncation': ("self.truncation_strategy = 'left'",),
            'template_truncation': ("self.truncation_strategy == 'raise'",),
            'qwen_template': ('class Qwen3VLTemplate', "do_sample_frames"),
            'rollout_identity': ('def _add_prompt_id_to_inputs', 's.request_id ='),
        }[name]
        hits = [{'line': i, 'text': line.strip()} for i, line in enumerate(text.splitlines(), 1)
                if any(pattern in line for pattern in patterns)]
        if not hits:
            raise RuntimeError(f'Expected audited v4.6.0 source pattern missing: {path}')
        evidence['sources'][name] = {'path': str(path), 'sha256': sha256(path), 'locations': hits}
    path = paths['advantages']
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_advantages')
    isolated = ast.Module(body=[node], type_ignores=[])
    env = {'torch': torch, 'Tuple': Tuple, 'Optional': Optional}
    exec(compile(ast.fix_missing_locations(isolated), str(path), 'exec'), env)
    rewards = torch.tensor([[0.], [1.], [2.], [3.]])
    actual, returned_rewards = env['compute_advantages'](rewards, torch.ones(1), 4, 'grpo', 'group')
    expected = (rewards[:, 0] - 1.5) / (rewards[:, 0].std(correction=1) + 1e-4)
    population = (rewards[:, 0] - 1.5) / (rewards[:, 0].std(correction=0) + 1e-4)
    if not torch.allclose(actual, expected) or torch.allclose(actual, population):
        raise RuntimeError('Installed framework group advantage is not the registered sample-std form')
    constants, _ = env['compute_advantages'](torch.ones(4, 1), torch.ones(1), 4, 'grpo', 'group')
    if not torch.equal(constants, torch.zeros(4)):
        raise RuntimeError('Constant-reward framework group does not yield zero advantages')
    evidence['advantage_numerical_check'] = {
        'method': 'Execute unchanged compute_advantages function AST extracted from identified framework source',
        'fixture_rewards': rewards[:, 0].tolist(), 'actual_advantages': actual.tolist(),
        'expected_sample_std_advantages': expected.tolist(), 'population_std_advantages': population.tolist(),
        'constant_group_advantages': constants.tolist(), 'correction': 1, 'denominator_epsilon': 1e-4,
        'source_function_sha256': hashlib.sha256(ast.get_source_segment(path.read_text(), node).encode()).hexdigest(),
        'passed': True,
    }
    evidence['grpo_truncation'] = 'Framework requires left/delete; audit uses raise and reserves completion length before training.'
    evidence['identity'] = 'Dataset request_id is overwritten per rollout; preserve dataset_request_id and log framework UUID separately.'
    return evidence


def tensor_hash(encoded, floating_dtype=None) -> str:
    import torch
    digest = hashlib.sha256()
    for key in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'):
        if key in encoded:
            value = encoded[key].detach().cpu().contiguous()
            if floating_dtype is not None and value.is_floating_point():
                value = value.to(floating_dtype).contiguous()
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def reconstructed_media_preview(encoded, processor, destination: Path):
    """Undo Qwen patch arrangement and normalization to show actual resized pixels."""
    from PIL import Image, ImageDraw
    destination.mkdir(parents=True, exist_ok=True)
    previews = []
    for kind, pixel_key, grid_key in (('overview', 'pixel_values', 'image_grid_thw'),
                                      ('window', 'pixel_values_videos', 'video_grid_thw')):
        grid = encoded[grid_key][0].tolist()
        gt, gh, gw = map(int, grid)
        proc = processor.image_processor if kind == 'overview' else getattr(processor, 'video_processor', processor.image_processor)
        patch = int(proc.patch_size)
        temporal = int(proc.temporal_patch_size)
        merge = int(proc.merge_size)
        raw = encoded[pixel_key][:gt * gh * gw].detach().cpu().float().numpy()
        arr = raw.reshape(gt, gh // merge, gw // merge, merge, merge, 3, temporal, patch, patch)
        arr = arr.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(gt * temporal, 3, gh * patch, gw * patch)
        mean = np.asarray(proc.image_mean).reshape(1, 3, 1, 1)
        std = np.asarray(proc.image_std).reshape(1, 3, 1, 1)
        arr = np.rint(np.clip((arr * std + mean) * 255, 0, 255)).astype(np.uint8)
        chosen = [0] if kind == 'overview' else sorted(set([0, len(arr) // 2, len(arr) - 1]))
        for index in chosen:
            path = destination / f'{kind}_processed_frame{index:03d}.png'
            Image.fromarray(arr[index].transpose(1, 2, 0)).save(path)
            previews.append(str(path))
    return previews


def encode_audit_row(row, template, processor, max_length, max_completion_length, preview_dir=None):
    audit_window_record(row, check_media_files=True)
    prompt = copy.deepcopy(row)
    if row['task'] == 'sft':
        prompt['messages'] = prompt['messages'][:2]
    # Actual inference-mode prefix with no target is encoded separately, retaining
    # all image/video processing. Metadata never becomes assistant supervision.
    template.set_mode('transformers')
    encoded_prompt = template.encode(prompt)
    pids = list(encoded_prompt['input_ids'])
    if len(pids) + max_completion_length > max_length:
        raise ValueError(f'Prompt needs {len(pids)} + {max_completion_length} completion tokens > {max_length}')
    if row['task'] == 'sft':
        template.set_mode('train')
        encoded = template.encode(copy.deepcopy(row))
        ids = list(encoded['input_ids'])
        labels = list(encoded['labels'])
        supervised = [i for i, label in enumerate(labels) if label != -100]
        if not supervised or supervised != list(range(supervised[0], len(ids))):
            raise ValueError('Assistant loss must be a nonempty contiguous suffix')
        if pids != ids[:len(pids)] or supervised[0] != len(pids):
            raise ValueError('SFT loss includes prompt tokens or prompt differs from inference prefix')
        answer_ids = processor.tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False)
        supervised_ids = [labels[i] for i in supervised]
        if supervised_ids[:len(answer_ids)] != answer_ids:
            raise ValueError('Supervised tokens do not match only the assistant answer')
        suffix_ids = template._encode_context_list(template.template_meta.suffix)[0]
        if supervised_ids[len(answer_ids):] != suffix_ids:
            raise ValueError('Unexpected tokens beyond assistant answer and registered EOS suffix')
        if any(labels[i] != ids[i] for i in supervised):
            raise ValueError('Assistant labels differ from target token IDs')
        if tensor_hash(encoded) != tensor_hash(encoded_prompt):
            raise ValueError('SFT answer changed the processed media')
    else:
        encoded, ids, labels, supervised = encoded_prompt, pids, None, []
        if encoded.get('labels') is not None:
            raise ValueError('GRPO/evaluation generation prompt unexpectedly has supervised labels')
        suffix_ids = []
    if len(ids) > max_length:
        raise ValueError('Unexpected truncation/overlength after strict encode')
    image_grid = encoded['image_grid_thw'].tolist()
    video_grid = encoded['video_grid_thw'].tolist()
    if len(image_grid) != 1 or len(video_grid) != 1:
        raise ValueError('One overview image and one video grid are required')
    video_processor = getattr(processor, 'video_processor', processor.image_processor)
    temporal = int(video_processor.temporal_patch_size)
    input_frames, processed_frames = len(row['videos'][0]), int(video_grid[0][0]) * temporal
    if input_frames != 4 * len(row['clip_ids']) or processed_frames != input_frames:
        raise ValueError(f'Processor frame resampling changed {input_frames} frames to {processed_frames}')
    merge = int(processor.image_processor.merge_size) ** 2
    expected_image_tokens = int(np.prod(image_grid[0])) // merge
    expected_video_tokens = int(np.prod(video_grid[0])) // merge
    actual_image_tokens = ids.count(template.image_token_id)
    actual_video_tokens = ids.count(template.video_token_id)
    if (actual_image_tokens, actual_video_tokens) != (expected_image_tokens, expected_video_tokens):
        raise ValueError('Grid/visual placeholder token counts differ')
    pixel_counts = (int(encoded['pixel_values'].shape[0]), int(encoded['pixel_values_videos'].shape[0]))
    if pixel_counts != (int(np.prod(image_grid[0])), int(np.prod(video_grid[0]))):
        raise ValueError('Media patch tensor/grid mismatch')
    if any(labels is not None and labels[i] != -100 for i, token in enumerate(ids)
           if token in (template.image_token_id, template.video_token_id)):
        raise ValueError('Visual tokens enter assistant-only SFT loss')
    result = {
        'request_id': row['request_id'], 'dataset_request_id': row.get('dataset_request_id', row['request_id']),
        'task': row['task'], 'clip_ids': row['clip_ids'], 'media_signature': row['media_signature'],
        'media_tensor_sha256': tensor_hash(encoded),
        'media_tensor_bf16_sha256': tensor_hash(encoded, __import__('torch').bfloat16),
        'prompt_tokens': len(pids), 'encoded_tokens': len(ids),
        'supervised_tokens': len(supervised), 'assistant_only_labels_verified': row['task'] == 'sft',
        'answer_eos_suffix_ids': suffix_ids, 'input_frames': input_frames, 'processed_frames': processed_frames,
        'image_grid_thw': image_grid, 'video_grid_thw': video_grid,
        'image_placeholder_tokens': actual_image_tokens, 'video_placeholder_tokens': actual_video_tokens,
        'pixel_values_shape': list(encoded['pixel_values'].shape),
        'pixel_values_videos_shape': list(encoded['pixel_values_videos'].shape),
        'input_ids_sha256': hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
        'labels_sha256': hashlib.sha256(np.asarray(labels,dtype=np.int64).tobytes()).hexdigest() if labels is not None else None,
        'no_truncation': True, 'completion_length_reserved': max_completion_length,
    }
    if preview_dir is not None:
        result['processed_previews'] = reconstructed_media_preview(encoded, processor, preview_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--inputs', type=Path, nargs='+', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rows-out', type=Path)
    parser.add_argument('--preview-dir', type=Path)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--max-completion-length', type=int, default=128)
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--source-root', type=Path, help='Root containing swift/, only for sources-only inspection')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'all_rows_encoded': False, 'gpu_training_verified': False, 'errors': [],
              'input_sha256': {}, 'n_rows_encoded': 0}
    started = time.monotonic()
    def save():
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    save()
    try:
        if args.sources_only:
            if args.source_root is None:
                raise ValueError('--sources-only requires --source-root')
            report['framework_source'] = source_evidence(args.source_root.resolve() / 'swift', 'downloaded_official_v4.6.0')
            report['status'] = 'SOURCE_INSPECTED'
            return
        if not args.inputs or args.model is None or not args.model.is_dir():
            raise ValueError('Full audit requires local --model and all --inputs JSONL files')
        import swift
        from swift.model import get_model_processor
        from swift.template import get_template
        if importlib.metadata.version('ms-swift') != '4.6.0':
            raise ValueError('The registered protocol requires ms-swift==4.6.0')
        report['versions'] = {package: importlib.metadata.version(package) for package in
                              ('ms-swift', 'transformers', 'torch', 'peft', 'trl', 'qwen-vl-utils')}
        report['framework_source'] = source_evidence(Path(swift.__file__).resolve().parent, 'installed_distribution')
        model, processor = get_model_processor(str(args.model.resolve()), load_model=False,
                                               download_model=False, use_hf=True, model_type='qwen3_vl')
        if model is not None:
            raise RuntimeError('Audit unexpectedly loaded full model weights')
        template = get_template(processor, template_type='qwen3_vl', max_length=args.max_length,
                                truncation_strategy='raise', remove_unused_columns=False,
                                loss_scale='default', padding_free=False)
        report['model_path'] = str(args.model.resolve())
        report['model_processor_files_sha256'] = model_processor_file_hashes(args.model.resolve())
        report['model_processor_semantic_sha256'] = model_processor_semantic_hashes(args.model.resolve())
        report['chat_template_sha256'] = hashlib.sha256(str(getattr(processor.tokenizer, 'chat_template', '')).encode()).hexdigest()
        report['template_class'] = type(template).__name__
        report['truncation_strategy'] = template.truncation_strategy
        report['visual_environment'] = {key: os.environ.get(key) for key in
                                        ('IMAGE_MAX_TOKEN_NUM', 'VIDEO_MIN_TOKEN_NUM', 'VIDEO_MAX_TOKEN_NUM', 'FPS', 'FPS_MAX_FRAMES', 'FPS_MIN_FRAMES')}
        destination = args.rows_out or args.out.with_suffix('.rows.jsonl')
        if destination.exists():
            raise FileExistsError(f'Refusing to append/reuse audit row file: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        seen_by_file = {}
        instances_by_file, counts_by_file, cache = {}, {}, {}
        report.update(n_actual_row_encodes=0, n_cached_identical_rows=0)
        max_tokens, min_frames, max_frames = 0, None, 0
        with destination.open('x') as handle:
            for path in args.inputs:
                absolute = path.resolve()
                report['input_sha256'][str(absolute)] = sha256(absolute)
                seen = seen_by_file.setdefault(str(absolute), {})
                instances = instances_by_file.setdefault(str(absolute),set())
                counts_by_file[str(absolute)] = 0
                for line_no, line in enumerate(path.read_text().splitlines(), 1):
                    if not line.strip():
                        raise ValueError(f'Blank line in {path}:{line_no}')
                    row = json.loads(line)
                    identity = row['request_id']
                    content_hash = canonical_hash({k:v for k,v in row.items() if k!='pool_instance_id'})
                    if identity in seen:
                        if row['task']=='eval' or content_hash!=seen[identity]:
                            raise ValueError(f'Duplicate eval ID or changed repeated train window in {path}: {identity}')
                        if not row.get('pool_instance_id'):
                            raise ValueError('Repeated train windows require explicit unique pool_instance_id')
                    instance = row.get('pool_instance_id')
                    if instance is not None and instance in instances:
                        raise ValueError('Duplicate pool instance ID')
                    if instance is not None: instances.add(instance)
                    seen[identity] = content_hash
                    counts_by_file[str(absolute)] += 1
                    preview = (args.preview_dir / f'{path.stem}_{row["request_id"]}') if args.preview_dir and len(seen) == 1 else None
                    audit_window_record(row, check_media_files=True)
                    if content_hash in cache:
                        result = copy.deepcopy(cache[content_hash])
                        report['n_cached_identical_rows'] += 1
                    else:
                        result = encode_audit_row(row, template, processor, args.max_length,
                                                  args.max_completion_length, preview)
                        cache[content_hash] = copy.deepcopy(result)
                        report['n_actual_row_encodes'] += 1
                    result.update(input_file=str(absolute), line_no=line_no,pool_instance_id=instance,
                                  row_content_sha256=content_hash)
                    handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    report['n_rows_encoded'] += 1
                    max_tokens = max(max_tokens, result['encoded_tokens'])
                    min_frames = result['processed_frames'] if min_frames is None else min(min_frames, result['processed_frames'])
                    max_frames = max(max_frames, result['processed_frames'])
                    if report['n_rows_encoded'] % 10 == 0:
                        save()
                        print(json.dumps({'encoded': report['n_rows_encoded'], 'max_tokens': max_tokens}), flush=True)
        if not report['n_rows_encoded']:
            raise ValueError('No input rows were audited')
        # Detect any concurrent mutation during the potentially long full-input audit.
        if any(sha256(path) != digest for path, digest in report['input_sha256'].items()):
            raise RuntimeError('Input dataset changed while encoding')
        report.update(status='COMPLETED', all_rows_encoded=True, rows_path=str(destination.resolve()),
                      rows_sha256=sha256(destination), max_encoded_tokens=max_tokens,
                      min_processed_frames=min_frames, max_processed_frames=max_frames,
                      per_file_rows=counts_by_file,
                      per_file_unique_windows={path: len(ids) for path, ids in seen_by_file.items()},
                      cache_policy='Only identical full row content excluding pool_instance_id reuses a prior actual encode; every row is validated and receives an audit row.',
                      max_length=args.max_length, max_completion_length=args.max_completion_length)
    except BaseException as exc:
        report['status'] = 'FAILED'
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
