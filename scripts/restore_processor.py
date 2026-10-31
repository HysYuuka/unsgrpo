#!/usr/bin/env python3
"""Restore the audited original processor bytes after a verified LoRA export.

Only metadata differences are allowed in exported model config. Architecture,
rope, token IDs and every other config value must match before any files change.
Exported weights are hashed before/after and never rewritten. Finally a real
processor/template encode of a fixed train SFT row must match the original and
its full-input audit exactly; this is a CPU media check, not GPU training proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

from audit_swift import (encode_audit_row, model_processor_file_hashes,
                         model_processor_semantic_hashes)
from verify_checkpoint import select_train_row

METADATA_KEYS = {'_name_or_path', 'transformers_version', 'torch_dtype', 'dtype'}
REMOVABLE_EXPORTED_PROCESSOR_FILES = {'added_tokens.json', 'special_tokens_map.json', 'chat_template.jinja'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config_semantics(value):
    if isinstance(value, dict):
        return {key: config_semantics(item) for key, item in value.items() if key not in METADATA_KEYS}
    if isinstance(value, list):
        return [config_semantics(item) for item in value]
    return value


def json_differences(first, second, path='$'):
    """List exact changed leaves and missing keys without hiding real defaults."""
    if isinstance(first, dict) and isinstance(second, dict):
        result = []
        for key in sorted(first.keys() | second.keys()):
            if key not in first:
                result.append({'path': f'{path}.{key}', 'original': '<absent>', 'exported': second[key]})
            elif key not in second:
                result.append({'path': f'{path}.{key}', 'original': first[key], 'exported': '<absent>'})
            else:
                result.extend(json_differences(first[key], second[key], f'{path}.{key}'))
        return result
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            return [{'path': path, 'original': first, 'exported': second}]
        return [difference for i, (a, b) in enumerate(zip(first, second))
                for difference in json_differences(a, b, f'{path}[{i}]')]
    if first != second:
        return [{'path': path, 'original': first, 'exported': second}]
    return []


def account_for_exact_derived_injections(original_config, exported_config, original_pad_token_id,
                                         *, config_sha256, tokenizer_files_sha256):
    """Accept only Swift's observed redundant fields, with independent values.

    Existing original keys are never overridden. Unknown fields and any wrong
    hidden size or pad token remain strict semantic differences.
    """
    original = config_semantics(original_config)
    exported = config_semantics(exported_config)
    injections = []
    hidden = original_config.get('text_config', {}).get('hidden_size')
    if ('hidden_size' not in original_config and type(hidden) is int and hidden > 0
            and exported_config.get('hidden_size') == hidden):
        exported.pop('hidden_size', None)
        injections.append({'path':'$.hidden_size','exported_value':hidden,
                           'derived_from':'original config.json $.text_config.hidden_size',
                           'source_sha256':config_sha256})
    if type(original_pad_token_id) is int and original_pad_token_id >= 0:
        if 'pad_token_id' not in original_config and exported_config.get('pad_token_id') == original_pad_token_id:
            exported.pop('pad_token_id', None)
            injections.append({'path':'$.pad_token_id','exported_value':original_pad_token_id,
                               'derived_from':'AutoTokenizer loaded from the audited original package .pad_token_id',
                               'source_tokenizer_files_sha256':tokenizer_files_sha256})
        text_original = original_config.get('text_config', {})
        text_exported = exported_config.get('text_config', {})
        if 'pad_token_id' not in text_original and text_exported.get('pad_token_id') == original_pad_token_id:
            exported['text_config'].pop('pad_token_id', None)
            injections.append({'path':'$.text_config.pad_token_id','exported_value':original_pad_token_id,
                               'derived_from':'AutoTokenizer loaded from the audited original package .pad_token_id',
                               'source_tokenizer_files_sha256':tokenizer_files_sha256})
    return json_differences(original, exported), injections


def weight_hashes(root):
    files = sorted(path for path in root.rglob('*') if path.is_file() and path.suffix in {'.safetensors', '.bin'})
    if not files:
        raise ValueError('No exported model weights found; this is not a model export')
    return {str(path.relative_to(root)): {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files}


def encoding(processor_path, row, max_length, max_completion_length):
    from swift.model import get_model_processor
    from swift.template import get_template
    model, processor = get_model_processor(str(processor_path), load_model=False, download_model=False,
                                           use_hf=True, model_type='qwen3_vl')
    if model is not None:
        raise RuntimeError('Processor restoration check unexpectedly loaded full model weights')
    template = get_template(processor, template_type='qwen3_vl', max_length=max_length,
                            truncation_strategy='raise', remove_unused_columns=False,
                            loss_scale='default', padding_free=False)
    return encode_audit_row(row, template, processor, max_length, max_completion_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exported', type=Path, required=True)
    parser.add_argument('--input-audit', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh processor-restoration receipt path')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'model_weights_modified': False, 'gpu_training_verified': False,
              'fixed_train_row_encode_verified': False, 'human_highlight_labels_read': False}
    started = time.monotonic()
    def save():
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    save()
    try:
        from importlib.metadata import version
        if version('ms-swift') != '4.6.0':
            raise ValueError('Requires the audited ms-swift==4.6.0 environment')
        audit = json.loads(args.input_audit.read_text())
        if audit.get('status') != 'COMPLETED' or not audit.get('all_rows_encoded'):
            raise ValueError('Original full-input processor audit has not passed')
        if audit.get('input_sha256', {}).get(str(args.inputs.resolve())) != sha256(args.inputs):
            raise ValueError('Fixed training input file differs from the full-input audit')
        original, exported = Path(audit['model_path']).resolve(), args.exported.resolve()
        if original == exported:
            raise ValueError('Refusing to treat the pinned original model as an export target')
        if not original.is_dir() or not exported.is_dir():
            raise ValueError('Original and exported local model directories must exist')
        report.update(original_model_path=str(original), exported_model_path=str(exported),
                      input_audit_path=str(args.input_audit.resolve()), input_audit_sha256=sha256(args.input_audit))
        original_files = model_processor_file_hashes(original)
        if original_files != audit.get('model_processor_files_sha256'):
            raise ValueError('Pinned original processor files changed after the full-input audit')
        exported_files = model_processor_file_hashes(exported)
        report.update(original_processor_files_sha256=original_files,
                      exported_processor_files_sha256=exported_files,
                      exported_processor_semantic_sha256=model_processor_semantic_hashes(exported),
                      original_processor_semantic_sha256=model_processor_semantic_hashes(original))
        config_original = json.loads((original / 'config.json').read_text())
        config_exported = json.loads((exported / 'config.json').read_text())
        report['model_config_raw_differences'] = json_differences(config_original, config_exported)
        raw_semantic_differences = json_differences(config_semantics(config_original), config_semantics(config_exported))
        from transformers import AutoTokenizer
        original_tokenizer = AutoTokenizer.from_pretrained(original, local_files_only=True)
        semantic_differences, injections = account_for_exact_derived_injections(
            config_original, config_exported, original_tokenizer.pad_token_id,
            config_sha256=original_files['config.json'],
            tokenizer_files_sha256={name:digest for name,digest in original_files.items()
                                    if name.startswith(('tokenizer','vocab','merges'))})
        report['model_config_semantic_differences_before_derived_accounting'] = raw_semantic_differences
        report['model_config_exact_derived_injections'] = injections
        report['model_config_semantic_differences'] = semantic_differences
        report['model_config_only_declared_metadata_differs'] = not raw_semantic_differences
        report['model_config_only_metadata_or_exact_derived_injections_differ'] = not semantic_differences
        if semantic_differences:
            raise ValueError('Exported model config changed non-metadata values: ' +
                             ', '.join(item['path'] for item in semantic_differences))
        unknown = set(exported_files) - set(original_files) - REMOVABLE_EXPORTED_PROCESSOR_FILES
        if unknown:
            raise ValueError(f'Unknown newly exported processor files require investigation: {sorted(unknown)}')
        report['exported_weights_before'] = weight_hashes(exported)
        report['processor_files_raw_differences'] = {
            name: {'original_sha256': original_files.get(name), 'exported_sha256': exported_files.get(name)}
            for name in sorted(set(original_files) | set(exported_files))
            if original_files.get(name) != exported_files.get(name)}
        report['removed_export_only_processor_files'] = sorted(set(exported_files) - set(original_files))
        save()
        # Copy the audited package byte for byte. No asset_receipt or model weight
        # is copied: merged weights have their own identity and must be retained.
        for name in original_files:
            source, destination = original / name, exported / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.processor-restore-tmp')
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        for name in report['removed_export_only_processor_files']:
            (exported / name).unlink()
        restored_files = model_processor_file_hashes(exported)
        report['restored_processor_files_sha256'] = restored_files
        if restored_files != original_files:
            raise RuntimeError('Restored processor package is not byte-identical to the original')
        report['exported_weights_after'] = weight_hashes(exported)
        if report['exported_weights_after'] != report['exported_weights_before']:
            report['model_weights_modified'] = True
            raise RuntimeError('Model weight hashes changed during processor restoration')
        visual = audit['visual_environment']
        for key, value in visual.items():
            if value is None or key in os.environ and os.environ[key] != value:
                raise ValueError(f'Runtime visual environment differs from the full audit: {key}')
            os.environ[key] = value
        row = select_train_row(args.inputs)
        report['fixed_training_row'] = {'request_id': row['request_id'],
                                         'pool_instance_id': row.get('pool_instance_id'),
                                         'media_signature': row['media_signature']}
        length, completion = audit['max_length'], audit['max_completion_length']
        original_encoded = encoding(original, row, length, completion)
        restored_encoded = encoding(exported, row, length, completion)
        keys = ('input_ids_sha256', 'labels_sha256', 'media_tensor_sha256', 'media_tensor_bf16_sha256',
                'prompt_tokens', 'encoded_tokens', 'supervised_tokens', 'image_grid_thw', 'video_grid_thw',
                'input_frames', 'processed_frames')
        report['original_actual_encode'] = {key: original_encoded[key] for key in keys}
        report['restored_actual_encode'] = {key: restored_encoded[key] for key in keys}
        if report['original_actual_encode'] != report['restored_actual_encode']:
            raise RuntimeError('Restored processor differs from original actual tokens, labels, media or grids')
        rows_path = Path(audit['rows_path'])
        if sha256(rows_path) != audit['rows_sha256']:
            raise ValueError('Original full-input audit row ledger changed')
        registered = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        registered = [item for item in registered if item.get('task') == 'sft' and
                      item['dataset_request_id'] == row.get('dataset_request_id', row['request_id'])]
        if not registered or any(registered[0][key] != original_encoded[key] for key in keys):
            raise RuntimeError('Current original processor encode differs from the registered full-input audit')
        report.update(status='COMPLETED', fixed_train_row_encode_verified=True,
                      restored_entire_processor_package_byte_identical=True,
                      asset_receipt_copied=False)
    except BaseException as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
