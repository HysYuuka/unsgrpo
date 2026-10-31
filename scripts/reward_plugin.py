"""Audited ms-swift v4.6.0 reward; dataset IDs survive framework rollout UUIDs.

HL_GRPO_DATA is the exact registered train_grpo JSONL. Framework kwargs are
compared against its canonical records, not merely against each other's length.
Truncation is recorded; malformed JSON receives -1 without repair, as specified.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from unsgrpo.core import reward_parts
from unsgrpo.data import audit_window_record, canonical_hash
from swift.rewards import ORM, orms


def _decoded_completion(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get('role') == 'assistant' and isinstance(value.get('content'), str):
        return value['content']
    if isinstance(value, list) and len(value) == 1:
        return _decoded_completion(value[0])
    raise TypeError('completion must be text or one decoded assistant message')


def normalized_image_paths(images):
    """Normalize Swift's documented Arrow image wrapper without accepting bytes.

    RowPreprocessor wraps an unchanged path as {'bytes': None, 'path': path}.
    This accepts only that exact representation and ordinary paths. Actual
    consumed pixel tensors are independently matched by training telemetry.
    """
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError('images must be a nonempty list of explicit paths')
    paths = []
    for item in images:
        if isinstance(item, dict):
            if set(item) != {'bytes', 'path'} or item['bytes'] is not None:
                raise ValueError('Only the exact bytes=None,path=... image wrapper is accepted')
            item = item['path']
        if not isinstance(item, str) or not item:
            raise ValueError('Image wrapper/path must contain one nonempty string path')
        paths.append(str(Path(item).resolve()))
    return paths


class HighlightRecurrenceReward(ORM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = Path(os.environ['HL_GRPO_DATA']).resolve()
        self.dataset_sha256 = hashlib.sha256(self.dataset_path.read_bytes()).hexdigest()
        self.expected = {}
        self.instances = {}
        for line in self.dataset_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            audit_window_record(row)
            key = row.get('dataset_request_id')
            if not key or key != row['request_id']:
                raise ValueError('GRPO dataset needs dataset_request_id equal to its original window request_id')
            content = {k:v for k,v in row.items() if k!='pool_instance_id'}
            if key in self.expected and canonical_hash(content)!=canonical_hash({k:v for k,v in self.expected[key].items() if k!='pool_instance_id'}):
                raise ValueError('Repeated dataset_request_id has different media, targets or prompt')
            instance = row.get('pool_instance_id')
            if key in self.expected and instance is None:
                raise ValueError('Repeated training windows require explicit pool_instance_id')
            if instance is not None:
                if instance in self.instances: raise ValueError('Duplicate training pool instance')
                self.instances[instance] = key
            if row['task'] != 'grpo':
                raise ValueError('reward registry must contain GRPO train inputs')
            self.expected[key] = row
        if not self.expected:
            raise ValueError('empty reward registry')
        self.group_size = int(os.environ.get('HL_GROUP_SIZE', '4'))
        self.weight = float(os.environ.get('HL_RANK_WEIGHT', '.8'))
        if self.group_size < 2 or not 0 <= self.weight <= 1:
            raise ValueError('invalid group size or rank weight')
        root = Path(os.environ['HL_ROLLOUT_LEDGER']).resolve()
        root.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = root.with_name(root.stem + f'.rank{os.environ.get("RANK", "0")}' + root.suffix)
        self.calls = 0

    def __call__(self, completions, pseudo_scores, dataset_request_id=None,
                 request_id=None, prompt_id=None, trainer_state=None, **kwargs):
        size = len(completions)
        columns = dict(kwargs, pseudo_scores=pseudo_scores, dataset_request_id=dataset_request_id,
                       request_id=request_id, prompt_id=prompt_id)
        required = ('pseudo_scores', 'dataset_request_id', 'request_id', 'prompt_id',
                    'media_signature', 'clip_ids', 'clip_bounds', 'sampled_pts', 'images', 'videos', 'messages')
        for key in required:
            value = columns.get(key)
            if not isinstance(value, (list, tuple)) or len(value) != size:
                raise ValueError(f'rollout column {key} must contain {size} corresponding entries')
        if self.instances and (not isinstance(columns.get('pool_instance_id'),(list,tuple)) or len(columns['pool_instance_id'])!=size):
            raise ValueError('Training pool_instance_id metadata is missing or misaligned')
        if not size or size % self.group_size:
            raise ValueError('single-card reward batches must contain complete G-sized prompt-major groups')
        now = datetime.now(timezone.utc).isoformat()
        checked, values = [], []
        for i, raw_completion in enumerate(completions):
            completion = _decoded_completion(raw_completion)
            dataset_id = dataset_request_id[i]
            if dataset_id not in self.expected:
                raise ValueError(f'unknown dataset_request_id: {dataset_id}')
            registered = self.expected[dataset_id]
            pool_instance = columns.get('pool_instance_id', [None]*size)[i]
            if self.instances and self.instances.get(pool_instance)!=dataset_id:
                raise ValueError('Unknown/misbound training pool instance')
            for key in ('pseudo_scores', 'media_signature', 'clip_ids', 'clip_bounds', 'sampled_pts', 'videos'):
                if canonical_hash(columns[key][i]) != canonical_hash(registered[key]):
                    raise ValueError(f'rollout {dataset_id} has misbound {key}')
            if normalized_image_paths(columns['images'][i]) != normalized_image_paths(registered['images']):
                raise ValueError(f'rollout {dataset_id} has misbound images path/content identity')
            # Swift appends the generated assistant to the same two-message prompt.
            actual_messages = columns['messages'][i]
            if actual_messages[:2] != registered['messages'] or len(actual_messages) != 3:
                raise ValueError('generated conversation has a changed prompt or unexpected turn count')
            if _decoded_completion(actual_messages[-1]) != completion:
                raise ValueError('reward completion differs from the conversation completion')
            if not isinstance(request_id[i], str) or not request_id[i] or not isinstance(prompt_id[i], str) or not prompt_id[i]:
                raise ValueError('missing framework rollout UUID or prompt ID')
            parts = reward_parts(completion, pseudo_scores[i], self.weight)
            checked.append({
                'timestamp_utc': now, 'call_index': self.calls, 'row_in_call': i,
                'optimizer_step_before_update': getattr(trainer_state, 'global_step', None),
                'dataset_request_id': dataset_id, 'pool_instance_id': pool_instance, 'rollout_request_id': request_id[i],
                'prompt_id': prompt_id[i], 'media_signature': columns['media_signature'][i],
                'clip_ids': columns['clip_ids'][i], 'sampled_pts': columns['sampled_pts'][i],
                'pseudo_scores': pseudo_scores[i], 'completion': completion,
                'completion_sha256': hashlib.sha256(completion.encode()).hexdigest(),
                'rank_weight': self.weight, 'reward': asdict(parts),
                'finish_reason': columns.get('finish_reason', [None] * size)[i],
                'is_truncated': columns.get('is_truncated', [None] * size)[i],
                'response_token_ids': columns.get('response_token_ids', [None] * size)[i],
                'media_tensor_sha256': columns.get('media_tensor_sha256', [None] * size)[i],
                'dataset_sha256': self.dataset_sha256,
                'images_paths_normalized': normalized_image_paths(columns['images'][i]),
                'image_wrapper_rule': 'strict bytes=None/path wrapper or string; actual tensors checked separately',
            })
            values.append(parts.total)
        for start in range(0, size, self.group_size):
            group = checked[start:start + self.group_size]
            for key in ('dataset_request_id', 'pool_instance_id', 'prompt_id', 'media_signature', 'pseudo_scores', 'clip_ids', 'sampled_pts'):
                if any(canonical_hash(row[key]) != canonical_hash(group[0][key]) for row in group[1:]):
                    raise ValueError(f'G completions have different {key}')
            uuids = [row['rollout_request_id'] for row in group]
            if len(set(uuids)) != len(uuids):
                raise ValueError('duplicate framework completion request UUID')
            tensor_hashes = [row['media_tensor_sha256'] for row in group]
            if any(x is not None for x in tensor_hashes) and len(set(tensor_hashes)) != 1:
                raise ValueError('same-group processed-media tensor hashes differ')
            grouped_rewards = np.asarray(values[start:start + self.group_size])
            for g, row in enumerate(group):
                row.update(group_in_call=start // self.group_size, completion_in_group=g,
                           group_reward_mean=float(grouped_rewards.mean()),
                           group_reward_variance_population=float(grouped_rewards.var()),
                           group_reward_std_sample=float(grouped_rewards.std(ddof=1)))
        # All checks happen before reward return or ledger write. Use one locked append
        # per batch; framework retries and rejected optimizer steps remain observable.
        with self.ledger.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            for row in checked:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)
        self.calls += 1
        return values


orms['highlight_recurrence'] = HighlightRecurrenceReward
