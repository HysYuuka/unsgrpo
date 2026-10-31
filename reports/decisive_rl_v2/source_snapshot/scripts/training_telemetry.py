#!/usr/bin/env python3
"""ms-swift 4.6 callback: observed training consumption and optimizer receipts.

Register with --external_plugins THIS_FILE --callbacks highlight_telemetry.
Required env: HL_INPUT_AUDIT_ROWS, HL_TRAINING_LEDGER; optional
HL_TELEMETRY_SUMMARY. A real Swift/Qwen GPU run is required for integration
acceptance; CPU stub tests only validate the bookkeeping and hook mechanics.

Source contracts inspected: swift/callbacks/base.py, trainers/mixin.py,
rlhf_trainers/{grpo_trainer,rollout_mixin}.py at official tag v4.6.0.
No encode, collator, dataloader or prefetch call increments exposure counters.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from types import MethodType
from functools import wraps
from zoneinfo import ZoneInfo


def json_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sentinel_sample_indices(numel, max_samples=16):
    """Exact integers: float32 linspace can round a large n-1 up to n."""
    if type(numel) is not int or numel < 1 or type(max_samples) is not int or max_samples < 1:
        raise ValueError("sentinel dimensions must be positive integers")
    count = min(numel, max_samples)
    return [0] if count == 1 else [i * (numel - 1) // (count - 1) for i in range(count)]


def recompute_grpo_loss(current, old, reference, advantages, mask, *, beta, epsilon_low, epsilon_high):
    """CPU float32 reconstruction of Swift v4.6's plain GRPO objective."""
    import torch
    current, old, reference, advantages, mask = [torch.as_tensor(value, dtype=torch.float32, device="cpu")
                                               for value in (current, old, reference, advantages, mask)]
    if current.ndim != 2 or old.shape != current.shape or reference.shape != current.shape or mask.shape != current.shape:
        raise ValueError("GRPO numerical audit needs aligned [B, completion] vectors")
    if not all(bool(torch.isfinite(value).all()) for value in (current, old, reference, advantages, mask)):
        raise FloatingPointError("non-finite values in real GRPO loss audit")
    if not bool(torch.all((mask == 0) | (mask == 1))) or bool(torch.any(mask.sum(-1) <= 0)):
        raise ValueError("GRPO loss audit requires nonempty binary completion masks")
    advantages = torch.broadcast_to(advantages, current.shape)
    ratio = torch.exp(current - old)
    clipped = ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
    raw_difference = reference - current
    safe_difference = raw_difference.clamp(-20, 20)
    raw_k3 = torch.exp(safe_difference) - safe_difference - 1
    k3 = raw_k3.clamp(-10, 10)
    per_token = -torch.minimum(ratio * advantages, clipped * advantages) + beta * k3
    lengths = mask.sum(-1)
    loss = ((per_token * mask).sum(-1) / lengths.clamp(min=1)).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite reconstructed GRPO loss")
    return {"recomputed_loss": float(loss), "completion_lengths": lengths.to(torch.int64).tolist(),
            "kl_logratio_clamp_activated_tokens": int(((raw_difference < -20) | (raw_difference > 20)).sum()),
            "kl_value_clamp_activated_tokens": int(((raw_k3 < -10) | (raw_k3 > 10)).sum()),
            "ppo_ratio_clipped_tokens": int((ratio != clipped).sum()),
            "length_normalization": "per_sequence_masked_completion_mean_then_batch_mean"}


def require_no_active_teacher(trainer, *, samples=(), grpo_batch=None):
    """Swift's _has_teacher is a capability flag, not an active teacher signal."""
    explicit_check = getattr(trainer, "_has_teacher_explicit", None)
    explicit = bool(explicit_check()) if callable(explicit_check) else False
    prompts = sum(bool(getattr(sample, "teacher_prompt", None) or getattr(sample, "extra", {}).get("teacher_prompt")) for sample in samples)
    image_overrides = sum(getattr(sample, "teacher_images", None) is not None for sample in samples)
    teacher_messages = sum(getattr(sample, "teacher_messages", None) is not None for sample in samples)
    token_logps = grpo_batch is not None and getattr(grpo_batch, "teacher_per_token_logps", None) is not None
    record = {"capability_flag": bool(getattr(trainer, "_has_teacher", False)), "explicit_teacher": explicit,
              "n_samples": len(samples), "n_teacher_prompts": prompts, "n_teacher_image_overrides": image_overrides,
              "n_teacher_message_views": teacher_messages, "teacher_token_logps_present": bool(token_logps),
              "active": bool(explicit or prompts or image_overrides or teacher_messages or token_logps)}
    if record["active"]:
        raise ValueError("plain GRPO audit refuses an actual teacher input, explicit model, or teacher token log probabilities")
    return record


def _timestamp():
    now = datetime.now(timezone.utc)
    return {"timestamp_utc": now.isoformat(), "timestamp_hong_kong": now.astimezone(ZoneInfo("Asia/Hong_Kong")).isoformat()}


def tensor_media_hash(inputs):
    """Exact audit_swift tensor_hash contract, applied to consumed tensors."""
    import torch
    digest = hashlib.sha256()
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if key in inputs:
            value = inputs[key].detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_parameter_hash(named_parameters):
    import torch
    digest = hashlib.sha256()
    for name, value in named_parameters:
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_kind(name):
    name = name.lower()
    if any(part in name for part in ("merger", "multi_modal_projector", "mm_projector", "aligner")):
        return "aligner"
    if any(part in name for part in ("visual.", "vision_tower", "vision_model", "vit.")):
        return "vision"
    return "language_or_other"


def source_contract(trainer):
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for package in ("ms-swift", "transformers", "accelerate", "trl", "peft", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    methods = {}
    for name in ("training_step", "_prepare_inputs", "_prepare_batch_inputs", "_get_per_token_logps_and_entropies"):
        value = getattr(trainer, name, None)
        if value is None:
            continue
        try:
            source = inspect.getsource(value)
            methods[name] = {"path": inspect.getsourcefile(value),
                             "sha256": hashlib.sha256(source.encode()).hexdigest()}
        except (OSError, TypeError):
            methods[name] = {"source_introspection_available": False}
    return {"packages": packages, "trainer_class": type(trainer).__module__ + "." + type(trainer).__name__,
            "methods": methods, "contract_target": "official ms-swift v4.6.0"}


class AuditIndex:
    def __init__(self, rows):
        self.by_id = {}
        self.by_signature = {}
        for row in rows:
            if row.get("task") not in ("sft", "grpo"):
                continue
            rid = row["dataset_request_id"]
            key = (row["task"], rid)
            previous = self.by_id.get(key)
            if previous is not None and previous != row:
                # Repeated pool instances may differ only in a sampling instance ID.
                ignored = {"pool_instance_id", "input_file", "line_no", "processed_previews"}
                if {k: v for k, v in previous.items() if k not in ignored} != {k: v for k, v in row.items() if k not in ignored}:
                    raise ValueError(f"inconsistent audit rows for {key}")
            self.by_id[key] = row
            for media in {row["media_tensor_sha256"], row.get("media_tensor_bf16_sha256")} - {None}:
                signature = (row["task"], row["input_ids_sha256"], media)
                candidate = self.by_signature.setdefault(signature, rid)
                if candidate != rid:
                    raise ValueError("audit cannot distinguish different request IDs by tokens and actual media")
        if not self.by_id:
            raise ValueError("audit contains no training rows")

    @staticmethod
    def _ids(inputs):
        ids = inputs["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("telemetry requires actual training microbatch=1")
        ids = ids[0].detach().cpu()
        if inputs.get("attention_mask") is not None:
            mask = inputs["attention_mask"]
            if mask.ndim != 2 or mask.shape[0] != 1:
                raise ValueError("unsupported attention mask in telemetry")
            ids = ids[mask[0].detach().cpu().bool()]
        return ids.tolist()

    def match_sft(self, inputs):
        import torch
        ids = self._ids(inputs)
        media = tensor_media_hash(inputs)
        signature = ("sft", json_hash(ids), media)
        if signature not in self.by_signature:
            raise ValueError("actual SFT tokens/media do not match full-input audit")
        rid = self.by_signature[signature]
        audit = self.by_id[("sft", rid)]
        labels = inputs.get("labels")
        if labels is None or labels.shape != inputs["input_ids"].shape:
            raise ValueError("missing aligned SFT labels")
        values = labels[0].detach().cpu()
        if inputs.get("attention_mask") is not None:
            values = values[inputs["attention_mask"][0].detach().cpu().bool()]
        if hashlib.sha256(values.to(torch.int64).numpy().tobytes()).hexdigest() != audit["labels_sha256"]:
            raise ValueError("consumed SFT labels differ from assistant-only audited labels")
        loss_tokens = int((values != -100).sum())
        if loss_tokens != audit["supervised_tokens"]:
            raise ValueError("SFT supervised token count differs from audit")
        return {"dataset_request_id": rid, "task": "sft", "media_tensor_sha256": media,
                "clip_ids": audit["clip_ids"], "prompt_tokens": audit["prompt_tokens"],
                "encoded_tokens": len(ids), "loss_tokens": loss_tokens,
                "input_ids_sha256": json_hash(ids), "assistant_only_labels_verified": True,
                "pool_instance_id": None,
                "identity_note": "Actual consumed tokens/media identify the window; identical sampled pool instances cannot be distinguished after SFT collation."}

    def match_grpo(self, sample, batch, group_id):
        import torch
        rid = sample.extra.get("dataset_request_id")
        if ("grpo", rid) not in self.by_id:
            raise ValueError("GRPO sample lost its stable audited dataset_request_id")
        audit = self.by_id[("grpo", rid)]
        inputs, signals = batch["model_inputs"], batch["grpo_batch"]
        ids = self._ids(inputs)
        media = tensor_media_hash(inputs)
        if media not in {audit["media_tensor_sha256"], audit.get("media_tensor_bf16_sha256")}:
            raise ValueError("actual GRPO media differs from audited media")
        prompt_tokens = audit["prompt_tokens"]
        if json_hash(ids[:prompt_tokens]) != audit["input_ids_sha256"]:
            raise ValueError("actual GRPO prompt tokens differ from audited prompt")
        if sample.extra.get("media_signature") != audit["media_signature"] or sample.extra.get("clip_ids") != audit["clip_ids"]:
            raise ValueError("GRPO stable ID is bound to different media metadata or clip order")
        encoded = sample.encoded
        if encoded is None or list(encoded["input_ids"]) != ids:
            raise ValueError("collated GRPO token IDs do not match their bound generated sample")
        labels = torch.as_tensor(encoded["labels"], dtype=torch.int64)
        width = int(signals.logits_to_keep)
        mask = signals.completion_mask.detach().cpu().bool()
        if (labels.shape != (len(ids),) or not 0 < width <= len(ids)
                or mask.shape != (1, width)
                or not torch.equal(mask[0], labels[-width:] != -100)
                or bool(torch.any(labels[:prompt_tokens] != -100))
                or int(mask.sum()) != int((labels != -100).sum())):
            raise ValueError("GRPO completion mask includes prompt/padding or differs from real sample labels")
        if any(label != -100 and label != token for label, token in zip(labels.tolist(), ids)):
            raise ValueError("GRPO masked token targets differ from consumed token IDs")
        detached = {}
        for field in ("old_per_token_logps", "ref_per_token_logps", "advantages"):
            value = getattr(signals, field, None)
            # Advantages are filled after _prepare_batch_inputs; checked again at consumption.
            detached[field] = None if value is None else not value.requires_grad
            if value is not None and value.requires_grad:
                raise ValueError(f"GRPO {field} has an active gradient graph")
        return {"dataset_request_id": rid, "task": "grpo", "group_id": group_id,
                "pool_instance_id": sample.extra.get("pool_instance_id"),
                "rollout_request_id": sample.request_id, "prompt_id": sample.prompt_id,
                "media_tensor_sha256": media, "media_signature": audit["media_signature"],
                "clip_ids": audit["clip_ids"], "pseudo_scores_sha256": json_hash(sample.extra["pseudo_scores"]),
                "prompt_tokens": prompt_tokens, "encoded_tokens": len(ids),
                "loss_tokens": int(signals.completion_mask.sum().item()),
                "input_ids_sha256": json_hash(ids), "truncated": bool(sample.is_truncated),
                "completion_region_token_ids": ids[-width:], "completion_mask": mask[0].to(torch.int64).tolist(),
                "logits_to_keep": width, "prompt_loss_tokens": int((labels[:prompt_tokens] != -100).sum()),
                "padding_loss_tokens": 0,
                "generation_token_ids": getattr(sample, "response_token_ids", None),
                "framework_add_eos": getattr(sample, "add_eos", None),
                "finish_reason": getattr(sample, "finish_reason", None),
                "detached_signals_at_encode": detached}


class ExposureLedger:
    """Transactional counters: a consumed microbatch is committed by a real update."""
    def __init__(self, path, *, mode, group_size=4, summary_path=None):
        if mode not in {"sft", "grpo"} or group_size < 2:
            raise ValueError("invalid telemetry mode/group size")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", buffering=1)
        self.summary_path = Path(summary_path) if summary_path else self.path.with_suffix(".summary.json")
        self.mode, self.group_size = mode, group_size
        self.pending = []
        self.counts = Counter()
        self.repetitions = Counter()
        self.successful_request_order = []
        self.committed_completion_ids = set()
        self.status = "RUNNING"
        self.started = time.monotonic()
        self.stage_costs = {}

    def write(self, event, **data):
        if event in {"stage", "training_microbatch", "optimizer_call"} and "wall_seconds" in data:
            stage = data.get("stage", "optimizer")
            cost = self.stage_costs.setdefault(stage, {"calls": 0, "wall_seconds": 0.0,
                                                      "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                                                      "cuda_measured": False})
            cost["calls"] += 1
            cost["wall_seconds"] += data["wall_seconds"]
            cost["cuda_measured"] |= data.get("cuda_measured", False)
            for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                if data.get(key) is not None:
                    cost[key] = max(cost[key] or 0, data[key])
        self.stream.write(json.dumps({**_timestamp(), "event": event, **data}, allow_nan=False) + "\n")

    def consumed(self, records, **stats):
        if not records:
            raise ValueError("training_step completed without identified consumed inputs")
        self.pending.extend(records)
        self.counts["attempted_microbatches"] += 1
        self.counts["attempted_completion_rows" if self.mode == "grpo" else "attempted_prompt_rows"] += len(records)
        self.write("training_microbatch", status="BACKWARD_RETURNED_PENDING_OPTIMIZER", records=records, **stats)

    def commit(self, *, skipped, global_step, optimizer_evidence):
        if not self.pending:
            raise ValueError("optimizer event has no successfully returned training_step")
        rows = self.pending
        if skipped:
            self.counts["skipped_optimizer_steps"] += 1
            self.counts["discarded_completion_rows" if self.mode == "grpo" else "discarded_prompt_rows"] += len(rows)
            self.write("optimizer_update", status="SKIPPED", global_step=global_step,
                       n_consumed_rows=len(rows), optimizer=optimizer_evidence)
            self.pending = []
            self.save_summary()
            return
        if self.mode == "grpo":
            groups = OrderedDict()
            for row in rows:
                groups.setdefault(row["group_id"], []).append(row)
            for group in groups.values():
                if len(group) != self.group_size:
                    raise ValueError("optimizer update consumes a partial or reused GRPO group")
                for key in ("dataset_request_id", "prompt_id", "media_tensor_sha256", "pseudo_scores_sha256", "clip_ids"):
                    if any(row[key] != group[0][key] for row in group):
                        raise ValueError(f"consumed GRPO group has mismatched {key}")
                uuids = [row["rollout_request_id"] for row in group]
                if len(set(uuids)) != self.group_size or set(uuids) & self.committed_completion_ids:
                    raise ValueError("duplicate or reused consumed GRPO completion UUID")
            prompt_rows = [group[0] for group in groups.values()]
            self.committed_completion_ids.update(row["rollout_request_id"] for row in rows)
            self.counts["rollout_completions_consumed_in_successful_updates"] += len(rows)
        else:
            prompt_rows = rows
        ids = [row["dataset_request_id"] for row in prompt_rows]
        self.successful_request_order.extend(ids)
        self.repetitions.update(ids)
        self.counts["successful_optimizer_updates"] += 1
        self.counts["prompt_exposures_in_successful_updates"] += len(ids)
        self.counts["effective_loss_tokens"] += sum(row["loss_tokens"] for row in rows)
        self.counts["encoded_tokens_consumed"] += sum(row["encoded_tokens"] for row in rows)
        self.counts["prompt_tokens_per_exposure"] += sum(row["prompt_tokens"] for row in prompt_rows)
        self.write("optimizer_update", status="SUCCESSFUL", global_step=global_step,
                   dataset_request_ids_in_order=ids, n_consumed_rows=len(rows),
                   effective_loss_tokens=sum(row["loss_tokens"] for row in rows),
                   optimizer=optimizer_evidence, totals=dict(self.counts))
        self.pending = []
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.save_summary()

    def save_summary(self, **extra):
        elapsed = time.monotonic() - self.started
        cuda = any(cost["cuda_measured"] for cost in self.stage_costs.values())
        result = {"status": self.status, "mode": self.mode, "counts": dict(self.counts),
                  "unique_windows": len(self.repetitions), "window_repetitions": dict(self.repetitions),
                  "dataset_request_id_order": self.successful_request_order,
                  "pending_uncommitted_rows": len(self.pending), "wall_seconds": elapsed,
                  "single_gpu_allocation_wall_hours": elapsed / 3600 if cuda else None,
                  "stage_costs": self.stage_costs,
                  "exact_resume_verified": False, "integration_verified": False,
                  "note": "Observed runtime receipts do not alone satisfy full GPU technical acceptance.", **extra}
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.summary_path.with_name(self.summary_path.name + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temp.replace(self.summary_path)
        return result

    def close(self, status="TRAINER_ENDED_REQUIRES_ACCEPTANCE", **extra):
        self.status = status
        result = self.save_summary(**extra)
        self.stream.close()
        return result


class CudaStage:
    def __init__(self, trainer):
        import torch
        self.enabled = torch.cuda.is_available() and str(trainer.accelerator.device).startswith("cuda")
        self.started = None

    def begin(self):
        import torch
        if self.enabled:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started = time.monotonic()

    def end(self):
        import torch
        if self.enabled:
            torch.cuda.synchronize()
        return {"wall_seconds": time.monotonic() - self.started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.enabled else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.enabled else None,
                "cuda_measured": self.enabled}


class HighlightTelemetry:
    """Mixin combined with Swift's TrainerCallback only at plugin registration."""
    def __init__(self, args, trainer):
        self.args, self.trainer = args, trainer
        self.ledger = None
        self.in_step = False
        self.bindings = {}
        self.generation_index = 0
        self.prepared = None
        self.optimizer_evidence = None

    def _fail(self, error):
        if self.ledger is not None and not self.ledger.stream.closed:
            self.ledger.status = "FAILED"
            self.ledger.write("failure", error=f"{type(error).__name__}: {error}")
            self.ledger.save_summary(error=f"{type(error).__name__}: {error}")

    def on_train_begin(self, args, state, control, **kwargs):
        import torch
        trainer = self.trainer
        if (getattr(args, "world_size", 1) != 1 or getattr(args, "per_device_train_batch_size", None) != 1
                or getattr(trainer.template, "padding_free", False) or getattr(args, "packing", False)
                or getattr(args, "async_generate", False) or getattr(args, "num_iterations", 1) != 1):
            raise ValueError("telemetry supports only audited single-GPU microbatch=1, unpacked synchronous num_iterations=1")
        audit_path = Path(os.environ["HL_INPUT_AUDIT_ROWS"])
        rows = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
        self.index = AuditIndex(rows)
        self.mode = "grpo" if hasattr(trainer, "_prepare_batch_inputs") and hasattr(trainer, "num_generations") else "sft"
        self.group_size = int(getattr(trainer, "num_generations", 4))
        self.ledger = ExposureLedger(os.environ["HL_TRAINING_LEDGER"], mode=self.mode,
                                     group_size=self.group_size, summary_path=os.environ.get("HL_TELEMETRY_SUMMARY"))
        self.trainable = [(name, p) for name, p in trainer.model.named_parameters() if p.requires_grad]
        self.frozen = [(name, p) for name, p in trainer.model.named_parameters() if not p.requires_grad]
        if not self.trainable or any("lora_" not in name.lower() or _module_kind(name) != "language_or_other" for name, _ in self.trainable):
            raise ValueError("only new language-side LoRA parameters may be trainable")
        self.initial_parameter_hash = tensor_parameter_hash(self.trainable)
        self.last_parameter_hash = self.initial_parameter_hash
        self.frozen_sentinel = self._frozen_sentinel()
        classification = {kind: sum(_module_kind(name) == kind for name, _ in self.frozen)
                          for kind in ("vision", "aligner", "language_or_other")}
        self.ledger.write("training_begin", mode=self.mode,
                          runtime_source_contract=source_contract(trainer),
                          audit_rows_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                          trainable_parameters=[{"name": name, "shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel()}
                                                for name, p in self.trainable],
                          initial_trainable_sha256=self.initial_parameter_hash,
                          non_adapter_requires_grad_false=True, frozen_parameter_counts=classification,
                          frozen_sentinel=self.frozen_sentinel, frozen_sentinel_is_full_weight_hash=False,
                          telemetry_synchronizes_cuda=True, exact_resume_verified=False)
        self._install_hooks()
        self.ledger.save_summary()
        return control

    def _frozen_sentinel(self):
        import torch
        result = {}
        for kind in ("vision", "aligner", "language_or_other"):
            values = [(n, p) for n, p in self.frozen if _module_kind(n) == kind and p.numel()]
            selected = sorted(set([0, len(values) // 2, len(values) - 1])) if values else []
            result[kind] = []
            for offset in selected:
                name, value = values[offset]
                ids = torch.tensor(sentinel_sample_indices(value.numel()), dtype=torch.long, device=value.device)
                samples = value.detach().reshape(-1)[ids].cpu().contiguous()
                result[kind].append({"name": name, "shape": list(value.shape), "sample_indices": ids.cpu().tolist(),
                                     "sample_sha256": hashlib.sha256(samples.view(torch.uint8).numpy().tobytes()).hexdigest()})
        return result

    def _gradient_evidence(self):
        import torch
        count, square = 0, 0.0
        nonzero = []
        for name, parameter in self.trainable:
            gradient = parameter.grad
            if gradient is None:
                continue
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"non-finite gradient: {name}")
            count += 1
            norm2 = float(gradient.detach().float().square().sum())
            square += norm2
            if norm2 > 0:
                nonzero.append(name)
        if not count:
            raise ValueError("optimizer has no trainable LoRA gradients")
        return {"finite": True, "n_gradient_tensors": count, "l2_norm": square ** 0.5,
                "n_nonzero_gradient_tensors": len(nonzero), "nonzero_gradient_names": nonzero}

    def _install_hooks(self):
        import torch
        trainer = self.trainer
        original_prepare = trainer._prepare_inputs
        original_step = trainer.training_step
        original_optimizer_step = trainer.optimizer.step
        self.forward_stage = CudaStage(trainer)

        def prepare(_trainer, inputs, *pos, **kw):
            prepared = original_prepare(inputs, *pos, **kw)
            if not self.in_step:
                return prepared
            if self.mode == "sft":
                self.prepared = [self.index.match_sft(prepared)]
            else:
                if id(prepared) not in self.bindings:
                    raise ValueError("consumed GRPO microbatch has no binding to generated samples")
                self.prepared = self.bindings.pop(id(prepared))
                signals = prepared["grpo_batch"]
                for field in ("old_per_token_logps", "ref_per_token_logps", "advantages"):
                    value = getattr(signals, field, None)
                    if value is None or value.requires_grad or not bool(torch.isfinite(value).all()):
                        raise ValueError(f"consumed GRPO {field} must exist, be finite and detached")
                rewards = getattr(trainer, "_rewards_per_func", None)
                if rewards is None or getattr(rewards, "requires_grad", False) or not bool(torch.isfinite(rewards).all()):
                    raise ValueError("GRPO reward is missing, non-finite or retains a gradient graph")
                for row in self.prepared:
                    row["old_ref_advantage_detached_at_consumption"] = True
                    row["requires_grad_at_consumption"] = {field: bool(getattr(signals, field).requires_grad)
                                                           for field in ("old_per_token_logps", "ref_per_token_logps", "advantages")}
                    row["requires_grad_at_consumption"]["rewards"] = bool(rewards.requires_grad)
                    eos = getattr(trainer.template.tokenizer, "eos_token_id", None)
                    row["tokenizer_eos_token_ids"] = eos if isinstance(eos, list) else [eos] if eos is not None else []
                    row["eos_loss_tokens"] = sum(mask for token, mask in zip(row["completion_region_token_ids"], row["completion_mask"])
                                                 if token in row["tokenizer_eos_token_ids"])
            self.forward_stage.begin()  # Isolate forward/backward from rollout and old/ref preparation.
            return prepared

        def training_step(_trainer, model, inputs, *pos, **kw):
            self.in_step, self.prepared = True, None
            started = time.monotonic()
            try:
                loss = original_step(model, inputs, *pos, **kw)
                stats = self.forward_stage.end()
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("non-finite returned training loss")
                self.ledger.consumed(self.prepared, global_step_before=int(trainer.state.global_step),
                                     returned_loss=float(loss.detach()), stage="forward_backward", **stats,
                                     training_step_total_wall_seconds=time.monotonic() - started)
                return loss
            except BaseException as error:
                self._fail(error)
                raise
            finally:
                self.in_step = False

        @wraps(original_optimizer_step)
        def optimizer_step(_optimizer, *pos, **kw):
            if not self.ledger.pending:
                raise ValueError("optimizer.step observed without completed microbatches")
            stage = CudaStage(trainer)
            try:
                gradient = self._gradient_evidence()
                before = tensor_parameter_hash(self.trainable)
                learning_rates = [float(group["lr"]) for group in trainer.optimizer.param_groups]
                if not learning_rates or any(not math.isfinite(value) or value < 0 for value in learning_rates):
                    raise ValueError("optimizer has invalid actual learning rates")
                stage.begin()
                result = original_optimizer_step(*pos, **kw)
                stats = stage.end()
                after = tensor_parameter_hash(self.trainable)
                skipped = bool(getattr(trainer.accelerator, "optimizer_step_was_skipped", False))
                self.optimizer_evidence = {"call_returned": True, "skipped": skipped, "gradient": gradient,
                                           "learning_rates_before_step": learning_rates,
                                           "trainable_sha256_before": before, "trainable_sha256_after": after,
                                           "parameter_changed": before != after, **stats}
                self.ledger.write("optimizer_call", stage="optimizer", **self.optimizer_evidence)
                self.last_parameter_hash = after
                return result
            except BaseException as error:
                self._fail(error)
                raise

        trainer._prepare_inputs = MethodType(prepare, trainer)
        trainer.training_step = MethodType(training_step, trainer)
        trainer.optimizer.step = MethodType(optimizer_step, trainer.optimizer)
        if self.mode == "grpo":
            self._install_grpo_hooks()

    def _install_grpo_hooks(self):
        import torch
        trainer = self.trainer
        original_batch = trainer._prepare_batch_inputs

        def prepare_batch(_trainer, samples, *pos, **kw):
            teacher_state = require_no_active_teacher(trainer, samples=samples)
            split = trainer.split_by_mini_batches
            captured = []
            def capture(_trainer, sample_rows, *args, **kwargs):
                chunks = split(sample_rows, *args, **kwargs)
                captured.append(chunks)
                return chunks
            trainer.split_by_mini_batches = MethodType(capture, trainer)
            stage = CudaStage(trainer)
            stage.begin()
            try:
                batches = original_batch(samples, *pos, **kw)
            finally:
                trainer.split_by_mini_batches = split
            stats = stage.end()
            if not self.in_step:
                return batches
            if self.bindings:
                raise ValueError("new rollout arrived before all previous generated microbatches were consumed")
            if len(captured) != 1 or len(captured[0]) != len(batches):
                raise ValueError("cannot verify actual framework sample-to-microbatch split")
            if not samples or len(samples) % self.group_size:
                raise ValueError("rollout does not contain complete G-sized groups")
            group_by_uuid = {}
            for start in range(0, len(samples), self.group_size):
                group = samples[start:start + self.group_size]
                baseline = {key: group[0].extra[key] for key in ("dataset_request_id", "media_signature", "pseudo_scores", "clip_ids")}
                for sample in group:
                    if any(sample.extra[key] != value for key, value in baseline.items()):
                        raise ValueError("generated G-sized group has mismatched input/q/order")
                    if sample.request_id in group_by_uuid:
                        raise ValueError("generated completion UUID is duplicated")
                    group_by_uuid[sample.request_id] = f"rollout_{self.generation_index}:group_{start // self.group_size}"
            for chunk, batch in zip(captured[0], batches):
                if len(chunk) != 1:
                    raise ValueError("actual GRPO encoded microbatch must contain one completion")
                sample = chunk[0]
                actual_teacher_state = require_no_active_teacher(trainer, samples=chunk, grpo_batch=batch["grpo_batch"])
                record = self.index.match_grpo(sample, batch, group_by_uuid[sample.request_id])
                record["teacher_activity"] = actual_teacher_state
                self.bindings[id(batch)] = [record]
            self.ledger.counts["rollout_completions_generated"] += len(samples)
            self.ledger.write("stage", stage="encode_old_ref_logps", rollout_index=self.generation_index,
                              n_generated_samples=len(samples), status="COMPLETED", teacher_activity=teacher_state, **stats)
            self.generation_index += 1
            return batches

        trainer._prepare_batch_inputs = MethodType(prepare_batch, trainer)
        for method_name, phase in (("_generate_completions", "generation"), ("_score_completions", "reward")):
            original = getattr(trainer, method_name)
            def measured(_trainer, *pos, _original=original, _phase=phase, **kw):
                stage = CudaStage(trainer)
                stage.begin()
                result = _original(*pos, **kw)
                stats = stage.end()
                if self.in_step:
                    self.ledger.write("stage", stage=_phase, status="COMPLETED", rollout_index=self.generation_index, **stats)
                return result
            setattr(trainer, method_name, MethodType(measured, trainer))
        original_logps = trainer._get_per_token_logps_and_entropies
        def logps(_trainer, *pos, **kw):
            grad_enabled = torch.is_grad_enabled()
            result = original_logps(*pos, **kw)
            log_probs = result[0]
            if self.in_step:
                if not bool(torch.isfinite(log_probs).all()) or bool(log_probs.requires_grad) != grad_enabled:
                    raise ValueError("actual policy log probabilities have wrong gradient state or non-finite values")
                self.ledger.write("policy_logps", role="current" if grad_enabled else "old_or_reference",
                                  requires_grad=bool(log_probs.requires_grad), grad_enabled=grad_enabled,
                                  n_tokens=log_probs.numel(), finite=True)
                if grad_enabled:
                    self._latest_current_logps = log_probs.detach().float().cpu()
            return result
        trainer._get_per_token_logps_and_entropies = MethodType(logps, trainer)
        if hasattr(trainer, "_compute_loss_and_metrics"):
            original_loss = trainer._compute_loss_and_metrics
            def audited_loss(_trainer, model, model_inputs, grpo_batch, *pos, **kw):
                if not self.in_step:
                    return original_loss(model, model_inputs, grpo_batch, *pos, **kw)
                teacher_state = require_no_active_teacher(trainer, grpo_batch=grpo_batch)
                if any(row.get("teacher_activity", {}).get("active") for row in self.prepared or []):
                    raise ValueError("consumed GRPO batch has active teacher inputs")
                if (getattr(trainer, "loss_type", None) != "grpo"
                        or getattr(trainer, "importance_sampling_level", None) != "token"
                        or getattr(trainer, "kl_in_reward", False)
                        or getattr(trainer, "top_entropy_quantile", 1.0) != 1.0
                        or getattr(trainer, "rollout_importance_sampling_mode", None) is not None
                        or getattr(trainer, "off_policy_sequence_mask_delta", None) is not None
                        or getattr(trainer.args, "delta", None) is not None
                        or getattr(trainer, "overlong_filter", False)
                        or getattr(trainer, "sdar_loss_coef", 0) != 0
                        or getattr(trainer, "chord_sft_iterator", None) is not None):
                    raise ValueError("loss audit refuses unregistered entropy/IS/offpolicy/auxiliary objective branches")
                self._latest_current_logps = None
                result = original_loss(model, model_inputs, grpo_batch, *pos, **kw)
                if self._latest_current_logps is None:
                    raise ValueError("GRPO loss audit did not observe the real current-policy forward")
                values = {"current_logps": self._latest_current_logps.tolist(),
                          "old_logps": grpo_batch.old_per_token_logps.detach().float().cpu().tolist(),
                          "reference_logps": grpo_batch.ref_per_token_logps.detach().float().cpu().tolist(),
                          "advantages": grpo_batch.advantages.detach().float().cpu().tolist(),
                          "completion_mask": grpo_batch.completion_mask.detach().cpu().to(torch.int64).tolist()}
                config = {"beta": float(trainer.beta), "epsilon_low": float(trainer.epsilon_low),
                          "epsilon_high": float(trainer.epsilon_high)}
                audit = recompute_grpo_loss(values["current_logps"], values["old_logps"], values["reference_logps"],
                                             values["advantages"], values["completion_mask"], **config)
                actual = float(result[0].detach())
                difference = abs(actual - audit["recomputed_loss"])
                tolerance = 1e-5 + 1e-4 * abs(audit["recomputed_loss"])
                self.ledger.write("grpo_loss_numeric", **values, **config, **audit, actual_framework_loss=actual,
                                  absolute_error=difference, absolute_tolerance=1e-5, relative_tolerance=1e-4,
                                  within_tolerance=difference <= tolerance,
                                  teacher_activity=teacher_state,
                                  entropy_logging_only=bool(getattr(trainer, "compute_entropy", False)),
                                  consumed_dataset_request_ids=[row["dataset_request_id"] for row in self.prepared or []],
                                  rollout_request_ids=[row["rollout_request_id"] for row in self.prepared or []])
                if not math.isfinite(actual) or difference > tolerance:
                    raise ValueError("real GRPO loss disagrees with token/mask/length-normalized numerical reconstruction")
                return result
            trainer._compute_loss_and_metrics = MethodType(audited_loss, trainer)

    def on_step_end(self, args, state, control, **kwargs):
        try:
            if self.optimizer_evidence is None:
                raise ValueError("step_end has no observed completed optimizer.step")
            evidence = self.optimizer_evidence
            self.ledger.commit(skipped=evidence["skipped"], global_step=int(state.global_step), optimizer_evidence=evidence)
            self.optimizer_evidence = None
        except BaseException as error:
            self._fail(error)
            raise
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.ledger is None or self.ledger.stream.closed:
            return control
        final_frozen = self._frozen_sentinel()
        stable = self.frozen_sentinel == final_frozen
        status = "TRAINER_ENDED_REQUIRES_ACCEPTANCE" if stable and not self.ledger.pending and not self.bindings else "FAILED"
        self.ledger.write("training_end", status=status, frozen_sentinel_unchanged=stable,
                          final_frozen_sentinel=final_frozen,
                          final_trainable_sha256=tensor_parameter_hash(self.trainable),
                          trainable_changed_since_begin=self.initial_parameter_hash != tensor_parameter_hash(self.trainable))
        self.ledger.close(status, frozen_sentinel_unchanged=stable,
                          trainable_changed_since_begin=self.initial_parameter_hash != self.last_parameter_hash)
        if status == "FAILED":
            raise ValueError("training ended with frozen-weight drift or uncommitted/unconsumed inputs")
        return control


def register_plugin():
    from swift.callbacks import TrainerCallback, callbacks_map
    class RegisteredHighlightTelemetry(HighlightTelemetry, TrainerCallback):
        pass
    callbacks_map["highlight_telemetry"] = RegisteredHighlightTelemetry


if os.environ.get("HL_TRAINING_LEDGER"):
    register_plugin()
