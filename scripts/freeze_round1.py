#!/usr/bin/env python3
"""Recompute technical acceptance from artifacts, then freeze the six-run plan.

Missing/inconsistent evidence writes INCOMPLETE and exits 2. No --passed flags
exist. This script reads training artifacts and label-boundary metadata only;
it never opens human annotation arrays. Technical smoke models are not the
formal common S0; the formal S0 must later be registered independently.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from unsgrpo.core import parse_scores, reward_parts
from unsgrpo.data import audit_split, audit_window_record, canonical_hash

REQUIRED_CHECKS = (
    "generation_reward_logps_backward_optimizer_save_reload", "same_group_media_q_id",
    "assistant_only_sft_no_grpo_answers", "strict_json_equal_reward_truncation",
    "finite_lora_gradients_parameter_update_frozen_backbone", "s0_bf16_tensor_merge_common_endpoint",
    "fixed_s0_reference_after_update", "old_ref_reward_advantage_detached",
    "completion_only_loss_eos_lengths_std_sampling", "stage_memory_time_recorded",
    "formal_rollout_concurrency_verified",
)
_HASH_CACHE = {}


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def numeric_vector(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and len(a) > 0 and np.isfinite(a).all(), "missing/non-finite probability vector")
    return a


def compare_logps(first, second, mean_tol, max_tol):
    a, b = numeric_vector(first["token_logps"]), numeric_vector(second["token_logps"])
    require(a.shape == b.shape and first["answer_token_ids"] == second["answer_token_ids"], "fixed answer token identities differ")
    delta = np.abs(a - b)
    require(delta.mean() <= mean_tol and delta.max() <= max_tol, "recomputed fixed-answer logp difference exceeds tolerance")
    return {"n_tokens": len(a), "mean_abs": float(delta.mean()), "max_abs": float(delta.max()),
            "mean_tolerance": mean_tol, "max_tolerance": max_tol}


def command_flags(command):
    result = {}
    for i, word in enumerate(command):
        if isinstance(word, str) and word.startswith("--"):
            values = []
            for value in command[i + 1:]:
                if str(value).startswith("--"):
                    break
                values.append(str(value))
            result[word] = values
    return result


def one(flags, name):
    values = flags.get(name, [])
    require(len(values) == 1, f"missing/ambiguous runtime argument {name}")
    return values[0]


def file_hash_map(root, expected, evidence_files):
    root = Path(root)
    require(root.is_dir() and bool(expected), f"missing model/checkpoint file hash table: {root}")
    for relative, info in expected.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), "model hash table escapes its root")
        require(path.is_file() and path.stat().st_size == info["bytes"] and sha(path) == info["sha256"],
                f"model/checkpoint file changed: {path}")
        evidence_files.add(path)


def normalized_adapter_name(name):
    return re.sub(r"(\.lora_[AB])\.[^.]+(\.weight)$", r"\1\2", name)


def checkpoint_matches_last_update(run, checkpoint, begin, last_optimizer, evidence_files):
    """Reconstruct telemetry's ordered trainable hash from actual saved tensors."""
    import torch
    from safetensors.torch import load_file
    weights = checkpoint / "adapter_model.safetensors"
    require(weights.is_file(), "technical endpoint has no adapter_model.safetensors")
    tensors = load_file(str(weights), device="cpu")
    digest = hashlib.sha256()
    for item in begin["trainable_parameters"]:
        name = item["name"]
        normalized = normalized_adapter_name(name)
        matches = [key for key in tensors if key == normalized or key.endswith("." + normalized)
                   or normalized.endswith("." + key)]
        require(len(matches) == 1, f"saved adapter cannot uniquely recover trainable tensor {name}")
        value = tensors[matches[0]].contiguous()
        require(list(value.shape) == item["shape"] and str(value.dtype) == item["dtype"], "saved adapter dtype/shape differs from runtime")
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    require(digest.hexdigest() == last_optimizer["trainable_sha256_after"], "saved adapter differs from last actual optimizer parameters")
    for path in checkpoint.iterdir():
        if path.is_file():
            evidence_files.add(path.resolve())
    return {"checkpoint": str(checkpoint), "saved_trainable_sha256": digest.hexdigest(),
            "exact_optimizer_resume_claimed": False}


def validate_run(root, mode, expected_updates, prompts_per_update, evidence_files, *, allow_formal_zero_signal=False):
    root = Path(root).resolve()
    receipt_path, summary_path, ledger_path = (root / name for name in ("receipt.json", "training_summary.json", "training_ledger.jsonl"))
    receipt, summary, ledger = read_json(receipt_path), read_json(summary_path), read_rows(ledger_path)
    require(not allow_formal_zero_signal or (mode == "grpo" and receipt.get("mode") == "formal"
                and expected_updates == 256 and prompts_per_update == 4),
            "zero-signal negative outcomes are permitted only for the complete registered formal GRPO budget")
    evidence_files.update({receipt_path, summary_path, ledger_path})
    require(receipt.get("exit_code") == 0 and receipt.get("status") in {"TRAINER_EXITED_SUCCESS_REQUIRES_AUDIT", "COMPLETED"}, "trainer did not exit successfully")
    require(summary.get("mode") == mode and summary.get("status") == "TRAINER_ENDED_REQUIRES_ACCEPTANCE", "telemetry did not finish cleanly")
    require(summary.get("pending_uncommitted_rows") == 0, "telemetry has uncommitted consumed samples")
    begin = [event for event in ledger if event["event"] == "training_begin"]
    end = [event for event in ledger if event["event"] == "training_end"]
    require(len(begin) == len(end) == 1 and not any(event["event"] == "failure" for event in ledger), "missing/duplicate begin/end or logged training failure")
    begin, end = begin[0], end[0]
    versions = begin["runtime_source_contract"]["packages"]
    require(versions.get("ms-swift") == "4.6.0", "runtime is not ms-swift 4.6.0")
    require(begin.get("initial_trainable_sha256") and begin.get("trainable_parameters"), "missing real trainable parameter inventory")
    require(all("lora_" in row["name"] and not any(tag in row["name"].lower() for tag in ("visual.", "vision_tower", "merger", "aligner"))
                for row in begin["trainable_parameters"]), "trainable parameters extend outside language LoRA")
    require(begin.get("non_adapter_requires_grad_false") is True, "frozen requires_grad inventory missing")
    require(begin.get("frozen_sentinel") == end.get("final_frozen_sentinel") and end.get("frozen_sentinel_unchanged") is True,
            "frozen sampled weights drifted or sentinel evidence absent")
    require(all(begin["frozen_parameter_counts"].get(kind, 0) > 0 for kind in ("vision", "aligner", "language_or_other")), "real Qwen vision/aligner/language frozen inventories missing")
    flags = command_flags(receipt["command"])
    if expected_updates is None:
        expected_updates = int(one(flags, "--max_steps"))
        require(expected_updates >= 1, "formal concurrency probe needs at least one actual optimizer update")
    require(int(one(flags, "--max_steps")) == expected_updates, "technical run update budget differs")
    require(one(flags, "--per_device_train_batch_size") == "1" and one(flags, "--packing") == "false"
            and one(flags, "--padding_free") == "false", "runtime batching differs from audited telemetry path")
    for key, value in {"--lora_rank": "16", "--lora_alpha": "32", "--lora_dropout": "0", "--freeze_vit": "true",
                       "--freeze_aligner": "true", "--freeze_llm": "false", "--bf16": "true", "--attn_impl": "sdpa",
                       "--max_length": "8192", "--lr_scheduler_type": "cosine", "--warmup_ratio": "0.05"}.items():
        require(one(flags, key) == value, f"actual LoRA/input/optimizer configuration differs: {key}")
    data_path = Path(one(flags, "--dataset")).resolve()
    data_rows = read_rows(data_path)
    evidence_files.add(data_path)
    require(data_rows and all(row["task"] == mode and row["split"] == "train" for row in data_rows), "technical data is not a training-only input pool")
    for row in data_rows:
        audit_window_record(row)
    registered = {row["dataset_request_id"]: row for row in data_rows}
    gas = int(one(flags, "--gradient_accumulation_steps"))
    group_size = 4 if mode == "grpo" else 1
    require(gas == prompts_per_update * group_size, "actual accumulation does not match technical prompt budget")
    if mode == "grpo":
        for key, value in {"--num_generations": "4", "--num_iterations": "1", "--temperature": "1", "--top_p": "1",
                           "--top_k": "0", "--repetition_penalty": "1", "--max_completion_length": "128",
                           "--loss_type": "grpo", "--scale_rewards": "group", "--importance_sampling_level": "token",
                           "--dynamic_sample": "false", "--overlong_filter": "false", "--beta": "0.04",
                           "--epsilon": "0.2", "--kl_in_reward": "false", "--advantage_estimator": "grpo",
                           "--reward_funcs": "highlight_recurrence"}.items():
            require(one(flags, key) == value, f"runtime sampling/loss differs: {key}")
        require(int(one(flags, "--generation_batch_size")) == prompts_per_update * 4, "rollout concurrency differs")
    pending, all_rows, ids, updates, optimizer_calls = [], [], [], [], []
    completion_ids = set()
    loss_tokens = encoded_tokens = 0
    for event in ledger:
        if event["event"] == "training_microbatch":
            require(event.get("status") == "BACKWARD_RETURNED_PENDING_OPTIMIZER" and math.isfinite(event["returned_loss"]), "invalid backward receipt")
            require(len(event["records"]) == 1, "unexpected actual microbatch size")
            pending.extend(event["records"])
        elif event["event"] == "optimizer_call":
            optimizer_calls.append(event)
            grad = event["gradient"]
            require(event.get("call_returned") is True and event.get("skipped") is False, "optimizer call skipped or failed")
            require(grad.get("finite") is True and grad["n_gradient_tensors"] > 0 and math.isfinite(grad["l2_norm"]), "missing/non-finite real LoRA gradients")
            require(event["parameter_changed"] == (event["trainable_sha256_before"] != event["trainable_sha256_after"]), "parameter change flag contradicts hashes")
        elif event["event"] == "optimizer_update":
            require(event.get("status") == "SUCCESSFUL" and len(pending) == gas, "skipped/partial update or exposure budget mismatch")
            require(any(all(call.get(key) == value for key, value in event["optimizer"].items()) for call in optimizer_calls),
                    "update has no matching actual optimizer call receipt")
            groups = OrderedDict()
            for row in pending:
                rid = row["dataset_request_id"]
                require(rid in registered and row["clip_ids"] == registered[rid]["clip_ids"], "consumed window ID/order differs from registered input")
                require(row["loss_tokens"] > 0 and row["encoded_tokens"] > row["prompt_tokens"] > 0, "invalid consumed token counts")
                if mode == "grpo":
                    require(row["rollout_request_id"] not in completion_ids, "rollout completion was reused")
                    completion_ids.add(row["rollout_request_id"])
                    require(row["pseudo_scores_sha256"] == hashlib.sha256(json.dumps(registered[rid]["pseudo_scores"], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(), "consumed q differs from registered q")
                    groups.setdefault(row["group_id"], []).append(row)
                else:
                    groups[str(len(groups))] = [row]
            for group in groups.values():
                require(len(group) == group_size, "consumed update has incomplete G group")
                for key in ("dataset_request_id", "clip_ids", "media_tensor_sha256"):
                    require(all(row[key] == group[0][key] for row in group), f"group {key} mismatch")
            order = [group[0]["dataset_request_id"] for group in groups.values()]
            require(order == event["dataset_request_ids_in_order"] and len(order) == prompts_per_update, "optimizer exposure order differs from consumed samples")
            all_rows.extend(pending); ids.extend(order); updates.append(event)
            loss_tokens += sum(row["loss_tokens"] for row in pending)
            encoded_tokens += sum(row["encoded_tokens"] for row in pending)
            pending = []
    require(not pending and len(updates) == len(optimizer_calls) == expected_updates, "successful update count differs from planned technical budget")
    parameter_changed_updates = sum(event["parameter_changed"] for event in optimizer_calls)
    has_nonzero_update = any(event["parameter_changed"] and event["gradient"]["n_nonzero_gradient_tensors"] > 0 for event in optimizer_calls)
    require(has_nonzero_update or allow_formal_zero_signal, "no actual nonzero-gradient LoRA parameter update observed")
    negative_outcome = None
    recomputed = {"successful_optimizer_updates": len(updates), "prompt_exposures_in_successful_updates": len(ids),
                  "effective_loss_tokens": loss_tokens, "encoded_tokens_consumed": encoded_tokens}
    if mode == "grpo":
        recomputed["rollout_completions_consumed_in_successful_updates"] = len(all_rows)
        generated = sum(event["n_generated_samples"] for event in ledger if event["event"] == "stage"
                        and event.get("stage") == "encode_old_ref_logps" and event.get("status") == "COMPLETED")
        require(generated == len(all_rows) == summary["counts"].get("rollout_completions_generated"),
                "actual generated and successfully consumed completion counts differ")
        recomputed["rollout_completions_generated"] = generated
    require(all(summary["counts"].get(key) == value for key, value in recomputed.items()), "summary counters contradict raw transactional ledger")
    require(summary["dataset_request_id_order"] == ids and summary["window_repetitions"] == dict(Counter(ids)), "summary sampling order/repetitions contradict ledger")
    require(summary["unique_windows"] == len(set(ids)), "unique-window count contradicts ledger")
    costs = summary["stage_costs"]
    required_stages = {"forward_backward", "optimizer"} | ({"generation", "reward", "encode_old_ref_logps"} if mode == "grpo" else set())
    require(required_stages <= set(costs), "stage timing/memory evidence missing")
    for name in required_stages:
        value = costs[name]
        require(value.get("cuda_measured") is True and value["wall_seconds"] > 0 and value["peak_allocated_bytes"] > 0
                and value["peak_reserved_bytes"] >= value["peak_allocated_bytes"], f"stage lacks real GPU timing/memory: {name}")
    if mode == "grpo":
        logps = [event for event in ledger if event["event"] == "policy_logps"]
        require(sum(event.get("role") == "current" and event.get("requires_grad") is True and event.get("finite") is True for event in logps) >= len(all_rows), "current policy logp gradient evidence missing")
        require(sum(event.get("role") == "old_or_reference" and event.get("requires_grad") is False and event.get("finite") is True for event in logps) >= 2 * len(all_rows), "old/ref detached probability evidence missing")
        for row in all_rows:
            require(row.get("old_ref_advantage_detached_at_consumption") is True, "consumption-time detached-signal evidence missing")
            teacher = row["teacher_activity"]
            require(teacher["active"] is False and teacher["explicit_teacher"] is False
                    and teacher["teacher_token_logps_present"] is False
                    and teacher["n_teacher_prompts"] == teacher["n_teacher_image_overrides"] == teacher["n_teacher_message_views"] == 0,
                    "plain GRPO consumption used actual teacher inputs/logps")
            gradient_flags = row["requires_grad_at_consumption"]
            require(set(gradient_flags) == {"old_per_token_logps", "ref_per_token_logps", "advantages", "rewards"}
                    and all(value is False for value in gradient_flags.values()), "old/ref/reward/advantage gradient flags differ")
            mask, tokens = row["completion_mask"], row["completion_region_token_ids"]
            require(len(mask) == len(tokens) == row["logits_to_keep"] and set(mask) <= {0, 1}
                    and sum(mask) == row["loss_tokens"] and row["prompt_loss_tokens"] == row["padding_loss_tokens"] == 0,
                    "completion mask/token/prompt/padding evidence is inconsistent")
            require(row["logits_to_keep"] <= row["encoded_tokens"] - row["prompt_tokens"], "completion region reaches into prompt")
            require(row.get("tokenizer_eos_token_ids") and row["eos_loss_tokens"] == sum(m for t, m in zip(tokens, mask) if t in row["tokenizer_eos_token_ids"]), "EOS loss convention not numerically recorded")
        numeric_events = [event for event in ledger if event["event"] == "grpo_loss_numeric"]
        require(len(numeric_events) == len(all_rows), "actual GRPO length-normalized loss numerical evidence missing")
        numeric_ids = []
        numeric_by_uuid = {}
        for event in numeric_events:
            teacher = event["teacher_activity"]
            require(teacher["active"] is False and teacher["explicit_teacher"] is False
                    and teacher["teacher_token_logps_present"] is False, "numerical GRPO objective contains an active teacher")
            require(event["beta"] == 0.04 and event["epsilon_low"] == event["epsilon_high"] == 0.2,
                    "actual numerical loss hyperparameters differ from the registered GRPO objective")
            current, old, ref, advantage, mask = (np.asarray(event[key], dtype=np.float32) for key in
                                                 ("current_logps", "old_logps", "reference_logps", "advantages", "completion_mask"))
            require(current.ndim == 2 and current.shape == old.shape == ref.shape == mask.shape and
                    all(np.isfinite(value).all() for value in (current, old, ref, advantage, mask)), "non-finite/misaligned GRPO numeric vectors")
            ratio = np.exp(current - old)
            clipped = np.clip(ratio, 1 - event["epsilon_low"], 1 + event["epsilon_high"])
            safe = np.clip(ref - current, -20, 20)
            raw_k3 = np.exp(safe) - safe - 1
            token_loss = -np.minimum(ratio * advantage, clipped * advantage) + event["beta"] * np.clip(raw_k3, -10, 10)
            lengths = mask.sum(-1)
            require(np.all(lengths > 0) and np.isin(mask, [0, 1]).all(), "empty/nonbinary numerical completion mask")
            reconstructed = float(((token_loss * mask).sum(-1) / lengths).mean())
            tolerance = 1e-5 + 1e-4 * abs(reconstructed)
            require(abs(reconstructed - event["actual_framework_loss"]) <= tolerance
                    and abs(reconstructed - event["recomputed_loss"]) <= tolerance,
                    "independent recomputation disagrees with actual GRPO objective/length normalization")
            require(event["completion_lengths"] == lengths.astype(int).tolist()
                    and event["kl_logratio_clamp_activated_tokens"] == int(np.count_nonzero((ref - current < -20) | (ref - current > 20)))
                    and event["kl_value_clamp_activated_tokens"] == int(np.count_nonzero((raw_k3 < -10) | (raw_k3 > 10))),
                    "recorded completion lengths or KL clamp counts contradict token vectors")
            numeric_ids.extend(event["rollout_request_ids"])
            require(len(event["rollout_request_ids"]) == 1, "numeric receipt must correspond to one actual completion microbatch")
            numeric_by_uuid[event["rollout_request_ids"][0]] = event
        require(len(numeric_ids) == len(set(numeric_ids)) and set(numeric_ids) == completion_ids,
                "GRPO numerical loss receipts do not bind exactly the consumed completion UUIDs")
        reward_paths = list(root.glob("rollouts.rank*.jsonl"))
        require(len(reward_paths) == 1, "single-card technical run needs exactly one reward ledger")
        reward_rows = read_rows(reward_paths[0]); evidence_files.add(reward_paths[0])
        reward_by_uuid = {row["rollout_request_id"]: row for row in reward_rows}
        require(len(reward_by_uuid) == len(reward_rows) == len(all_rows) and set(reward_by_uuid) == completion_ids,
                "reward and actually consumed rollout UUID sets differ")
        for row in all_rows:
            reward = reward_by_uuid[row["rollout_request_id"]]
            source = registered[row["dataset_request_id"]]
            instances = {item.get("pool_instance_id"): item["dataset_request_id"] for item in data_rows}
            require(row.get("pool_instance_id") == reward.get("pool_instance_id")
                    and instances.get(row.get("pool_instance_id")) == row["dataset_request_id"], "reward/consumption pool instance identity differs")
            require(reward["dataset_request_id"] == row["dataset_request_id"] and reward["media_signature"] == source["media_signature"]
                    and reward["clip_ids"] == source["clip_ids"] and reward["pseudo_scores"] == source["pseudo_scores"]
                    and reward["sampled_pts"] == source["sampled_pts"] and reward["dataset_sha256"] == sha(data_path),
                    "reward metadata binds to a different actual dataset input")
            require(hashlib.sha256(reward["completion"].encode()).hexdigest() == reward["completion_sha256"], "reward completion hash changed")
            calculated = asdict(reward_parts(reward["completion"], reward["pseudo_scores"], reward["rank_weight"]))
            require(calculated == reward["reward"], "reward ledger differs from recomputed strict reward")
        reward_groups = OrderedDict()
        for row in reward_rows:
            reward_groups.setdefault((row["call_index"], row["group_in_call"]), []).append(row)
        import torch
        for group in reward_groups.values():
            require(len(group) == 4, "reward ledger contains partial G group")
            rewards = torch.tensor([row["reward"]["total"] for row in group], dtype=torch.float32)
            actual_advantages = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-4)
            for row, expected_advantage in zip(group, actual_advantages.tolist()):
                numeric = numeric_by_uuid[row["rollout_request_id"]]
                advantages = np.broadcast_to(np.asarray(numeric["advantages"]), np.asarray(numeric["completion_mask"]).shape)
                active = advantages[np.asarray(numeric["completion_mask"], dtype=bool)]
                require(np.allclose(active, expected_advantage, rtol=1e-4, atol=1e-5),
                        "real GRPO advantage disagrees with actual group rewards/sample std/epsilon")
        if not has_nonzero_update:
            negative_outcome = validate_formal_zero_signal(
                begin, end, optimizer_calls, reward_groups, numeric_events, recomputed,
                expected_updates=expected_updates, prompts_per_update=prompts_per_update)
    # Swift creates a fresh vN-timestamp subdirectory below the requested root.
    checkpoints = sorted((path for path in Path(receipt["output_dir"]).rglob("checkpoint-*") if path.is_dir()),
                         key=lambda path: int(path.name.rsplit("-", 1)[1]))
    require(checkpoints, "trainer produced no saved checkpoint")
    require(len({path.parent.resolve() for path in checkpoints}) == 1, "requested output root contains multiple trainer versions; endpoint is ambiguous")
    checkpoint = checkpoints[-1]
    trainer_state_path = checkpoint / "trainer_state.json"
    require(read_json(trainer_state_path)["global_step"] == expected_updates, "saved trainer_state does not match technical endpoint")
    evidence_files.add(trainer_state_path)
    checkpoint_info = checkpoint_matches_last_update(root, checkpoint, begin, optimizer_calls[-1], evidence_files)
    return {"root": str(root), "mode": mode, "counts_recomputed": recomputed,
            "checkpoint": str(checkpoint), "checkpoint_evidence": checkpoint_info,
            "versions": versions, "stage_costs": costs, "consumed_rows": all_rows,
            "dataset_path": str(data_path), "model_path": str(Path(one(flags, "--model")).resolve()),
            "begin": begin, "end": end, "negative_outcome": negative_outcome,
            "parameter_changed_updates": parameter_changed_updates}


def validate_formal_zero_signal(begin, end, optimizer_calls, reward_groups, numerical_events, counts,
                                *, expected_updates, prompts_per_update):
    """A fully observed stationary negative result; never a technical shortcut."""
    require(expected_updates == len(optimizer_calls) == 256 and prompts_per_update == 4
            and counts["successful_optimizer_updates"] == 256
            and counts["prompt_exposures_in_successful_updates"] == len(reward_groups) == 1024
            and counts["rollout_completions_generated"] == counts["rollout_completions_consumed_in_successful_updates"]
            == len(numerical_events) == 4096, "stationary negative result lacks the complete frozen formal budget")
    initial = begin["initial_trainable_sha256"]
    require(initial == end["final_trainable_sha256"] and end.get("trainable_changed_since_begin") is False,
            "stationary negative result changed the adapter's initial-to-final identity")
    learning_rates = []
    for event in optimizer_calls:
        require(event["call_returned"] is True and event["skipped"] is False
                and event["gradient"]["finite"] is True and event["gradient"]["n_gradient_tensors"] > 0
                and event["gradient"]["l2_norm"] == 0 and event["gradient"]["n_nonzero_gradient_tensors"] == 0
                and event["parameter_changed"] is False
                and event["trainable_sha256_before"] == event["trainable_sha256_after"] == initial,
                "stationary negative outcome has nonzero/missing gradients, parameter drift or optimizer failure")
        rates = event.get("learning_rates_before_step")
        require(isinstance(rates, list) and rates and all(math.isfinite(value) and value >= 0 for value in rates),
                "stationary negative result lacks actual pre-step optimizer learning-rate observations")
        learning_rates.extend(rates)
    require(any(value > 0 for value in learning_rates), "all actual optimizer learning rates were zero")
    for group in reward_groups.values():
        values = np.asarray([row["reward"]["total"] for row in group], dtype=np.float64)
        require(len(group) == 4 and np.isfinite(values).all() and np.std(values, ddof=1) == 0,
                "a group has reward variance; zero gradients cannot be accepted as a zero-signal result")
    for event in numerical_events:
        current, old, reference, advantages, mask = (np.asarray(event[key], dtype=np.float64) for key in
            ("current_logps", "old_logps", "reference_logps", "advantages", "completion_mask"))
        active = mask.astype(bool)
        require(np.any(active) and np.array_equal(current, reference) and np.array_equal(current, old)
                and np.all(np.broadcast_to(advantages, mask.shape)[active] == 0),
                "stationary negative result has nonzero advantages or current/old/reference logp differences")
        dref = np.clip(reference - current, -20, 20)
        kl = np.clip(np.exp(dref) - dref - 1, -10, 10)
        require(np.all(kl[active] == 0) and event["actual_framework_loss"] == 0
                and event["recomputed_loss"] == 0, "stationary negative result has nonzero KL or loss")
    return "ALL_GROUP_ADVANTAGES_ZERO_NO_POLICY_UPDATE"


def validate_snapshot(report_path, expected_adapter, evidence_files):
    path = Path(report_path).resolve(); evidence_files.add(path)
    report = read_json(path)
    require(report.get("mode") == "snapshot" and report.get("status") == "COMPLETED" and report.get("gpu_checkpoint_verification") is True,
            "save/reload snapshot is not a completed real GPU inference")
    require(report.get("human_highlight_labels_read") is False and report["versions"]["ms-swift"] == "4.6.0", "invalid snapshot label/version boundary")
    source = report["fixed_training_example"]
    require(sha(source["inputs_path"]) == source["inputs_sha256"], "fixed training-example source changed after snapshot")
    candidates = [row for row in read_rows(source["inputs_path"])
                  if row["request_id"] == source["request_id"] and row.get("pool_instance_id") == source.get("pool_instance_id")]
    require(any(canonical_hash(row) == source["row_sha256"] for row in candidates), "fixed snapshot row cannot be recovered from its source")
    evidence_files.add(Path(source["inputs_path"]).resolve())
    require(Path(report["adapter_path"]).resolve() == Path(expected_adapter).resolve(), "reload snapshot uses a different trained adapter")
    file_hash_map(report["adapter_path"], report["adapter_file_hashes"], evidence_files)
    file_hash_map(report["model_path"], report["model_file_hashes"], evidence_files)
    require(report["fixed_answer"]["supervised_tokens_including_eos"] == len(numeric_vector(report["fixed_answer"]["token_logps"])), "snapshot answer/EOS dimensions differ")
    require(isinstance(report["greedy"]["response"], str) and report["greedy"]["completion_tokens"] > 0, "saved adapter did not produce a reloaded generation")
    require(report.get("stages") and all(value["wall_seconds"] > 0 and value["peak_cuda_allocated_bytes"] > 0 for value in report["stages"].values()), "snapshot lacks actual GPU stage measurements")
    return report


def validate_quantized_merge(snapshot, merged, merged_path, evidence_files, acceptance):
    """Re-execute full CPU tensor arithmetic; a status flag is insufficient."""
    from types import SimpleNamespace
    from diagnose_merge import diagnose
    from verify_bf16_merge import logp_gap

    basis = "verified_bf16_tensor_merge_common_s0"
    require(acceptance.get("acceptance_basis") == basis and acceptance.get("definition")
            and acceptance.get("premerge_functional_equivalence_claimed") is False
            and acceptance.get("initial_tolerances_are_diagnostic_only") is True,
            "candidate does not explicitly define the actual BF16 common endpoint")
    require(acceptance["fixed_answer_logp_mean_absolute_error_max"] == 0.03
            and acceptance["fixed_answer_logp_max_absolute_error_max"] == 0.25,
            "initial probability diagnostic tolerances must remain unchanged")
    def checked_report(entry, name):
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str)
                and isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64,
                f"{name} has no completed immutable artifact identity")
        path = Path(entry["path"]).resolve()
        require(sha(path) == entry["sha256"], f"registered {name} evidence changed")
        evidence_files.add(path)
        return read_json(path)
    initial = acceptance["initial_failure"]
    original = checked_report(initial["comparison"], "initial_failed_comparison")
    old_diagnosis = checked_report(initial["tensor_diagnosis"], "initial_tensor_diagnosis")
    require(original.get("status") == "FAILED", "initial probability failure was overwritten or hidden")
    old_snapshot_path = Path(original["snapshot_path"])
    require(sha(old_snapshot_path) == original["snapshot_sha256"], "initial failure's snapshot changed")
    evidence_files.add(old_snapshot_path.resolve())
    initial_gap = logp_gap(read_json(old_snapshot_path), original)
    require(initial_gap["within_initial_tolerance"] is False, "registered initial failure no longer reproduces")
    require(old_diagnosis["observed_full_model_logp_comparison"]["compare_sha256"] == initial["comparison"]["sha256"],
            "historical tensor diagnosis is bound to a different initial failure")
    if acceptance.get("original_failed_compare") is not None:
        legacy_original = checked_report(acceptance["original_failed_compare"], "preserved_original_failed_comparison")
        require(legacy_original == original, "legacy original failure reference disagrees with initial_failure")
    current_identity = acceptance["current_diagnosis"]
    require(current_identity.get("status") == "COMPLETED", "current technical endpoint tensor proof is still pending")
    diagnosis = checked_report(current_identity, "current_tensor_diagnosis")
    # Current endpoints may be within the unchanged diagnostic tolerance or fail
    # only that tolerance. They need not reproduce the earlier endpoint's gap.
    gap = logp_gap(snapshot, merged)
    require(diagnosis.get("status") == "COMPLETED" and not diagnosis.get("failures"), "full merge tensor diagnosis is incomplete")
    observed = diagnosis["observed_full_model_logp_comparison"]
    require(observed["compare_sha256"] == sha(merged_path) and observed["snapshot_sha256"] == merged["snapshot_sha256"]
            and observed["original_status_retained"] == merged["status"] and observed["threshold_modified"] is False,
            "current tensor proof is not bound to the current unchanged probability comparison")
    roots = {"base": Path(snapshot["model_path"]).resolve(), "adapter": Path(snapshot["adapter_path"]).resolve(),
             "merged": Path(merged["model_path"]).resolve()}
    for role, root in roots.items():
        require(Path(diagnosis[role]).resolve() == root, f"tensor diagnosis uses a different {role} identity")
        for name, digest in diagnosis["weight_files_sha256"][role].items():
            path = root / name
            require(sha(path) == digest, f"{role} weight changed after tensor diagnosis")
            evidence_files.add(path)
    per_lora, non_lora = diagnosis["per_lora_matrix"], diagnosis["non_lora_tensors"]
    names = [row["tensor"] for row in per_lora + non_lora]
    require(len(set(names)) == len(names) == diagnosis["n_model_tensors"]
            and len(per_lora) == diagnosis["n_lora_matrices"] > 0
            and len(non_lora) == diagnosis["n_non_lora_tensors"] > 0,
            "full-tensor diagnosis inventory is incomplete or duplicated")
    require(all(row["exact_bf16_mismatch_elements"] == 0 and row["scaling"] == 2.0
                and row["n_elements"] == math.prod(row["shape"]) for row in per_lora)
            and all(row["bitwise_unchanged"] is True and row["n_elements"] == math.prod(row["shape"]) for row in non_lora),
            "per-tensor diagnosis contains a real merge mismatch")
    independent = {"model_files_modified": False, "gpu_used": False}
    arguments = SimpleNamespace(base=roots["base"], adapter=roots["adapter"], merged=roots["merged"], threads=8,
                                snapshot=Path(merged["snapshot_path"]), compare=Path(merged_path))
    started = time.monotonic()
    with redirect_stdout(io.StringIO()):
        diagnose(arguments, independent)
    require(not independent["failures"] and independent["aggregate"]["different_from_exact_bf16_elements"] == 0
            and all(row["bitwise_unchanged"] for row in independent["non_lora_tensors"]),
            "independent full-tensor arithmetic recomputation failed")
    require(independent["n_model_tensors"] == diagnosis["n_model_tensors"]
            and independent["weight_files_sha256"] == diagnosis["weight_files_sha256"],
            "independent tensor proof differs from the registered weight identity")
    for name in ("diagnose_merge.py", "verify_bf16_merge.py"):
        evidence_files.add((Path(__file__).parent / name).resolve())
    return {"acceptance_basis": basis, "merge_validation_passed": True,
            "merge_logp_within_initial_tolerance": gap["within_initial_tolerance"], "merge_numeric": gap,
            "premerge_functional_equivalence_claimed": False, "definition": acceptance["definition"],
            "initial_failure": initial, "initial_failure_gap": initial_gap,
            "current_diagnosis": current_identity,
            "independent_tensor_recomputation": {"n_model_tensors": independent["n_model_tensors"],
                "n_lora_matrices": independent["n_lora_matrices"], "n_non_lora_tensors": independent["n_non_lora_tensors"],
                "aggregate": independent["aggregate"], "wall_seconds": time.monotonic() - started, "gpu_used": False}}


def validate_merge(snapshot, merged_path, reference_path, evidence_files, tolerances):
    merged, reference = read_json(merged_path), read_json(reference_path)
    evidence_files.update({Path(merged_path).resolve(), Path(reference_path).resolve()})
    require(merged.get("status") in {"FAILED", "COMPLETED"} and reference.get("status") == "COMPLETED" and merged["mode"] == "compare-merged"
            and reference["mode"] == "reference-update", "merge/reference check reports incomplete")
    require(reference.get("gpu_reference_update_verified") is True, "reference evidence did not execute on GPU")
    require(merged["snapshot_sha256"] == sha(merged["snapshot_path"]), "merge snapshot binding changed")
    require(read_json(merged["snapshot_path"])["fixed_training_example"] == snapshot["fixed_training_example"] == merged["fixed_training_example"] == reference["fixed_training_example"], "merge/reference checks did not use the same fixed training example")
    merge_evidence = validate_quantized_merge(snapshot, merged, merged_path, evidence_files, tolerances)
    require(Path(reference["model_path"]).resolve() == Path(merged["model_path"]).resolve(), "reference check does not use merged technical S0")
    file_hash_map(merged["model_path"], merged["model_file_hashes"], evidence_files)
    file_hash_map(reference["model_path"], reference["model_file_hashes"], evidence_files)
    require(reference["frozen_base_parameters_before"] == reference["frozen_base_parameters_after"], "full frozen base fingerprints drifted")
    require(reference["frozen_base_parameters_before"] and all(not item["requires_grad"] for item in reference["frozen_base_parameters_before"].values()), "full base frozen inventory missing")
    changed = [name for name, before in reference["trainable_parameters_before"].items()
               if before["sha256"] != reference["trainable_parameters_after"][name]["sha256"]]
    require(changed, "fresh reference-test LoRA did not actually change")
    grads = reference["integrity_test_update"]["per_parameter_gradients"]
    require(grads and all(math.isfinite(item["norm"]) and math.isfinite(item["max_abs"]) for item in grads.values())
            and any(item["norm"] > 0 for item in grads.values()), "fresh reference-test LoRA gradients invalid")
    invariance = compare_logps(reference["reference_before"], reference["reference_after"], 0, 0)
    compare_logps(reference["merged_base_before_adapter"], reference["reference_before"], 0, 0)
    compare_logps(reference["merged_base_before_adapter"], reference["new_adapter_initial"], 1e-6, 1e-6)
    reload_numeric = compare_logps(reference["updated_adapter"], reference["reloaded_adapter"], 1e-6, 1e-6)
    require(reference["updated_greedy"]["response"] == reference["reloaded_greedy"]["response"], "fresh adapter save/reload greedy output differs")
    file_hash_map(reference["saved_adapter_path"], reference["saved_adapter_file_hashes"], evidence_files)
    return {**merge_evidence, "reference_invariance_recomputed": invariance,
            "fresh_adapter_reload_recomputed": reload_numeric, "n_changed_fresh_lora_parameters": len(changed)}


def reward_numeric_probe():
    legal = reward_parts('{"scores":[10,90]}', [0.1, 0.9])
    constant = reward_parts('{"scores":[50,50]}', [0.1, 0.9])
    bad = ['{"scores":[10,90]', '{"scores":[true,90]}', '{"scores":[10,90],"extra":1}', '{"scores":[10,90],"scores":[10,90]}', '{"scores":[NaN,90]}']
    values = [reward_parts(text, [0.1, 0.9]) for text in bad]
    require(legal.valid and all(not item.valid and item.total == -1 for item in values) and legal.total > constant.total,
            "strict/invalid/truncated/constant reward numerical check failed")
    return {"scope": "CPU numerical probe executed by freeze script; does not prove GPU trainer integration",
            "legal": asdict(legal), "constant_scores": asdict(constant), "invalid_examples": bad,
            "invalid_rewards": [asdict(item) for item in values]}


def validate_null_reward_diagnostic(specification, evidence_files):
    """Verify the real zero-signal run without accepting it as an update proof."""
    root = Path(specification["run_dir"]).resolve()
    required_files = {"receipt.json", "training_summary.json", "training_ledger.jsonl", "rollouts.rank0.jsonl"}
    require(set(specification["files"]) == required_files, "zero-signal diagnostic file inventory differs")
    for name, digest in specification["files"].items():
        path = root / name
        require(sha(path) == digest, f"zero-signal evidence changed: {name}")
        evidence_files.add(path)
    receipt, summary = read_json(root / "receipt.json"), read_json(root / "training_summary.json")
    ledger, rewards = read_rows(root / "training_ledger.jsonl"), read_rows(root / "rollouts.rank0.jsonl")
    require(receipt.get("exit_code") == 0 and summary.get("status") == "TRAINER_ENDED_REQUIRES_ACCEPTANCE",
            "zero-signal diagnostic did not complete the real trainer loop")
    begin = [row for row in ledger if row["event"] == "training_begin"]
    end = [row for row in ledger if row["event"] == "training_end"]
    calls = [row for row in ledger if row["event"] == "optimizer_call"]
    updates = [row for row in ledger if row["event"] == "optimizer_update"]
    numerical = [row for row in ledger if row["event"] == "grpo_loss_numeric"]
    micros = [row for row in ledger if row["event"] == "training_microbatch"]
    require(len(begin) == len(end) == 1 and len(calls) == len(updates) == 4 and len(rewards) == len(numerical) == len(micros) == 16,
            "zero-signal diagnostic lacks its four-update / sixteen-completion actual loop")
    require(begin[0]["runtime_source_contract"]["packages"]["ms-swift"] == "4.6.0"
            and begin[0]["initial_trainable_sha256"] == end[0]["final_trainable_sha256"]
            and begin[0]["frozen_sentinel"] == end[0]["final_frozen_sentinel"], "zero-signal parameter/frozen state does not reproduce")
    require(all(row["call_returned"] is True and row["skipped"] is False
                and row["gradient"]["finite"] is True and row["gradient"]["n_gradient_tensors"] > 0
                and row["gradient"]["l2_norm"] == 0 and row["gradient"]["n_nonzero_gradient_tensors"] == 0
                and row["trainable_sha256_before"] == row["trainable_sha256_after"] for row in calls),
            "claimed zero-signal optimizer/gradient observations are inconsistent")
    require(all(row["cuda_measured"] is True and row["peak_allocated_bytes"] > 0 for row in micros),
            "zero-signal diagnostic has no actual GPU training measurements")
    require(all(row["status"] == "SUCCESSFUL" and len(row["dataset_request_ids_in_order"]) == 1 for row in updates),
            "zero-signal exposure/update receipts are incomplete")
    consumed = [record for row in micros for record in row["records"]]
    uuids = [row["rollout_request_id"] for row in consumed]
    require(len(set(uuids)) == 16 and set(uuids) == {row["rollout_request_id"] for row in rewards},
            "zero-signal reward and actual consumption completion UUIDs differ")
    groups = OrderedDict()
    for row in rewards:
        calculated = asdict(reward_parts(row["completion"], row["pseudo_scores"], row["rank_weight"]))
        require(calculated == row["reward"] and calculated["valid"] is False and calculated["total"] == -1,
                "zero-signal diagnostic contains a legal or non-minus-one answer")
        groups.setdefault((row["call_index"], row["group_in_call"]), []).append(row)
    require(len(groups) == 4 and all(len(group) == 4 and np.std([row["reward"]["total"] for row in group], ddof=1) == 0
            and all(row["group_reward_std_sample"] == 0 for row in group) for group in groups.values()),
            "real equal-reward group standard deviations do not reproduce as zero")
    for event in numerical:
        advantage, mask = np.asarray(event["advantages"]), np.asarray(event["completion_mask"], dtype=bool)
        require(np.isfinite(advantage).all() and np.all(np.broadcast_to(advantage, mask.shape)[mask] == 0)
                and abs(event["actual_framework_loss"]) <= 1e-7 and abs(event["recomputed_loss"]) <= 1e-7,
                "zero-signal advantages or actual length-normalized loss are nonzero")
    require(summary["counts"]["successful_optimizer_updates"] == 4
            and summary["counts"]["prompt_exposures_in_successful_updates"] == 4
            and summary["counts"]["rollout_completions_generated"] == 16
            and summary["counts"]["rollout_completions_consumed_in_successful_updates"] == 16,
            "zero-signal diagnostic summary counters disagree")
    return {"status": "OBSERVED_ZERO_SIGNAL", "run_dir": str(root), "n_optimizer_calls": 4, "n_prompt_exposures": 4,
            "n_generated_and_consumed_completions": 16, "invalid_answer_fraction": 1.0,
            "all_rewards": -1, "group_reward_std": 0, "actual_advantages": 0,
            "observed_lora_gradient_norms": 0, "lora_parameter_changed": False,
            "does_not_satisfy_nonzero_update_acceptance": True, "excluded_from_formal_exposures": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, default=PROJECT / "configs/round1.candidate.json")
    for option in ("sft-smoke", "grpo-smoke", "formal-concurrency", "input-audit", "snapshot", "merged-compare",
                   "reference-update", "grpo-reload", "split", "prepared", "pseudo"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--sft-reload", type=Path)
    p.add_argument("--dependency-lock", type=Path, default=PROJECT / "configs/environment.freeze.txt")
    p.add_argument("--models-root", type=Path, default=Path("/root/autodl-fs/unsgrpo/models"))
    p.add_argument("--smoke-input-audit", type=Path)
    p.add_argument("--subset-receipt", type=Path)
    p.add_argument("--annotations-root", type=Path, action="append", default=[])
    p.add_argument("--extra-freeze", type=Path, nargs="*", default=[])
    p.add_argument("--out-technical", type=Path, default=PROJECT / "reports/technical_acceptance.json")
    p.add_argument("--out-freeze", type=Path, default=PROJECT / "reports/freeze.json")
    args = p.parse_args(argv)
    if args.out_freeze.exists() and read_json(args.out_freeze).get("status") == "FROZEN":
        raise FileExistsError("refusing to overwrite an already frozen protocol")
    report = {"status": "INCOMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_training_verified": False, "checks": {key: False for key in REQUIRED_CHECKS},
              "evidence": {}, "missing_or_invalid": [], "human_highlight_labels_read_by_this_script": False,
              "technical_models_are_not_formal_common_s0": True, "exact_optimizer_resume_verified": False}
    files = set()
    def attempt(name, fn):
        try:
            result = fn()
            report["evidence"][name] = result
            return result
        except Exception as exc:
            report["missing_or_invalid"].append({"check": name, "error": f"{type(exc).__name__}: {exc}"})
            return None
    def checked_candidate():
        obj = read_json(args.candidate)
        expected = {"S0": (256, 16, 1), "S+": (256, 4, 1), "G-CAL": (256, 4, 4), "G-RANK": (256, 4, 4)}
        require(set(obj["matrix"]) == {"B0", "B1", *expected}, "candidate does not contain the six registered arms")
        for name, (updates, exposures, group) in expected.items():
            arm = obj["matrix"][name]
            require((arm["updates"], arm["prompt_exposures_per_update"], arm["group_size"], arm["seed"]) ==
                    (updates, exposures, group, 42), f"candidate budget differs from original first round: {name}")
        require(obj["matrix"]["B0"]["updates"] == obj["matrix"]["B1"]["updates"] == 0, "baselines must not train")
        require(obj["merge_numeric_acceptance"]["fixed_answer_logp_mean_absolute_error_max"] == 0.03 and
                obj["merge_numeric_acceptance"]["fixed_answer_logp_max_absolute_error_max"] == 0.25,
                "merge tolerances differ from preregistered BF16 thresholds")
        technical = obj["technical_budget"]
        require(technical.get("reason") and technical.get("excluded_from_formal_exposures") is True
                and technical.get("human_evaluation_labels_used") is False, "technical budget revision lacks its scope/reason")
        for name in ("SFT", "GRPO"):
            require(type(technical[name]["updates"]) is int and technical[name]["updates"] > 0
                    and technical[name]["prompt_exposures_per_update"] == 1,
                    f"invalid explicit technical {name} update/exposure budget")
        require(technical["SFT"]["group_size"] == 1 and technical["GRPO"]["group_size"] == 4
                and technical["CONCURRENCY"]["minimum_updates"] >= 2
                and technical["CONCURRENCY"]["prompt_exposures_per_update"] == 4
                and technical["CONCURRENCY"]["group_size"] == 4, "technical grouping/concurrency specification differs")
        return obj
    candidate = attempt("candidate", checked_candidate)
    def boundary():
        roots = args.annotations_root or [Path("/root/autodl-fs/unsgrpo/datasets/labels_eval_only")]
        hits = []
        for root in roots:
            if root.exists():
                hits.extend(str(path) for path in root.rglob("*") if path.is_file() and (path.suffix == ".npz" or path.name == "conversion_receipt.json"))
        hits.extend(str(path) for path in (PROJECT / "reports").glob("*v0*.json"))
        require(not hits, f"evaluation/converted-label artifacts already exist before freeze: {hits}")
        require(candidate["data"]["human_training_labels"] is False, "candidate permits human training labels")
        return {"checked_annotation_roots": [str(root) for root in roots], "derived_evaluation_artifacts_found": hits,
                "limit": "Filesystem and provenance checks cannot prove absence of every possible out-of-band read."}
    boundary_ok = attempt("label_boundary", boundary)
    def assets():
        split, prepared, pseudo = read_rows(args.split), read_rows(args.prepared), read_json(args.pseudo)
        split_check = audit_split(split); audit_split(prepared)
        require({(row["video_id"], row["split"], row["source_group"]) for row in split} ==
                {(row["video_id"], row["split"], row["source_group"]) for row in prepared}, "split/prepared source identities differ")
        require(pseudo["prepared_manifest_sha256"] == sha(args.prepared) and set(pseudo["videos"]) == {row["video_id"] for row in prepared}, "pseudo data does not bind to prepared split")
        require(pseudo["protocol"]["embedding"] == "umap" and set(pseudo["protocol"]["fit_video_ids"]) == {row["video_id"] for row in split if row["split"] == "train"}, "proxy does not use train-only real UMAP")
        for row in split:
            path = Path(row["path"]); require(sha(path) == row["sha256"], f"raw video hash changed: {path}"); files.add(path.resolve())
        for row in prepared:
            path = Path(row["features_path"]); require(sha(path) == row["features_sha256"], "CLIP feature file changed"); files.add(path.resolve())
            media = row.get("media_files_sha256", {})
            expected_media = {path for clip in row["frame_paths"] for path in clip} | {row["overview"], row["frame_bounds_path"]}
            require(set(media) == expected_media, "prepared media/native PTS seal is absent or incomplete")
            for path, digest in media.items():
                require(sha(path) == digest, f"prepared media changed: {path}"); files.add(Path(path).resolve())
        model_assets = {}
        for role, id_key, revision_key in (("policy", "policy", "policy_revision"), ("features", "features", "features_revision")):
            receipts = []
            for path in args.models_root.rglob("asset_receipt.json"):
                item = read_json(path)
                if item.get("repo") == candidate["models"][id_key] and item.get("resolved_revision") == candidate["models"][revision_key]:
                    receipts.append((path, item))
            require(len(receipts) == 1, f"cannot uniquely bind {role} model to resolved revision and weight hashes")
            path, item = receipts[0]; files.add(path.resolve())
            for asset in item["files"]:
                local = path.parent / asset["file"]
                require(local.stat().st_size == asset["bytes"] and sha(local) == asset["sha256"], f"{role} asset hash mismatch: {local}")
                files.add(local.resolve())
            if role == "features":
                require(all(row["encoder_asset_receipt_sha256"] == sha(path) for row in prepared), "CLIP features were produced by a different encoder asset identity")
            model_assets[role] = {"receipt": str(path), "receipt_sha256": sha(path), "revision": item["resolved_revision"]}
        files.update(path.resolve() for path in args.pseudo.parent.iterdir() if path.is_file())
        return {"split_audit": split_check, "n_videos": len(split), "split_sha256": sha(args.split),
                "prepared_sha256": sha(args.prepared), "pseudo_sha256": sha(args.pseudo), "model_assets": model_assets}
    assets_ok = attempt("assets_and_split", assets)
    def audit(path=args.input_audit, require_full=True):
        obj = read_json(path)
        require(obj.get("status") == "COMPLETED" and obj.get("all_rows_encoded") is True and not obj.get("errors"), "full real input audit incomplete")
        rows_path = Path(obj["rows_path"]); require(sha(rows_path) == obj["rows_sha256"], "input audit row hash changed")
        rows = read_rows(rows_path); require(len(rows) == obj["n_rows_encoded"] > 0, "audit row count differs")
        for path, digest in obj["input_sha256"].items():
            require(sha(path) == digest, "audited model input changed"); files.add(Path(path).resolve())
        if require_full:
            require({"train_sft.jsonl", "train_grpo.jsonl", "eval.jsonl"} <= {Path(path).name for path in obj["input_sha256"]},
                    "formal audit is missing training or all-covering evaluation input files")
        for row in rows:
            require(row["no_truncation"] is True and row["input_frames"] == row["processed_frames"], "audit contains truncation/frame resampling")
            if row["task"] == "sft":
                require(row["supervised_tokens"] > 0 and row["answer_eos_suffix_ids"] and row["labels_sha256"], "assistant/EOS audit absent")
        std = obj["framework_source"]["advantage_numerical_check"]
        actual = numeric_vector(std["actual_advantages"])
        rewards = np.asarray(std["fixture_rewards"], dtype=float)
        expected = (rewards - rewards.mean()) / (rewards.std(ddof=1) + 1e-4)
        require(std["correction"] == 1 and np.allclose(actual, expected) and np.array_equal(std["constant_group_advantages"], np.zeros(4)), "actual framework sample-std/equal-reward numerical probe differs")
        for source in obj["framework_source"]["sources"].values():
            require(sha(source["path"]) == source["sha256"], "audited framework source changed"); files.add(Path(source["path"]).resolve())
        files.add(rows_path.resolve())
        return {"n_rows": len(rows), "rows_path": str(rows_path), "rows_sha256": sha(rows_path), "std_correction": 1,
                "input_sha256": obj["input_sha256"], "visual_environment": obj["visual_environment"], "model_path": obj["model_path"]}
    audit_ok = attempt("full_input_and_framework_audit", audit)
    smoke_audit_ok = attempt("technical_subset_input_audit", lambda: audit(args.smoke_input_audit, False)) if args.smoke_input_audit else audit_ok
    def subset_provenance():
        require(audit_ok and smoke_audit_ok, "both formal and technical input audits must complete")
        require(audit_ok["visual_environment"] == smoke_audit_ok["visual_environment"] and
                Path(audit_ok["model_path"]).resolve() == Path(smoke_audit_ok["model_path"]).resolve(),
                "technical subset changes model or visual budget")
        if args.smoke_input_audit is None:
            return {"technical_subset_used": False}
        require(args.subset_receipt is not None, "separate technical subset requires source-row receipt")
        receipt = read_json(args.subset_receipt); files.update({args.subset_receipt.resolve(), args.smoke_input_audit.resolve()})
        source_by_name = {Path(path).name: Path(path) for path in audit_ok["input_sha256"]}
        indices = receipt["source_line_indices"]
        require(indices and all(type(index) is int and index >= 0 for index in indices), "invalid technical subset row indices")
        for target_path in smoke_audit_ok["input_sha256"]:
            target_path = Path(target_path); name = target_path.name
            require(name in source_by_name and sha(source_by_name[name]) == receipt["source"][name], "technical subset source hash differs from formal pool")
            source_rows, target_rows = read_rows(source_by_name[name]), read_rows(target_path)
            selected = [source_rows[index] for index in indices]
            require(len(target_rows) == len(selected) and [row["request_id"] for row in target_rows] == receipt["request_ids"], "technical subset IDs/row counts differ")
            strip = lambda row: {key: value for key, value in row.items() if key != "pool_instance_id"}
            require(all(strip(first) == strip(second) for first, second in zip(selected, target_rows)),
                    "technical subset changes original prompt, media, q or source identity")
        return {"technical_subset_used": True, "source_line_indices": indices,
                "request_ids": receipt["request_ids"], "subset_receipt_sha256": sha(args.subset_receipt),
                "formal_training_exposure_counts_include_technical_subset": False}
    subset_ok = attempt("technical_subset_provenance", subset_provenance)
    numeric = attempt("reward_numeric_probe", reward_numeric_probe)
    null_diagnostic = attempt("real_equal_reward_zero_signal", lambda: validate_null_reward_diagnostic(
        candidate["technical_budget"]["null_reward_diagnostic"], files)) if candidate else None
    runs = {}
    technical = candidate["technical_budget"] if candidate else {}
    specifications = (("sft_smoke", args.sft_smoke, "sft", "SFT"),
                      ("grpo_smoke", args.grpo_smoke, "grpo", "GRPO"),
                      ("formal_concurrency", args.formal_concurrency, "grpo", "CONCURRENCY"))
    def checked_technical_run(path, mode, budget):
        obj = validate_run(path, mode, budget.get("updates"), budget["prompt_exposures_per_update"], files)
        require(obj["counts_recomputed"]["successful_optimizer_updates"] >= budget.get("minimum_updates", budget.get("updates", 1)),
                "formal concurrency probe did not reach its explicit minimum update budget")
        return obj
    for name, path, mode, budget_name in specifications:
        budget = technical.get(budget_name, {})
        result = attempt(name, lambda path=path, mode=mode, budget=budget:
                         checked_technical_run(path, mode, budget))
        if result:
            runs[name] = result
            # Keep heavy per-completion evidence in immutable source ledgers.
            report["evidence"][name] = {key: value for key, value in result.items() if key not in {"consumed_rows", "begin", "end"}}
    def run_audit_bindings():
        require(subset_ok and smoke_audit_ok and len(runs) == 3, "technical run/audit evidence incomplete")
        allowed = smoke_audit_ok["input_sha256"]
        for name, run in runs.items():
            require(run["dataset_path"] in {str(Path(path).resolve()) for path in allowed}, f"technical {name} dataset is outside audited subset")
            require(run["begin"]["audit_rows_sha256"] == smoke_audit_ok["rows_sha256"], "actual trainer used different input audit rows")
        return {"bound_runs": list(runs), "technical_audit_rows_sha256": smoke_audit_ok["rows_sha256"]}
    run_bindings_ok = attempt("run_input_audit_bindings", run_audit_bindings)
    snapshots = {}
    for name, path in (("sft_smoke", args.sft_reload or args.snapshot), ("grpo_smoke", args.grpo_reload)):
        if name in runs:
            result = attempt(name + "_reload", lambda path=path, name=name: validate_snapshot(path, runs[name]["checkpoint"], files))
            if result:
                snapshots[name] = result
    merge_ok = attempt("merge_and_reference", lambda: validate_merge(snapshots["sft_smoke"], args.merged_compare,
                              args.reference_update, files, candidate["merge_numeric_acceptance"])) if "sft_smoke" in snapshots else None
    def technical_ancestry():
        merged = read_json(args.merged_compare)
        require(merge_ok and len(runs) == 3, "technical merge/run evidence incomplete")
        require(Path(runs["sft_smoke"]["model_path"]).resolve() == Path(audit_ok["model_path"]).resolve(), "technical SFT uses a different base policy")
        require(all(Path(runs[name]["model_path"]).resolve() == Path(merged["model_path"]).resolve()
                    for name in ("grpo_smoke", "formal_concurrency")), "GRPO smoke/concurrency do not share the verified merged technical S0")
        return {"technical_merged_s0": merged["model_path"], "formal_common_s0_registered_later": True}
    ancestry_ok = attempt("technical_model_ancestry", technical_ancestry)
    both = "sft_smoke" in runs and "grpo_smoke" in runs
    report["checks"].update({
        "generation_reward_logps_backward_optimizer_save_reload": both and len(snapshots) == 2,
        "same_group_media_q_id": "grpo_smoke" in runs and "formal_concurrency" in runs,
        "assistant_only_sft_no_grpo_answers": bool(audit_ok) and both,
        "strict_json_equal_reward_truncation": bool(numeric and audit_ok and null_diagnostic),
        "finite_lora_gradients_parameter_update_frozen_backbone": both and bool(merge_ok),
        "s0_bf16_tensor_merge_common_endpoint": bool(merge_ok), "fixed_s0_reference_after_update": bool(merge_ok),
        "old_ref_reward_advantage_detached": "grpo_smoke" in runs,
        "completion_only_loss_eos_lengths_std_sampling": bool(audit_ok) and "grpo_smoke" in runs,
        "stage_memory_time_recorded": both and "formal_concurrency" in runs,
        "formal_rollout_concurrency_verified": "formal_concurrency" in runs,
    })
    for key in REQUIRED_CHECKS:
        if report["checks"][key] is not True:
            report["missing_or_invalid"].append({"check": key, "error": "required evidence did not pass recomputation"})
    files.update(path.resolve() for path in (args.candidate, args.input_audit, args.split, args.prepared, args.pseudo, args.dependency_lock))
    files.update(path.resolve() for folder in ("src", "scripts", "configs") for path in (PROJECT / folder).rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".sh", ".json", ".txt", ".toml"})
    files.update(path.resolve() for path in (PROJECT / "storage-env.sh", PROJECT / "pyproject.toml", PROJECT / "docs/EVALUATION_PROTOCOL.md"))
    files.update(path.resolve() for path in args.extra_freeze)
    hashed = {}
    for path in sorted(files):
        try:
            hashed[str(path)] = sha(path)
        except Exception as exc:
            report["missing_or_invalid"].append({"check": "freeze_files", "error": f"{path}: {exc}"})
    ready = bool(candidate and boundary_ok and assets_ok and audit_ok and subset_ok and run_bindings_ok and ancestry_ok
                 and all(report["checks"].values()) and not report["missing_or_invalid"])
    if ready:
        report.update(status="COMPLETED", gpu_training_verified=True)
    report["evidence_files_sha256"] = hashed
    args.out_technical.parent.mkdir(parents=True, exist_ok=True)
    args.out_technical.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    hashed[str(args.out_technical.resolve())] = sha(args.out_technical)
    freeze = {"status": "FROZEN" if ready else "INCOMPLETE", "created_utc": report["created_utc"],
              "technical_acceptance": str(args.out_technical.resolve()), "files": hashed,
              "matrix": candidate.get("matrix", {}) if candidate else {}, "protocol": candidate,
              "human_annotations_read_for_training": False, "label_boundary": report["evidence"].get("label_boundary"),
              "missing_or_invalid": report["missing_or_invalid"]}
    args.out_freeze.parent.mkdir(parents=True, exist_ok=True)
    args.out_freeze.write_text(json.dumps(freeze, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": freeze["status"], "n_passed_checks": sum(report["checks"].values()),
                      "n_required_checks": len(REQUIRED_CHECKS), "n_missing_or_invalid": len(report["missing_or_invalid"])}))
    return 0 if ready else 2


if __name__ == "__main__":
    sys.exit(main())
