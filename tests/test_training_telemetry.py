"""CPU bookkeeping/hook tests; never substitute these for Swift/Qwen GPU smoke."""
import hashlib
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch


path = Path(__file__).parents[1] / "scripts" / "training_telemetry.py"
spec = importlib.util.spec_from_file_location("unsgrpo_training_telemetry_test", path)
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)


@pytest.mark.parametrize("numel", [1, 2, 15, 16, 2**24 + 1, 2**24 + 4, 151936 * 2048, 2**40 + 3])
def test_large_frozen_sentinel_indices_use_exact_integer_bounds(numel):
    indices = telemetry.sentinel_sample_indices(numel)
    assert indices[0] == 0
    assert indices[-1] == numel - 1
    assert len(indices) == min(16, numel)
    assert len(set(indices)) == len(indices)
    assert all(type(index) is int and 0 <= index < numel for index in indices)


def test_grpo_loss_reconstruction_hand_calculated_length_normalization():
    # Ratio=1, KL=0. Sequence A: -2 averaged across two valid tokens;
    # sequence B: +1 across one valid token. Equal-sequence mean = -0.5,
    # whereas pooling three tokens would give -1. Masked token cannot count.
    result = telemetry.recompute_grpo_loss([[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                                           [[2, 2], [-1, -1]], [[1, 1], [1, 0]],
                                           beta=0.04, epsilon_low=0.2, epsilon_high=0.2)
    assert result["recomputed_loss"] == -0.5
    assert result["completion_lengths"] == [2, 1]


def test_grpo_loss_reconstruction_records_actual_kl_and_ratio_clamps():
    result = telemetry.recompute_grpo_loss([[0, 0]], [[-2, 0]], [[30, -30]], [[1, 1]], [[1, 1]],
                                           beta=0.04, epsilon_low=0.2, epsilon_high=0.2)
    assert result["kl_logratio_clamp_activated_tokens"] == 2
    assert result["kl_value_clamp_activated_tokens"] == 2
    assert result["ppo_ratio_clipped_tokens"] == 1
    # clipped positive-advantage losses (-1.2,-1), capped KL contribution +0.4 each.
    assert result["recomputed_loss"] == pytest.approx(-0.7)


def test_teacher_capability_without_real_teacher_is_allowed_and_recorded():
    trainer = SimpleNamespace(_has_teacher=True, _has_teacher_explicit=lambda: False)
    sample = SimpleNamespace(teacher_prompt=None, teacher_images=None, teacher_messages=None, extra={})
    result = telemetry.require_no_active_teacher(trainer, samples=[sample],
                                                grpo_batch=SimpleNamespace(teacher_per_token_logps=None))
    assert result["capability_flag"] is True
    assert result["active"] is False and result["explicit_teacher"] is False
    assert result["n_teacher_prompts"] == 0


@pytest.mark.parametrize("active_source", ["explicit", "token_logps", "prompt"])
def test_actual_teacher_signals_are_rejected(active_source):
    trainer = SimpleNamespace(_has_teacher=True, _has_teacher_explicit=lambda: active_source == "explicit")
    sample = SimpleNamespace(teacher_prompt="teacher" if active_source == "prompt" else None,
                             teacher_images=None, teacher_messages=None, extra={})
    batch = SimpleNamespace(teacher_per_token_logps=torch.tensor([0.1]) if active_source == "token_logps" else None)
    with pytest.raises(ValueError, match="actual teacher"):
        telemetry.require_no_active_teacher(trainer, samples=[sample], grpo_batch=batch)


def test_loss_reconstruction_against_installed_unchanged_swift_core_cpu():
    swift_spec = importlib.util.find_spec("swift")
    if swift_spec is None:
        pytest.skip("installed Swift is required for source-bound CPU comparison")
    source = Path(swift_spec.origin).parent / "rlhf_trainers" / "grpo_trainer.py"
    tree = ast.parse(source.read_text())
    method = next(method for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer"
                  for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "_compute_loss_and_metrics")
    namespace = {"torch": torch, "Dict": Dict, "Any": Any, "GRPOBatch": object,
                 "nanmin": torch.min, "nanmax": torch.max, "gather_object": lambda values: values}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    current = torch.tensor([[0.0, -0.2, -0.5], [-0.1, -0.3, -0.6]], requires_grad=True)
    old = torch.tensor([[-1.0, -0.2, -0.5], [-0.1, -0.4, -0.6]])
    reference = torch.tensor([[30.0, -30.0, -0.5], [-0.1, -0.3, -0.6]])
    advantages = torch.tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    signals = SimpleNamespace(completion_mask=mask, truncated_mask=torch.zeros(2, dtype=torch.bool),
                              ref_per_token_logps=reference, old_per_token_logps=old,
                              advantages=advantages, rollout_per_token_logps=None)
    trainer = SimpleNamespace(model=torch.nn.Linear(1, 1), compute_entropy=False, overlong_filter=False,
                              beta=0.04, kl_in_reward=False, rollout_importance_sampling_mode=None,
                              log_rollout_offpolicy_metrics=False, disable_rollout_importance_sampling=True,
                              importance_sampling_level="token", _has_teacher=True,
                              off_policy_sequence_mask_delta=None, loss_type="grpo", epsilon_low=0.2,
                              epsilon_high=0.2, sdar_loss_coef=0, chord_sft_iterator=None,
                              args=SimpleNamespace(delta=None),
                              accelerator=SimpleNamespace(gather_for_metrics=lambda value: value),
                              _get_per_token_logps_and_entropies=lambda *args, **kwargs: (current, None))
    actual, _ = namespace["_compute_loss_and_metrics"](trainer, trainer.model, {}, signals)
    reconstructed = telemetry.recompute_grpo_loss(current.detach(), old, reference, advantages, mask,
                                                 beta=0.04, epsilon_low=0.2, epsilon_high=0.2)
    assert actual.item() == pytest.approx(reconstructed["recomputed_loss"], abs=1e-5, rel=1e-4)
    actual.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()


def row(dataset_id="window", *, group="g0", completion="c0", media="same"):
    return {"dataset_request_id": dataset_id, "group_id": group, "rollout_request_id": completion,
            "prompt_id": "prompt_0", "media_tensor_sha256": media, "pseudo_scores_sha256": "q",
            "clip_ids": [0, 1], "prompt_tokens": 10, "encoded_tokens": 14, "loss_tokens": 4}


def test_prefetch_never_counts_and_skipped_update_does_not_commit_exposure(tmp_path):
    ledger = telemetry.ExposureLedger(tmp_path / "ledger.jsonl", mode="sft")
    ledger.write("encoded_or_prefetched_fixture", dataset_request_id="must_not_count")
    assert ledger.save_summary()["counts"] == {}
    ledger.consumed([row("a")])
    assert ledger.save_summary()["counts"].get("prompt_exposures_in_successful_updates", 0) == 0
    ledger.commit(skipped=True, global_step=1, optimizer_evidence={"skipped": True})
    assert ledger.save_summary()["counts"].get("successful_optimizer_updates", 0) == 0
    ledger.consumed([row("b")])
    ledger.commit(skipped=False, global_step=2, optimizer_evidence={"skipped": False})
    result = ledger.close()
    assert result["dataset_request_id_order"] == ["b"]
    assert result["counts"]["skipped_optimizer_steps"] == 1
    assert result["counts"]["prompt_exposures_in_successful_updates"] == 1


def test_grpo_repeated_same_window_is_two_exposures_eight_completions(tmp_path):
    ledger = telemetry.ExposureLedger(tmp_path / "ledger.jsonl", mode="grpo")
    for group in ("g0", "g1"):
        for i in range(4):
            ledger.consumed([row(group=group, completion=f"{group}-{i}")])
    ledger.commit(skipped=False, global_step=1, optimizer_evidence={})
    result = ledger.close()
    assert result["counts"]["prompt_exposures_in_successful_updates"] == 2
    assert result["counts"]["rollout_completions_consumed_in_successful_updates"] == 8
    assert result["counts"]["effective_loss_tokens"] == 32
    assert result["unique_windows"] == 1
    assert result["window_repetitions"] == {"window": 2}
    assert result["dataset_request_id_order"] == ["window", "window"]


@pytest.mark.parametrize("defect", ["partial", "different_media", "duplicate_completion"])
def test_grpo_group_content_and_full_consumption_required(tmp_path, defect):
    ledger = telemetry.ExposureLedger(tmp_path / "ledger.jsonl", mode="grpo")
    rows = [row(completion=f"c{i}") for i in range(4)]
    if defect == "partial":
        rows.pop()
    elif defect == "different_media":
        rows[-1]["media_tensor_sha256"] = "different"
    else:
        rows[-1]["rollout_request_id"] = "c0"
    for record in rows:
        ledger.consumed([record])
    with pytest.raises(ValueError):
        ledger.commit(skipped=False, global_step=1, optimizer_evidence={})
    assert ledger.counts["successful_optimizer_updates"] == 0
    ledger.close("FAILED_EXPECTED_TEST_FIXTURE")


def test_ledger_refuses_append_to_prior_run(tmp_path):
    ledger = telemetry.ExposureLedger(tmp_path / "ledger.jsonl", mode="sft")
    with pytest.raises(FileExistsError):
        telemetry.ExposureLedger(tmp_path / "ledger.jsonl", mode="sft")
    ledger.close()


def fake_inputs_and_audit():
    inputs = {"input_ids": torch.tensor([[7, 8, 9]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
              "labels": torch.tensor([[-100, -100, 9]]),
              "pixel_values": torch.tensor([[0.1, 0.2]]), "pixel_values_videos": torch.tensor([[0.3, 0.4]]),
              "image_grid_thw": torch.tensor([[1, 1, 1]]), "video_grid_thw": torch.tensor([[1, 1, 1]])}
    audit = {"task": "sft", "dataset_request_id": "source__000000_000001", "clip_ids": [0, 1],
             "input_ids_sha256": telemetry.json_hash([7, 8, 9]),
             "media_tensor_sha256": telemetry.tensor_media_hash(inputs),
             "media_signature": "source-media", "prompt_tokens": 2, "supervised_tokens": 1,
             "labels_sha256": hashlib.sha256(np.array([-100, -100, 9], dtype=np.int64).tobytes()).hexdigest()}
    return inputs, audit


def test_audit_matches_consumed_pixels_and_assistant_labels_not_metadata_only():
    inputs, audit = fake_inputs_and_audit()
    index = telemetry.AuditIndex([audit])
    assert index.match_sft(inputs)["dataset_request_id"] == audit["dataset_request_id"]
    changed_labels = dict(inputs, labels=torch.tensor([[7, -100, 9]]))
    with pytest.raises(ValueError, match="labels differ"):
        index.match_sft(changed_labels)
    changed_media = dict(inputs, pixel_values=inputs["pixel_values"] + 0.01)
    with pytest.raises(ValueError, match="tokens/media"):
        index.match_sft(changed_media)


def test_identical_window_pool_repeats_allow_different_audit_line_numbers():
    _, audit = fake_inputs_and_audit()
    telemetry.AuditIndex([dict(audit, line_no=1, pool_instance_id="p1"),
                          dict(audit, line_no=2, pool_instance_id="p2")])


class ToyLoRAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.ones(2))


class FakeTrainer:
    def __init__(self):
        self.model = ToyLoRAModel()
        self.template = SimpleNamespace(padding_free=False, tokenizer=SimpleNamespace(eos_token_id=9))
        self.accelerator = SimpleNamespace(device=torch.device("cpu"), optimizer_step_was_skipped=False)
        self.state = SimpleNamespace(global_step=0)
        self.optimizer = torch.optim.SGD([self.model.lora_A], lr=0.1)

    def _prepare_inputs(self, inputs):
        return inputs

    def training_step(self, model, inputs):
        self._prepare_inputs(inputs)
        loss = (model.lora_A.sum() - 0.5).square()
        loss.backward()
        return loss.detach()


def test_actual_cpu_optimizer_hook_tracks_gradient_parameter_change_and_freeze(tmp_path, monkeypatch):
    inputs, audit = fake_inputs_and_audit()
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(json.dumps(audit) + "\n")
    monkeypatch.setenv("HL_INPUT_AUDIT_ROWS", str(audit_path))
    monkeypatch.setenv("HL_TRAINING_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("HL_TELEMETRY_SUMMARY", str(tmp_path / "summary.json"))
    args = SimpleNamespace(world_size=1, per_device_train_batch_size=1, packing=False, num_iterations=1, async_generate=False)
    trainer = FakeTrainer()
    base_before = trainer.model.base.weight.detach().clone()
    lora_before = trainer.model.lora_A.detach().clone()
    callback = telemetry.HighlightTelemetry(args, trainer)
    callback.on_train_begin(args, trainer.state, None)
    trainer.training_step(trainer.model, inputs)
    trainer.training_step(trainer.model, inputs)
    assert callback.ledger.counts["successful_optimizer_updates"] == 0
    trainer.optimizer.step()
    assert callback.optimizer_evidence["parameter_changed"]
    assert callback.optimizer_evidence["gradient"]["finite"]
    assert callback.optimizer_evidence["learning_rates_before_step"] == [0.1]
    trainer.state.global_step = 1
    callback.on_step_end(args, trainer.state, None)
    callback.on_train_end(args, trainer.state, None)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["counts"]["successful_optimizer_updates"] == 1
    assert result["counts"]["prompt_exposures_in_successful_updates"] == 2
    assert result["counts"]["effective_loss_tokens"] == 2
    assert result["frozen_sentinel_unchanged"]
    assert result["trainable_changed_since_begin"]
    assert result["integration_verified"] is False
    assert torch.equal(base_before, trainer.model.base.weight)
    assert not torch.equal(lora_before, trainer.model.lora_A)


def test_nonfinite_gradient_never_commits_successful_update(tmp_path, monkeypatch):
    inputs, audit = fake_inputs_and_audit()
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(json.dumps(audit) + "\n")
    monkeypatch.setenv("HL_INPUT_AUDIT_ROWS", str(audit_path))
    monkeypatch.setenv("HL_TRAINING_LEDGER", str(tmp_path / "ledger.jsonl"))
    args = SimpleNamespace(world_size=1, per_device_train_batch_size=1)
    trainer = FakeTrainer()
    callback = telemetry.HighlightTelemetry(args, trainer)
    callback.on_train_begin(args, trainer.state, None)
    trainer.training_step(trainer.model, inputs)
    trainer.model.lora_A.grad[0] = float("nan")
    with pytest.raises(FloatingPointError):
        trainer.optimizer.step()
    assert callback.ledger.counts["successful_optimizer_updates"] == 0
    assert callback.ledger.status == "FAILED"
    callback.ledger.close("FAILED_EXPECTED_TEST_FIXTURE")


class FakeGRPOTrainer(FakeTrainer):
    """Exercise actual wrapper boundaries, not the Swift/Qwen implementations."""
    def __init__(self, inputs, audit):
        super().__init__()
        self.num_generations = 4
        self.inputs, self.audit = inputs, audit
        self.buffer = None
        self.cursor = 0
        self._rewards_per_func = torch.tensor([[0.0], [0.1], [0.2], [0.3]])

    def split_by_mini_batches(self, samples):
        return [[sample] for sample in samples]

    def _generate_completions(self, samples):
        return samples

    def _score_completions(self, samples):
        return samples

    def _get_per_token_logps_and_entropies(self, model, inputs, signals):
        return model.lora_A.sum().reshape(1, 1) * 0.1, None

    def _prepare_batch_inputs(self, samples):
        batches = []
        for chunk in self.split_by_mini_batches(samples):
            model_inputs = {k: v for k, v in self.inputs.items() if k != "labels"}
            signals = SimpleNamespace(completion_mask=torch.ones(1, 1), advantages=torch.tensor([1.0]), logits_to_keep=1)
            chunk[0].encoded = {"input_ids": [7, 8, 9], "labels": [-100, -100, 9]}
            with torch.no_grad():
                signals.old_per_token_logps = self._get_per_token_logps_and_entropies(self.model, model_inputs, signals)[0]
                signals.ref_per_token_logps = self._get_per_token_logps_and_entropies(self.model, model_inputs, signals)[0]
            batches.append({"model_inputs": model_inputs, "grpo_batch": signals})
        return batches

    def _prepare_inputs(self, inputs):
        if self.buffer is None:
            samples = [SimpleNamespace(extra={"dataset_request_id": self.audit["dataset_request_id"],
                                               "media_signature": self.audit["media_signature"],
                                               "clip_ids": [0, 1], "pseudo_scores": [0.25, 0.75]},
                                       request_id=f"completion-{i}", prompt_id="prompt_0", is_truncated=False)
                       for i in range(4)]
            samples = self._generate_completions(samples)
            samples = self._score_completions(samples)
            self.buffer = self._prepare_batch_inputs(samples)
        result = self.buffer[self.cursor]
        self.cursor += 1
        return result

    def training_step(self, model, inputs):
        batch = self._prepare_inputs(inputs)
        logps = self._get_per_token_logps_and_entropies(model, batch["model_inputs"], batch["grpo_batch"])[0]
        loss = -logps.sum()
        loss.backward()
        return loss.detach()


def test_grpo_buffered_real_consumption_and_detached_logps_hooks(tmp_path, monkeypatch):
    inputs, audit = fake_inputs_and_audit()
    audit.update(task="grpo", input_ids_sha256=telemetry.json_hash([7, 8]), supervised_tokens=0)
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(json.dumps(audit) + "\n")
    monkeypatch.setenv("HL_INPUT_AUDIT_ROWS", str(audit_path))
    monkeypatch.setenv("HL_TRAINING_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("HL_TELEMETRY_SUMMARY", str(tmp_path / "summary.json"))
    args = SimpleNamespace(world_size=1, per_device_train_batch_size=1, packing=False, num_iterations=1, async_generate=False)
    trainer = FakeGRPOTrainer(inputs, audit)
    trainer._has_teacher = True
    trainer._has_teacher_explicit = lambda: False
    callback = telemetry.HighlightTelemetry(args, trainer)
    callback.on_train_begin(args, trainer.state, None)
    trainer.training_step(trainer.model, None)
    assert callback.ledger.counts["rollout_completions_generated"] == 4
    assert callback.ledger.counts["prompt_exposures_in_successful_updates"] == 0
    assert len(callback.bindings) == 3
    for _ in range(3):
        trainer.training_step(trainer.model, None)
    trainer.optimizer.step()
    trainer.state.global_step = 1
    callback.on_step_end(args, trainer.state, None)
    callback.on_train_end(args, trainer.state, None)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["counts"]["prompt_exposures_in_successful_updates"] == 1
    assert summary["counts"]["rollout_completions_consumed_in_successful_updates"] == 4
    assert summary["counts"]["effective_loss_tokens"] == 4
    events = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    logps = [event for event in events if event["event"] == "policy_logps"]
    assert sum(event["role"] == "current" and event["requires_grad"] for event in logps) == 4
    assert sum(event["role"] == "old_or_reference" and not event["requires_grad"] for event in logps) == 8
    assert set(summary["stage_costs"]) == {"generation", "reward", "encode_old_ref_logps", "forward_backward", "optimizer"}
    assert summary["integration_verified"] is False
    consumed = [event for event in events if event["event"] == "training_microbatch"]
    assert all(event["records"][0]["teacher_activity"]["capability_flag"] is True
               and event["records"][0]["teacher_activity"]["active"] is False for event in consumed)


def test_installed_transformers_cpu_trainer_callback_optimizer_order(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    inputs, audit = fake_inputs_and_audit()
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(json.dumps(audit) + "\n")
    monkeypatch.setenv("HL_INPUT_AUDIT_ROWS", str(audit_path))
    monkeypatch.setenv("HL_TRAINING_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("HL_TELEMETRY_SUMMARY", str(tmp_path / "summary.json"))

    class Model(ToyLoRAModel):
        def forward(self, input_ids=None, labels=None, **kwargs):
            return {"loss": (self.lora_A.sum() - 0.5).square(),
                    "logits": torch.zeros(1, 3, 10) + self.lora_A.sum()}

    args = transformers.TrainingArguments(output_dir=str(tmp_path / "checkpoint"),
                                          use_cpu=True, per_device_train_batch_size=1,
                                          gradient_accumulation_steps=2, max_steps=1,
                                          learning_rate=0.01, report_to=[], disable_tqdm=True,
                                          save_strategy="no", logging_strategy="no",
                                          remove_unused_columns=False)
    trainer = transformers.Trainer(model=Model(), args=args,
                                   train_dataset=[{"example": 0}, {"example": 1}],
                                   data_collator=lambda rows: dict(inputs))
    trainer.template = SimpleNamespace(padding_free=False)
    callback_type = type("CPURegisteredTelemetry", (telemetry.HighlightTelemetry, transformers.TrainerCallback), {})
    trainer.add_callback(callback_type(args, trainer))
    trainer.train()
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["counts"]["successful_optimizer_updates"] == 1
    assert result["counts"]["prompt_exposures_in_successful_updates"] == 2
    assert result["trainable_changed_since_begin"]
    assert result["frozen_sentinel_unchanged"]
    assert result["integration_verified"] is False
