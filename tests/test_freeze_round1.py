"""Integrity gates: declarations cannot replace raw numerical/file evidence."""
import hashlib
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_round1.py"
spec = importlib.util.spec_from_file_location("freeze_round1_test", SCRIPT)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def test_logp_passed_boolean_cannot_override_recomputed_difference():
    before = {"token_logps": [-1.0, -2.0], "answer_token_ids": [4, 5], "passed": True}
    after = {"token_logps": [-1.0, -1.0], "answer_token_ids": [4, 5], "passed": True}
    with pytest.raises(ValueError, match="difference exceeds"):
        freeze.compare_logps(before, after, 0.03, 0.25)
    assert freeze.compare_logps(before, before, 0, 0)["max_abs"] == 0


def test_strict_reward_probe_executes_real_core():
    result = freeze.reward_numeric_probe()
    assert result["legal"]["valid"]
    assert all(row["total"] == -1 and not row["valid"] for row in result["invalid_rewards"])
    assert result["legal"]["total"] > result["constant_scores"]["total"]


def test_saved_adapter_must_match_last_optimizer_tensor_hash(tmp_path):
    name = "model.language_model.layer.lora_A.default.weight"
    saved_name = "model.language_model.layer.lora_A.weight"
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    digest = hashlib.sha256()
    for data in (name.encode(), str(value.dtype).encode(), json.dumps(list(value.shape)).encode(), value.view(torch.uint8).numpy().tobytes()):
        digest.update(data)
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    save_file({saved_name: value}, str(checkpoint / "adapter_model.safetensors"))
    begin = {"trainable_parameters": [{"name": name, "shape": [2, 3], "dtype": "torch.float32"}]}
    evidence = {"trainable_sha256_after": digest.hexdigest()}
    result = freeze.checkpoint_matches_last_update(tmp_path, checkpoint, begin, evidence, set())
    assert result["saved_trainable_sha256"] == digest.hexdigest()
    save_file({saved_name: value + 1}, str(checkpoint / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="differs from last actual optimizer"):
        freeze.checkpoint_matches_last_update(tmp_path, checkpoint, begin, evidence, set())


def test_declared_success_without_raw_training_events_rejected(tmp_path):
    (tmp_path / "receipt.json").write_text(json.dumps({"exit_code": 0, "status": "COMPLETED", "gpu_training_verified": True}))
    (tmp_path / "training_summary.json").write_text(json.dumps({"mode": "grpo", "status": "TRAINER_ENDED_REQUIRES_ACCEPTANCE",
                                                               "pending_uncommitted_rows": 0, "all_checks_passed": True}))
    (tmp_path / "training_ledger.jsonl").write_text("")
    with pytest.raises(ValueError, match="begin/end"):
        freeze.validate_run(tmp_path, "grpo", 4, 1, set())


def test_missing_artifacts_write_incomplete_and_never_freeze(tmp_path, monkeypatch):
    original_project = Path(__file__).parents[1]
    candidate = tmp_path / "candidate.json"
    candidate.write_text((original_project / "configs/round1.candidate.json").read_text())
    monkeypatch.setattr(freeze, "PROJECT", tmp_path)
    arguments = ["--candidate", str(candidate)]
    for option in ("sft-smoke", "grpo-smoke", "formal-concurrency", "input-audit", "snapshot", "merged-compare",
                   "reference-update", "grpo-reload", "split", "prepared", "pseudo"):
        arguments.extend(["--" + option, str(tmp_path / (option + "-missing"))])
    arguments.extend(["--out-technical", str(tmp_path / "technical.json"), "--out-freeze", str(tmp_path / "freeze.json"),
                      "--annotations-root", str(tmp_path / "unused-labels")])
    assert freeze.main(arguments) == 2
    report = json.loads((tmp_path / "technical.json").read_text())
    protocol = json.loads((tmp_path / "freeze.json").read_text())
    assert report["status"] == protocol["status"] == "INCOMPLETE"
    assert report["gpu_training_verified"] is False
    assert not any(report["checks"].values())
    assert report["evidence"]["reward_numeric_probe"]["legal"]["valid"]


def test_model_hash_table_tampering_detected(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    path = model / "config.json"
    path.write_text("{}")
    table = {"config.json": {"bytes": 2, "sha256": freeze.sha(path)}}
    freeze.file_hash_map(model, table, set())
    path.write_text("{\"different\":true}")
    with pytest.raises(ValueError, match="file changed"):
        freeze.file_hash_map(model, table, set())


def test_old_tensor_diagnosis_cannot_validate_a_new_current_endpoint(tmp_path):
    """Synthetic metadata fixture; no GPU execution or full acceptance claimed."""
    def write(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return {"path": str(path), "sha256": freeze.sha(path)}

    def snapshot(model):
        return {"status": "COMPLETED", "mode": "snapshot", "gpu_checkpoint_verification": True,
                "fixed_answer": {"token_logps": [-1.0, -2.0], "answer_token_ids": [4, 5]},
                "greedy": {"response": "synthetic only"}, "model_path": str(model),
                "adapter_path": str(tmp_path / "new-adapter")}

    def comparison(first, snap_entry, delta, status):
        second = [-1.0 - delta, -2.0 - delta]
        differences = np.abs(np.asarray(first["fixed_answer"]["token_logps"]) - second)
        return {"mode": "compare-merged", "status": status, "gpu_checkpoint_verification": status == "COMPLETED",
                "error": "RuntimeError: Merge exceeded preregistered BF16 logp tolerance; synthetic fixture",
                "snapshot_path": snap_entry["path"], "snapshot_sha256": snap_entry["sha256"],
                "model_path": str(tmp_path / "new-merged"), "fixed_answer": {"token_logps": second, "answer_token_ids": [4, 5]},
                "greedy": {"response": "synthetic only"},
                "stages": {key: {"wall_seconds": 1, "peak_cuda_allocated_bytes": 1}
                           for key in ("load_model", "fixed_answer_forward", "greedy_generation")},
                "merge_logp_comparison": {"mean_abs_tolerance": .03, "max_abs_tolerance": .25,
                                          "mean_abs": float(differences.mean()), "max_abs": float(differences.max()),
                                          "passed": status == "COMPLETED"}}
    old_snapshot = snapshot(tmp_path / "old-base")
    old_snap_entry = write("old-snapshot.json", old_snapshot)
    old_compare = write("old-compare.json", comparison(old_snapshot, old_snap_entry, .05, "FAILED"))
    historical_diagnosis = write("historical-diagnosis.json", {"status": "COMPLETED", "failures": [],
         "observed_full_model_logp_comparison": {"compare_sha256": old_compare["sha256"]}})
    current_snapshot = snapshot(tmp_path / "new-base")
    current_snap_entry = write("new-snapshot.json", current_snapshot)
    current_comparison = comparison(current_snapshot, current_snap_entry, .01, "COMPLETED")
    current_compare = write("new-compare.json", current_comparison)
    wrong_current = write("wrong-current-diagnosis.json", {"status": "COMPLETED", "failures": [],
         "base": str(tmp_path / "old-base"),
         "observed_full_model_logp_comparison": {"compare_sha256": current_compare["sha256"],
             "snapshot_sha256": current_snap_entry["sha256"], "original_status_retained": "COMPLETED", "threshold_modified": False}})
    acceptance = {"acceptance_basis": "verified_bf16_tensor_merge_common_s0", "definition": "actual merged endpoint",
                  "premerge_functional_equivalence_claimed": False, "initial_tolerances_are_diagnostic_only": True,
                  "fixed_answer_logp_mean_absolute_error_max": .03, "fixed_answer_logp_max_absolute_error_max": .25,
                  "initial_failure": {"comparison": old_compare, "tensor_diagnosis": historical_diagnosis},
                  "original_failed_compare": old_compare, "current_diagnosis": {"status": "COMPLETED", **wrong_current}}
    with pytest.raises(ValueError, match="different base identity"):
        freeze.validate_quantized_merge(current_snapshot, current_comparison, current_compare["path"], set(), acceptance)


@pytest.fixture
def stationary_formal_classification_fixture():
    """Classification-only synthetic fixture; does not satisfy outer GPU gates."""
    identity = "a" * 64
    begin = {"initial_trainable_sha256": identity}
    end = {"final_trainable_sha256": identity, "trainable_changed_since_begin": False}
    call = {"call_returned": True, "skipped": False, "gradient": {"finite": True, "n_gradient_tensors": 2,
             "l2_norm": 0, "n_nonzero_gradient_tensors": 0}, "parameter_changed": False,
             "trainable_sha256_before": identity, "trainable_sha256_after": identity,
             "learning_rates_before_step": [1e-5]}
    calls = [copy.deepcopy(call) for _ in range(256)]
    calls[0]["learning_rates_before_step"] = [0.0]
    groups = {i: [{"reward": {"total": float(i % 3) / 2}} for _ in range(4)] for i in range(1024)}
    numerical = {"current_logps": [[-1, -2]], "old_logps": [[-1, -2]], "reference_logps": [[-1, -2]],
                 "advantages": [[0, 0]], "completion_mask": [[1, 1]], "actual_framework_loss": 0, "recomputed_loss": 0}
    events = [copy.deepcopy(numerical) for _ in range(4096)]
    counts = {"successful_optimizer_updates": 256, "prompt_exposures_in_successful_updates": 1024,
              "rollout_completions_generated": 4096, "rollout_completions_consumed_in_successful_updates": 4096}
    return begin, end, calls, groups, events, counts


def test_stationary_formal_negative_outcome_requires_complete_evidence(stationary_formal_classification_fixture):
    result = freeze.validate_formal_zero_signal(*stationary_formal_classification_fixture,
                                               expected_updates=256, prompts_per_update=4)
    assert result == "ALL_GROUP_ADVANTAGES_ZERO_NO_POLICY_UPDATE"


@pytest.mark.parametrize("defect", ["positive_variance", "nonzero_gradient", "all_lr_zero", "reference_drift",
                                   "nonzero_advantage", "nonzero_loss", "parameter_change", "incomplete_budget"])
def test_stationary_negative_classifier_rejects_false_shortcuts(stationary_formal_classification_fixture, defect):
    begin, end, calls, groups, events, counts = stationary_formal_classification_fixture
    if defect == "positive_variance":
        groups[0][0]["reward"]["total"] += 1e-6
    elif defect == "nonzero_gradient":
        calls[10]["gradient"]["l2_norm"] = 1e-6
        calls[10]["gradient"]["n_nonzero_gradient_tensors"] = 1
    elif defect == "all_lr_zero":
        for call in calls:
            call["learning_rates_before_step"] = [0.0]
    elif defect == "reference_drift":
        events[0]["reference_logps"][0][0] += 1e-6
    elif defect == "nonzero_advantage":
        events[0]["advantages"][0][0] = 1e-10
    elif defect == "nonzero_loss":
        events[0]["actual_framework_loss"] = 1e-8
    elif defect == "parameter_change":
        calls[0]["trainable_sha256_after"] = "b" * 64
        calls[0]["parameter_changed"] = True
    else:
        counts["prompt_exposures_in_successful_updates"] = 1023
    with pytest.raises(ValueError):
        freeze.validate_formal_zero_signal(begin, end, calls, groups, events, counts,
                                           expected_updates=256, prompts_per_update=4)


@pytest.mark.parametrize("mode,receipt_mode,prompts", [("sft", "formal", 16), ("grpo", "smoke", 4)])
def test_zero_signal_permission_does_not_apply_to_sft_or_technical_runs(tmp_path, mode, receipt_mode, prompts):
    (tmp_path / "receipt.json").write_text(json.dumps({"mode": receipt_mode}))
    (tmp_path / "training_summary.json").write_text("{}")
    (tmp_path / "training_ledger.jsonl").write_text("")
    with pytest.raises(ValueError, match="only for the complete registered formal GRPO"):
        freeze.validate_run(tmp_path, mode, 256, prompts, set(), allow_formal_zero_signal=True)
