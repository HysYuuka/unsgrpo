"""Controller safety tests only; never launch a model, GPU task or real labels."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/execute_round1.py"
SPEC = importlib.util.spec_from_file_location("execute_round1_tested", SCRIPT)
driver_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver_module)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    work = tmp_path / "work"
    monkeypatch.setattr(driver_module, "WORK", work)
    state = work / "runs/fixture_round"
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    target = tmp_path / "system_python"
    target.write_text("fixture executable; never run")
    executable.symlink_to(target)
    args = argparse.Namespace(state_dir=state, python=executable, checkpoints=work / "checkpoints",
                              freeze=tmp_path / "freeze.json", input_audit=tmp_path / "audit.json",
                              data=tmp_path / "inputs", model=tmp_path / "base", manifest=tmp_path / "prepared.jsonl",
                              pseudo=tmp_path / "pseudo.json", annotations_zip=tmp_path / "labels.zip", resume=False)
    driver = driver_module.Driver(args)
    driver.common = tmp_path / "common_s0.json"
    return driver


def checkpoint(root, version="v0-20261027-190000", step=256):
    path = root / version / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(b"schema fixture, not model weights")
    (path / "adapter_config.json").write_text("{}")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return path


def gap_fixtures(driver):
    driver.checks.mkdir(parents=True)
    snapshot = {"status": "COMPLETED", "mode": "snapshot", "gpu_checkpoint_verification": True,
                "fixed_answer": {"token_logps": [-1., -2.]}, "greedy": {"response": "fixture"}}
    compared = {"status": "FAILED", "mode": "compare-merged", "gpu_checkpoint_verification": False,
                "error": "RuntimeError: Merge exceeded preregistered BF16 logp tolerance; retain failure and investigate",
                "fixed_answer": {"token_logps": [-1.1, -2.2]}, "greedy": {"response": "fixture"},
                "merge_logp_comparison": {"mean_abs": .15000000000000013, "max_abs": .20000000000000018,
                                          "mean_abs_tolerance": .03, "max_abs_tolerance": .25, "passed": False},
                "stages": {name: {"wall_seconds": 1., "peak_cuda_allocated_bytes": 1024}
                           for name in ("load_model", "fixed_answer_forward", "greedy_generation")}}
    (driver.checks / "S0_snapshot.json").write_text(json.dumps(snapshot))
    (driver.checks / "S0_merge_compare.json").write_text(json.dumps(compared))
    return compared


def test_plan_does_not_create_state_or_read_labels(controller):
    plan = controller.plan()
    assert not controller.root.exists()
    assert plan["executes_commands"] is False
    stages = plan["stages"]
    assert stages.index("S0_bitwise_verify") < stages.index("common_S0_register")
    assert stages.index("S0_reference_verify") < stages.index("common_S0_register")
    assert all(stages.index("infer_" + name) < stages.index("annotation_extract")
               for name in ("B0", "B1", "S0", "SPLUS", "G-CAL", "G-RANK"))


def test_venv_python_symlink_is_not_resolved_away(controller):
    assert controller.python == str(controller.args.python.absolute())
    assert controller.python != str(controller.args.python.resolve())


def test_unique_version_zero_final_endpoint_only(tmp_path):
    expected = checkpoint(tmp_path)
    assert driver_module.final_checkpoint(tmp_path) == expected
    checkpoint(tmp_path, "v1-20261027-190001")
    with pytest.raises(ValueError, match="exactly one"):
        driver_module.final_checkpoint(tmp_path)


def test_single_wrong_version_is_not_silently_selected(tmp_path):
    checkpoint(tmp_path, "v1-20261027-190001")
    with pytest.raises(ValueError, match="Unexpected output version"):
        driver_module.final_checkpoint(tmp_path)


def test_completed_stage_hash_revalidated_on_resume(controller, tmp_path):
    artifact = tmp_path / "finished.json"
    artifact.write_text("old")
    stage = {"status": "COMPLETED", "artifacts": {str(artifact): driver_module.sha(artifact)}}
    controller.verify_completed(stage)
    artifact.write_text("new")
    with pytest.raises(ValueError, match="changed/missing"):
        controller.verify_completed(stage)


def test_failed_controller_never_restarts_trainer(controller, monkeypatch):
    controller.args.resume = True
    controller.args.freeze.write_text("{}")
    controller.root.mkdir(parents=True)
    state = {"status": "FAILED", "freeze_sha256": driver_module.sha(controller.args.freeze),
             "execution_config": controller.execution_config(),
             "stages": {name: {"status": "PLANNED"} for name in controller.stages()}}
    driver_module.atomic(controller.state_path, state)
    monkeypatch.setattr(controller, "validate_freeze", lambda: {})
    monkeypatch.setattr(driver_module.subprocess, "Popen", lambda *a, **kw: pytest.fail("no subprocess allowed"))
    with pytest.raises(ValueError, match="Failed controller"):
        controller.execute()


def test_only_registered_logp_tolerance_failure_can_continue(controller, monkeypatch):
    compared = gap_fixtures(controller)
    def fake_command(argv, name, *, allowed_diagnostic_failure=None, **kwargs):
        assert allowed_diagnostic_failure(1) is True
    monkeypatch.setattr(controller, "command", fake_command)
    assert controller.compare("S0_compare") == [controller.checks / "S0_merge_compare.json"]
    compared["error"] = "RuntimeError: CUDA out of memory"
    (controller.checks / "S0_merge_compare.json").write_text(json.dumps(compared))
    with pytest.raises(ValueError, match="other failures block"):
        controller.compare("S0_compare")


def test_common_registration_keeps_failed_initial_gap_false(controller, monkeypatch):
    gap_fixtures(controller)
    endpoint = checkpoint(controller.output_root / "fixture_S0")
    controller.runs["S0"].mkdir(parents=True)
    parent = {"status": "COMPLETED", "mode": "formal", "method_id": "S0", "gpu_training_verified": True,
              "endpoint": {"path": str(endpoint),
              "files": {"adapter_model.safetensors": driver_module.sha(endpoint / "adapter_model.safetensors")}}}
    (controller.runs["S0"] / "receipt.json").write_text(json.dumps(parent))
    (controller.runs["S0"] / "acceptance.json").write_text("{}")
    controller.args.freeze.write_text("{}")
    controller.merged.mkdir(parents=True)
    (controller.merged / "model.safetensors").write_bytes(b"not real weights; fixture only")
    (controller.checks / "S0_bf16_merge_verification.json").write_text(json.dumps({
        "status": "COMPLETED", "merge_validation_passed": True,
        "acceptance_basis": "verified_bf16_tensor_merge_common_s0", "merge_formula_verified_all_tensors": True,
        "aggregate": {"different_from_exact_bf16_elements": 0}}))
    (controller.checks / "S0_reference_integrity.json").write_text(json.dumps({
        "status": "COMPLETED", "gpu_reference_update_verified": True, "reference_invariance": {"exact_equal": True}}))
    controller.merge_run.mkdir(parents=True)
    (controller.merge_run / "processor_restore.json").write_text(json.dumps({"status": "COMPLETED",
        "model_weights_modified": False, "restored_entire_processor_package_byte_identical": True,
        "fixed_train_row_encode_verified": True}))
    monkeypatch.setattr(controller, "checkpoint", lambda method: endpoint)
    controller.register_common("common_S0_register")
    result = driver_module.read(controller.common)
    assert result["merge_logp_within_initial_tolerance"] is False
    assert result["merge_validation_passed"] is True
    assert result["premerge_functional_equivalence_claimed"] is False
    assert "merge_logp_passed" not in result
