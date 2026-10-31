"""Report gates on synthetic bookkeeping fixtures; no model/GPU claims are made."""
import collections
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_round1.py"
SPEC = importlib.util.spec_from_file_location("report_round1_tested", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def put(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": reporter.digest(path)}


@pytest.fixture
def consumed_sft(tmp_path):
    index = tmp_path / "index.json"
    put(index, {})
    data = tmp_path / "inputs.jsonl"
    put(data, {})
    audit_rows = tmp_path / "audit.jsonl"
    put(audit_rows, {})
    endpoint = tmp_path / "endpoint"
    endpoint.mkdir()
    put(endpoint / "adapter_config.json", {})
    (endpoint / "adapter_model.safetensors").write_bytes(b"synthetic schema fixture, not model weights")
    ids = [f"fixture_{i}" for i in range(16)]
    order = ids * 256
    events = [{"event": "training_begin", "audit_rows_sha256": reporter.digest(audit_rows),
               "non_adapter_requires_grad_false": True}]
    for step in range(256):
        events.extend([
            {"event": "training_microbatch", "records": [
                {"dataset_request_id": rid, "loss_tokens": 13, "encoded_tokens": 256} for rid in ids]},
            {"event": "optimizer_update", "status": "SUCCESSFUL", "dataset_request_ids_in_order": ids,
             "optimizer": {"call_returned": True, "skipped": False, "gradient": {"finite": True},
                           "parameter_changed": step < 255}},
        ])
    events.append({"event": "training_end", "frozen_sentinel_unchanged": True})
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(e) + "\n" for e in events))
    receipt = {"status": "COMPLETED", "method_id": "S0", "mode": "formal", "gpu_training_verified": True,
               "command": ["swift", "sft", "--dataset", str(data), "--model", str(tmp_path / "base"),
                           "--max_steps", "256", "--seed", "42"]}
    summary = {"status": "TRAINER_ENDED_REQUIRES_ACCEPTANCE", "pending_uncommitted_rows": 0,
               "counts": {"successful_optimizer_updates": 256, "prompt_exposures_in_successful_updates": 4096,
                          "effective_loss_tokens": 4096 * 13, "encoded_tokens_consumed": 4096 * 256},
               "dataset_request_id_order": order, "window_repetitions": dict(collections.Counter(order)),
               "stage_costs": {"fixture": {"cuda_measured": True, "wall_seconds": 1.0,
                                          "peak_allocated_bytes": 1234, "peak_reserved_bytes": 2345}},
               "single_gpu_allocation_wall_hours": 1.0, "exact_resume_verified": False}
    entry = {"training_receipt": put(tmp_path / "receipt.json", receipt),
             "telemetry_summary": put(tmp_path / "summary.json", summary),
             "training_ledger": {"path": str(ledger), "sha256": reporter.digest(ledger)},
             "endpoint": {"path": str(endpoint), "files": {p.name: reporter.digest(p) for p in endpoint.iterdir()}}}
    frozen = {str(data): reporter.digest(data), str(audit_rows): reporter.digest(audit_rows)}
    common = {"merged_model": str(tmp_path / "merged"), "sft_endpoint": str(endpoint)}
    return index, entry, frozen, common, receipt, summary, events


def check(fixture):
    index, entry, frozen, common, *_ = fixture
    audit = reporter.Audit(index)
    result = reporter.audit_training(audit, "S0", entry, frozen, common, "c" * 64)
    return audit, result


def test_consistent_256_update_consumption_contract(consumed_sft):
    audit, result = check(consumed_sft)
    assert not audit.errors
    assert result["observed_counts"]["prompt_exposures_in_successful_updates"] == 4096
    assert result["unique_windows"] == 16  # Exposures are not unique windows.


def test_smoke_with_matching_counts_still_rejected(consumed_sft):
    index, entry, _, _, receipt, *_ = consumed_sft
    receipt["mode"] = "smoke"
    entry["training_receipt"] = put(index.parent / "receipt.json", receipt)
    audit, result = check(consumed_sft)
    assert result["status"] == "INCOMPLETE"
    assert any("smoke" in error for error in audit.errors)


def test_four_real_updates_cannot_be_declared_256(consumed_sft):
    _, entry, _, _, _, _, events = consumed_sft
    path = Path(entry["training_ledger"]["path"])
    path.write_text("".join(json.dumps(e) + "\n" for e in [events[0], *events[1:9], events[-1]]))
    entry["training_ledger"]["sha256"] = reporter.digest(path)
    audit, result = check(consumed_sft)
    assert result["status"] == "INCOMPLETE"
    assert any("实际successful_optimizer_updates=4" in error for error in audit.errors)


def test_missing_cost_is_not_imputed_to_zero(consumed_sft):
    index, entry, _, _, _, summary, _ = consumed_sft
    summary["single_gpu_allocation_wall_hours"] = None
    entry["telemetry_summary"] = put(index.parent / "summary.json", summary)
    audit, _ = check(consumed_sft)
    assert any("未观测训练GPU占用墙钟" in error for error in audit.errors)


def test_missing_index_produces_incomplete_without_fake_ap(tmp_path):
    path = tmp_path / "index.json"
    put(path, {})
    result = reporter.build(path)
    assert result["status"] == "INCOMPLETE"
    assert all(run["evaluation"] is None for run in result["runs"].values())


def test_synthetic_evaluation_cannot_be_a_real_round(tmp_path):
    evaluation = put(tmp_path / "fixture_metrics.json", {"status": "COMPLETED", "runs": {},
                     "protocol": "V0-strict-median-noninterpolated-AP-v1",
                     "require_run_provenance": False, "synthetic_fixture_only": True})
    index = tmp_path / "index.json"
    put(index, {"evaluation": evaluation})
    result = reporter.build(index)
    assert result["status"] == "INCOMPLETE"
    assert any("不能使用合成fixture结果" in error for error in result["errors"])


def test_changed_artifact_hash_rejected(tmp_path):
    index = tmp_path / "index.json"
    put(index, {})
    artifact = tmp_path / "input.json"
    reference = put(artifact, {"version": 1})
    artifact.write_text('{"version":2}\n')
    audit = reporter.Audit(index)
    assert audit.file(reference, "input") is None
    assert any("SHA256不匹配" in error for error in audit.errors)
