#!/usr/bin/env python3
"""Audit real round-one artifacts and write a Chinese report, never impute results.

Index schema (every file reference has path + sha256): protocol_freeze,
prediction_freeze, evaluation, common_s0, runs[run].inference_receipt; trained
runs additionally have training_receipt, telemetry_summary, training_ledger,
endpoint={path, files:{relative_file:sha256}}, and GRPO runs rollout_ledger.
Optional references: proxy_diagnostics, storage_audit, sync_receipt,
qualitative_failures. Relative paths resolve from the index directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from unsgrpo.evaluation import RUN_IDS, paired_bootstrap, validate_six_run_freeze

TRAINED = ("S0", "S+", "G-CAL", "G-RANK")
EXPECTED = {"S0": (256, 4096, 0), "S+": (256, 1024, 0),
            "G-CAL": (256, 1024, 4096), "G-RANK": (256, 1024, 4096)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def number(x, digits=4):
    return f"{x:.{digits}f}" if finite(x) else "未观测"


def percent(x):
    return f"{100*x:.2f}%" if finite(x) else "未观测"


def flag(command, key):
    try:
        return command[command.index(key) + 1]
    except (ValueError, IndexError):
        return None


class Audit:
    def __init__(self, index_path):
        self.index_path = Path(index_path).resolve()
        self.base = self.index_path.parent
        self.index = json.loads(self.index_path.read_text())
        self.errors, self.warnings, self.verified_files = [], [], {}

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def path(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.base / path).resolve()

    def file(self, reference, label, *, required=True):
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            if required:
                self.errors.append(f"{label}: 缺少带SHA256的文件引用")
            return None
        path = self.path(reference["path"])
        expected = reference.get("sha256")
        if not self.require(path.is_file(), f"{label}: 文件不存在 {path}"):
            return None
        if not self.require(isinstance(expected, str) and len(expected) == 64, f"{label}: 缺少SHA256"):
            return None
        actual = digest(path)
        if not self.require(actual == expected, f"{label}: SHA256不匹配 {path}"):
            return None
        self.verified_files[str(path)] = actual
        return path

    def json(self, reference, label, *, required=True):
        path = self.file(reference, label, required=required)
        if path is None:
            return None
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as e:
            self.errors.append(f"{label}: 无法读取JSON {type(e).__name__}: {e}")
            return None

    def tree(self, reference, label):
        if not isinstance(reference, dict) or not reference.get("path") or not reference.get("files"):
            self.errors.append(f"{label}: 缺少端点路径及逐文件哈希")
            return None
        root = self.path(reference["path"])
        checked = {}
        for name, expected in reference["files"].items():
            metadata = expected if isinstance(expected, dict) else {}
            expected = metadata.get("sha256") if metadata else expected
            p = (root / name).resolve()
            if not self.require(p.is_relative_to(root), f"{label}: 文件越过端点目录"):
                continue
            if self.file({"path": str(p), "sha256": expected}, f"{label}/{name}"):
                if "bytes" in metadata:
                    self.require(p.stat().st_size == metadata["bytes"], f"{label}/{name}: 文件字节数不一致")
                checked[name] = expected
        return {"path": str(root), "files": checked}


def audit_training(audit, run, entry, frozen, common, common_hash):
    before_errors = len(audit.errors)
    receipt = audit.json(entry.get("training_receipt"), f"{run}训练收据")
    summary = audit.json(entry.get("telemetry_summary"), f"{run}遥测摘要")
    ledger_path = audit.file(entry.get("training_ledger"), f"{run}训练账本")
    endpoint = audit.tree(entry.get("endpoint"), f"{run}端点")
    if endpoint:
        audit.require("adapter_config.json" in endpoint["files"] and
                      any(k.endswith(".safetensors") for k in endpoint["files"]), f"{run}: 端点缺少adapter配置或权重")
    result = {"status": "INCOMPLETE", "receipt": receipt, "summary": summary, "endpoint": endpoint}
    if not receipt or not summary or not ledger_path:
        return result
    audit.require(receipt.get("status") == "COMPLETED", f"{run}: 训练收据尚未审计为COMPLETED")
    audit.require(receipt.get("method_id") == run, f"{run}: 正式训练method_id不匹配")
    audit.require(receipt.get("mode") == "formal", f"{run}: smoke/技术检查不能代替正式训练")
    audit.require(receipt.get("gpu_training_verified") is True, f"{run}: 未确认真实GPU训练")
    audit.require(summary.get("status") in {"COMPLETED", "TRAINER_ENDED_REQUIRES_ACCEPTANCE"}, f"{run}: 遥测未正常结束")
    audit.require(summary.get("pending_uncommitted_rows") == 0, f"{run}: 仍有未提交输入")
    command = receipt.get("command", [])
    dataset = flag(command, "--dataset")
    if dataset:
        path = Path(dataset).resolve()
        audit.require(str(path) in frozen, f"{run}: 实际训练数据未列入协议冻结")
    else:
        audit.errors.append(f"{run}: 训练命令缺少明确数据路径")
    updates, exposures, completions = EXPECTED[run]
    audit.require(flag(command, "--max_steps") == str(updates), f"{run}: 实际命令更新数偏离256")
    audit.require(flag(command, "--seed") == "42", f"{run}: 实际训练seed偏离42")
    if run != "S0":
        audit.require(receipt.get("common_s0_sha256") == common_hash, f"{run}: 训练开始未绑定相同共同S0文件哈希")
        model = flag(command, "--model")
        audit.require(bool(model) and common and common.get("merged_model") and str(Path(model).resolve()) == str(Path(common["merged_model"]).resolve()),
                      f"{run}: 训练底座不是共同合并S0")
    elif endpoint and common:
        parent = common.get("sft_endpoint", common.get("sft_checkpoint"))
        audit.require(bool(parent) and str(Path(parent).resolve()) == endpoint["path"], f"S0: common_s0未绑定本次正式S0端点")
    pending, success_order, seen_completion_ids = [], [], set()
    observed, begin, end, parameter_changes = Counter(), None, None, 0
    policy_roles = Counter()
    for event in records(ledger_path):
        kind = event.get("event")
        if kind == "training_begin":
            audit.require(begin is None, f"{run}: 账本含多个训练起点，不是已验证的精确恢复")
            begin = event
            audit.require(event.get("audit_rows_sha256") in frozen.values(), f"{run}: 真实输入审计行未绑定冻结哈希")
            audit.require(event.get("non_adapter_requires_grad_false") is True, f"{run}: 底座冻结证据缺失")
        elif kind == "training_microbatch":
            pending.extend(event.get("records", []))
        elif kind == "policy_logps":
            role = event.get("role")
            policy_roles[role] += 1
            audit.require(event.get("finite") is True and event.get("requires_grad") == (role == "current"),
                          f"{run}: 当前/old/ref概率梯度或有限性异常")
        elif kind == "optimizer_update":
            if event.get("status") == "SKIPPED":
                pending = []
                observed["skipped_optimizer_steps"] += 1
                continue
            if event.get("status") != "SUCCESSFUL":
                continue
            evidence = event.get("optimizer", {})
            audit.require(evidence.get("call_returned") is True and evidence.get("skipped") is False,
                          f"{run}: 成功更新缺少真实optimizer调用证据")
            audit.require(evidence.get("gradient", {}).get("finite") is True, f"{run}: 更新梯度未验证有限")
            parameter_changes += int(evidence.get("parameter_changed") is True)
            actual_ids = []
            if run.startswith("G-"):
                groups = defaultdict(list)
                for row in pending:
                    groups[row.get("group_id")].append(row)
                    uid = row.get("rollout_request_id")
                    audit.require(isinstance(uid, str) and uid not in seen_completion_ids, f"{run}: 消费completion ID缺失或复用")
                    seen_completion_ids.add(uid)
                for group in groups.values():
                    audit.require(len(group) == 4, f"{run}: 成功更新含不完整G=4组")
                    for field in ("dataset_request_id", "media_tensor_sha256", "pseudo_scores_sha256", "clip_ids"):
                        audit.require(all(row.get(field) == group[0].get(field) for row in group), f"{run}: 同组{field}错位")
                    actual_ids.append(group[0].get("dataset_request_id"))
                observed["rollout_completions_consumed_in_successful_updates"] += len(pending)
            else:
                actual_ids = [row.get("dataset_request_id") for row in pending]
            audit.require(actual_ids == event.get("dataset_request_ids_in_order"), f"{run}: 更新曝光顺序与已消费输入不一致")
            expected_per_update = 16 if run == "S0" else 4
            audit.require(len(actual_ids) == expected_per_update, f"{run}: 单次实际输入曝光预算不一致")
            observed["successful_optimizer_updates"] += 1
            observed["prompt_exposures_in_successful_updates"] += len(actual_ids)
            observed["effective_loss_tokens"] += sum(row.get("loss_tokens", 0) for row in pending)
            observed["encoded_tokens_consumed"] += sum(row.get("encoded_tokens", 0) for row in pending)
            success_order.extend(actual_ids)
            pending = []
        elif kind == "training_end":
            end = event
            audit.require(event.get("frozen_sentinel_unchanged") is True, f"{run}: 冻结参数抽查不稳定")
        elif kind == "failure":
            audit.errors.append(f"{run}: 训练账本含失败事件")
    audit.require(begin is not None and end is not None and not pending, f"{run}: 账本缺少完整begin/end或仍有待消费行")
    expected_counts = {"successful_optimizer_updates": updates, "prompt_exposures_in_successful_updates": exposures,
                       "rollout_completions_consumed_in_successful_updates": completions}
    for key, wanted in expected_counts.items():
        audit.require(observed[key] == wanted, f"{run}: 实际{key}={observed[key]}，预登记为{wanted}")
        audit.require(summary.get("counts", {}).get(key, 0 if not completions and "rollout" in key else None) == observed[key],
                      f"{run}: 遥测摘要{key}与真实账本不一致")
    for key in ("effective_loss_tokens", "encoded_tokens_consumed"):
        audit.require(summary.get("counts", {}).get(key) == observed[key] and observed[key] > 0, f"{run}: 有效token账本缺失或不一致")
    audit.require(summary.get("dataset_request_id_order") == success_order, f"{run}: 摘要曝光顺序不一致")
    audit.require(summary.get("window_repetitions") == dict(Counter(success_order)), f"{run}: 摘要窗口重复数不一致")
    if parameter_changes == 0:
        specific = "ALL_GROUP_ADVANTAGES_ZERO_NO_POLICY_UPDATE"
        audit.require(run.startswith("G-") and receipt.get("allow_formal_zero_signal") is True
                      and receipt.get("negative_outcome") == specific and receipt.get("parameter_changed_updates") == 0,
                      f"{run}: 未变LoRA只能作为专项验收的正式GRPO全程零信号负结果，不能泛化为成功")
        acceptance = audit.json({"path": receipt.get("acceptance_path"), "sha256": receipt.get("acceptance_sha256")},
                                f"{run}全程零信号专项验收")
        if acceptance:
            audit.require(acceptance.get("status") == "COMPLETED" and acceptance.get("negative_outcome") == specific
                          and acceptance.get("allow_formal_zero_signal") is True and acceptance.get("parameter_changed_updates") == 0
                          and acceptance.get("evidence_files_sha256", {}).get(str(ledger_path)) == entry["training_ledger"]["sha256"],
                          f"{run}: 全程零信号验收未绑定当前真实消费账本")
        result["negative_outcome"] = receipt.get("negative_outcome")
    elif receipt.get("negative_outcome") is not None:
        audit.errors.append(f"{run}: 负结果标记与实际参数变化冲突")
    if run.startswith("G-"):
        audit.require(policy_roles["current"] > 0 and policy_roles["old_or_reference"] >= 2, f"{run}: 缺少current/old/ref概率实测")
        audit.require(summary.get("counts", {}).get("rollout_completions_generated") == completions, f"{run}: 实际生成rollout数不是4096")
        rollout_path = audit.file(entry.get("rollout_ledger"), f"{run}奖励账本")
        if rollout_path:
            rewards = list(records(rollout_path))
            uids = [r.get("rollout_request_id") for r in rewards]
            audit.require(len(uids) == completions and len(set(uids)) == len(uids) and set(uids) == seen_completion_ids,
                          f"{run}: 奖励账本与已生成/已消费completion集合不同")
            weight = 0.0 if run == "G-CAL" else 0.8
            for row in rewards:
                audit.require(row.get("rank_weight") == weight, f"{run}: 奖励配方与预登记不同")
                audit.require(dataset and row.get("dataset_sha256") == frozen.get(str(Path(dataset).resolve())), f"{run}: 奖励数据哈希不匹配")
            result["rollout_diagnostics"] = summarize_rollouts(rewards)
    costs = summary.get("stage_costs", {})
    audit.require(bool(costs) and any(v.get("cuda_measured") is True for v in costs.values()), f"{run}: 无真实GPU分阶段成本")
    audit.require(all(finite(v.get("wall_seconds")) and (not v.get("cuda_measured") or
                      (finite(v.get("peak_allocated_bytes")) and finite(v.get("peak_reserved_bytes")))) for v in costs.values()),
                  f"{run}: 分阶段成本缺项，不将缺失内存/耗时补0")
    audit.require(finite(summary.get("single_gpu_allocation_wall_hours")) and summary["single_gpu_allocation_wall_hours"] > 0,
                  f"{run}: 未观测训练GPU占用墙钟")
    result.update(observed_counts=dict(observed), parameter_changed_updates=parameter_changes,
                  unique_windows=len(set(success_order)), exact_resume_verified=summary.get("exact_resume_verified") is True)
    result["status"] = "COMPLETED" if len(audit.errors) == before_errors else "INCOMPLETE"
    return result


def summarize_rollouts(rows):
    parts = [r.get("reward", {}) for r in rows]
    legal = [p for p in parts if p.get("valid") is True]
    groups = {}
    for row in rows:
        key = (row.get("call_index"), row.get("group_in_call"))
        groups[key] = row.get("group_reward_variance_population")
    variances = [v for v in groups.values() if finite(v)]
    mean = lambda values: float(np.mean(values)) if values and all(finite(x) for x in values) else None
    return {"n_completions": len(rows), "legal_fraction": len(legal) / len(rows) if rows else None,
            "mean_total_reward": mean([p.get("total") for p in parts]),
            "ranking_among_legal": mean([p.get("ranking") for p in legal]),
            "calibration_among_legal": mean([p.get("calibration") for p in legal]),
            "n_groups": len(groups), "mean_group_reward_variance": mean(variances),
            "zero_variance_group_fraction": sum(v == 0 for v in variances) / len(variances) if variances else None,
            "truncated_true": sum(r.get("is_truncated") is True for r in rows),
            "truncation_unknown": sum(r.get("is_truncated") is None for r in rows)}


def build(index_path):
    audit = Audit(index_path)
    index = audit.index
    protocol = audit.json(index.get("protocol_freeze"), "协议冻结")
    common = audit.json(index.get("common_s0"), "共同S0")
    evaluation = audit.json(index.get("evaluation"), "独立评价结果")
    freeze_path = audit.file(index.get("prediction_freeze"), "六组预测冻结")
    project = Path(__file__).resolve().parents[1]
    frozen = {}
    if protocol:
        audit.require(protocol.get("status") == "FROZEN", "正式协议尚未FROZEN")
        for path, expected in protocol.get("files", {}).items():
            actual_path = Path(path)
            actual_path = actual_path if actual_path.is_absolute() else project / actual_path
            if audit.file({"path": str(actual_path), "sha256": expected}, "冻结输入/代码"):
                frozen[str(actual_path.resolve())] = expected
        audit.require(bool(frozen), "协议冻结没有可验证文件")
        technical_path = protocol.get("technical_acceptance")
        if technical_path:
            p = Path(technical_path); p = p if p.is_absolute() else project / p
            if audit.require(str(p.resolve()) in frozen, "GPU技术验收收据未冻结"):
                technical = json.loads(p.read_text())
                required = ("generation_reward_logps_backward_optimizer_save_reload", "same_group_media_q_id",
                            "assistant_only_sft_no_grpo_answers", "strict_json_equal_reward_truncation",
                            "finite_lora_gradients_parameter_update_frozen_backbone", "s0_bf16_tensor_merge_common_endpoint",
                            "fixed_s0_reference_after_update", "old_ref_reward_advantage_detached",
                            "completion_only_loss_eos_lengths_std_sampling", "stage_memory_time_recorded",
                            "formal_rollout_concurrency_verified")
                audit.require(technical.get("status") == "COMPLETED" and technical.get("gpu_training_verified") is True
                              and all(technical.get("checks", {}).get(k) is True for k in required), "真实GPU技术闭环证据不完整")
        else:
            audit.errors.append("协议缺少真实GPU技术验收收据")
        for run, (updates, exposures, completions) in EXPECTED.items():
            spec = protocol.get("matrix", {}).get(run, {})
            audit.require(spec.get("updates") == updates and spec.get("prompt_exposures_per_update") == exposures // updates
                          and spec.get("group_size") == (4 if completions else 1) and spec.get("seed") == 42,
                          f"{run}: 冻结矩阵不是原256步/指定曝光的首轮")
    common_hash = index.get("common_s0", {}).get("sha256")
    if common:
        audit.require(common.get("status") == "COMPLETED" and common.get("merge_validation_passed") is True
                      and common.get("acceptance_basis") == "verified_bf16_tensor_merge_common_s0"
                      and common.get("premerge_functional_equivalence_claimed") is False,
                      "共同S0必须是已逐位验证的实际BF16合并端点，不得将初拟logp容差失败改成通过")
        verification = audit.json(common.get("merge_bitwise_verification"), "共同S0逐位合并验收")
        reference_check = audit.json(common.get("reference_verification"), "共同S0固定参考验收")
        processor = audit.json(common.get("processor_restore"), "共同S0 processor恢复验收")
        snapshot = audit.json(common.get("snapshot"), "正式S0合并前概率快照")
        compared = audit.json(common.get("merge_comparison"), "正式S0合并后原始概率比较")
        if verification:
            audit.require(verification.get("status") == "COMPLETED" and verification.get("merge_validation_passed") is True
                          and verification.get("acceptance_basis") == "verified_bf16_tensor_merge_common_s0"
                          and verification.get("merge_formula_verified_all_tensors") is True
                          and verification.get("aggregate", {}).get("different_from_exact_bf16_elements") == 0,
                          "共同S0缺少完整逐位BF16合并零差异证据")
        if reference_check:
            audit.require(reference_check.get("status") == "COMPLETED" and reference_check.get("gpu_reference_update_verified") is True
                          and reference_check.get("reference_invariance", {}).get("exact_equal") is True,
                          "共同S0固定参考在新adapter更新后未保持不变")
        if processor:
            audit.require(processor.get("status") == "COMPLETED" and processor.get("fixed_train_row_encode_verified") is True
                          and processor.get("restored_entire_processor_package_byte_identical") is True
                          and processor.get("model_weights_modified") is False, "共同S0视觉接口未通过恢复验收")
        if snapshot and compared:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from verify_bf16_merge import logp_gap
                measured = logp_gap(snapshot, compared)
                audit.require(common.get("merge_logp_within_initial_tolerance") == measured["within_initial_tolerance"]
                              and common.get("merge_logp_gap") == measured, "共同S0量化gap或原阈状态与原始概率比较不一致")
            except (KeyError, TypeError, ValueError) as e:
                audit.errors.append(f"共同S0原始量化gap无法复核: {e}")
        weights = common.get("merged_model_files", common.get("model_file_hashes", common.get("model_files")))
        model_tree = audit.tree({"path": common.get("merged_model"), "files": weights}, "共同合并S0模型")
        audit.require(model_tree and "config.json" in model_tree["files"] and
                      any(name.endswith(".safetensors") for name in model_tree["files"]), "共同S0缺少完整模型配置或权重身份")
    audited_predictions = None
    if freeze_path:
        try:
            audited_predictions = validate_six_run_freeze(freeze_path)
        except Exception as e:
            audit.errors.append(f"六组冻结预测不完整: {type(e).__name__}: {e}")
    if evaluation:
        audit.require(evaluation.get("status") == "COMPLETED", "评价尚未COMPLETED")
        audit.require(evaluation.get("require_run_provenance") is True and evaluation.get("synthetic_fixture_only") is False,
                      "评价必须绑定真实六组运行归因，不能使用合成fixture结果")
        audit.require(evaluation.get("protocol") == "V0-strict-median-noninterpolated-AP-v1", "评价协议不是预登记V0")
        audit.require(set(evaluation.get("runs", {})) == set(RUN_IDS), "评价结果不是完整六组")
        if audited_predictions:
            audit.require(evaluation.get("prediction_freeze_sha256") == audited_predictions["freeze_sha256"], "评价未绑定当前六组冻结预测")
            audit.require(evaluation.get("manifest_sha256") == audited_predictions["manifest_sha256"], "评价manifest哈希不一致")
    run_results = {}
    audit.require(set(index.get("runs", {})) == set(RUN_IDS), "产物索引必须恰好包含六组")
    for run in RUN_IDS:
        before_errors = len(audit.errors)
        entry = index.get("runs", {}).get(run, {})
        inference = audit.json(entry.get("inference_receipt"), f"{run}推理收据")
        row = {"inference": inference, "evaluation": (evaluation or {}).get("runs", {}).get(run)}
        if run in TRAINED:
            row["training"] = audit_training(audit, run, entry, frozen, common, common_hash)
        if inference:
            audit.require(inference.get("status") == "COMPLETED" and inference.get("run_id") == run, f"{run}: 推理未实际COMPLETED或ID不符")
            audit.require(inference.get("human_annotations_read") is False, f"{run}: 推理标签隔离未验证")
            for key in ("inputs_sha256", "manifest_sha256", "pseudo_sha256"):
                audit.require(inference.get(key) in frozen.values(), f"{run}: 推理{key}未绑定已冻结文件")
            if audited_predictions:
                pred = audited_predictions["runs"][run]
                audit.require(inference.get("predictions_sha256") == pred["sha256"], f"{run}: 推理收据与冻结预测哈希不一致")
                audit.require((row["evaluation"] or {}).get("prediction_sha256") == pred["sha256"], f"{run}: 评价预测哈希不一致")
                audit.require(inference.get("n_videos") == len(audited_predictions["test_videos"]), f"{run}: 推理视频覆盖数不符")
            if run != "B0":
                audit.require(inference.get("greedy") is True and inference.get("max_completion_tokens") == 128, f"{run}: 统一greedy/输出预算不符")
                audit.require(inference.get("input_audit_sha256") in frozen.values(), f"{run}: 实际推理输入审计未冻结")
                audit.require(inference.get("inference_input_max_length") == 8192 and bool(inference.get("truncation_rule")), f"{run}: 缺少最大编码长度/截断处理声明")
                model_tree = audit.tree(inference.get("model"), f"{run}实际推理模型")
                adapter_tree = audit.tree(inference.get("adapter"), f"{run}实际推理adapter") if inference.get("adapter") else None
                if run == "B1":
                    audit.require(adapter_tree is None, "B1: 零样本推理不能加载adapter")
                if run in TRAINED:
                    audit.require(inference.get("common_s0_sha256") == common_hash, f"{run}: 推理未绑定共同S0")
                    audit.require(inference.get("training_receipt", {}).get("sha256") == entry.get("training_receipt", {}).get("sha256"), f"{run}: 推理未绑定相应正式训练收据")
                    audit.require(model_tree and common and model_tree["path"] == str(Path(common["merged_model"]).resolve()), f"{run}: 推理底座不是共同S0")
                    if run == "S0":
                        audit.require(adapter_tree is None, "S0: 应直接推理共同合并S0")
                    else:
                        endpoint = row["training"].get("endpoint")
                        audit.require(adapter_tree and endpoint and adapter_tree["path"] == endpoint["path"] and
                                      all(adapter_tree["files"].get(k) == v for k, v in endpoint["files"].items()
                                          if Path(k).suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}),
                                      f"{run}: 推理adapter不是本分支审计端点")
                for field in ("proxy_legal_rate", "proxy_mean_reward", "cross_window_mean_range", "wall_seconds", "gpu_hours", "peak_cuda_allocated_bytes"):
                    audit.require(finite(inference.get(field)), f"{run}: 缺少已观测推理字段{field}")
                audit.require(finite(inference.get("proxy_legal_rate")) and 0 <= inference["proxy_legal_rate"] <= 1, f"{run}: 合法率范围错误")
            if row["evaluation"]:
                e = row["evaluation"]
                per_video = e.get("per_video", [])
                audit.require(bool(per_video) and e.get("n_videos") == len(per_video), f"{run}: 缺少逐视频结果")
                if audited_predictions:
                    expected = {v["video_id"]: v for v in audited_predictions["test_videos"]}
                    audit.require(len(per_video) == len(expected) and {v.get("video_id") for v in per_video} == set(expected), f"{run}: 逐视频评价集合与冻结manifest不同")
                    audit.require(all(v.get("source_group") == expected.get(v.get("video_id"), {}).get("source_group") for v in per_video), f"{run}: 逐视频评价来源组不一致")
                if per_video and all(finite(v.get("mean_ap")) and 0 <= v["mean_ap"] <= 1 for v in per_video):
                    audit.require(np.isclose(e.get("video_macro_ap", np.nan), np.mean([v["mean_ap"] for v in per_video]), atol=1e-12), f"{run}: 宏平均与逐视频AP不一致")
                else:
                    audit.errors.append(f"{run}: 逐视频AP缺失或非有限")
        row["status"] = "COMPLETED" if len(audit.errors) == before_errors else "INCOMPLETE"
        run_results[run] = row
    valid_inference = [r["inference"] for r in run_results.values() if r["inference"]]
    if valid_inference:
        for key in ("inputs_sha256", "manifest_sha256", "pseudo_sha256"):
            audit.require(len({r.get(key) for r in valid_inference}) == 1, f"六组推理{key}不一致")
        visuals = [r.get("visual_environment") for r in valid_inference]
        audit.require(all(isinstance(v, dict) and v and v == visuals[0] for v in visuals), "六组推理视觉预算未完整记录或不一致")
    paired = None
    if evaluation and set(evaluation.get("runs", {})) == set(RUN_IDS):
        try:
            a = evaluation["runs"]["G-RANK"]["per_video"]
            b = {v["video_id"]: v for v in evaluation["runs"]["S+"]["per_video"]}
            audit.require(set(b) == {v["video_id"] for v in a}, "主比较逐视频集合不一致")
            paired = paired_bootstrap([v["mean_ap"] - b[v["video_id"]]["mean_ap"] for v in a],
                                      source_groups=[v["source_group"] for v in a], n_boot=10000, seed=42)
            declared = evaluation.get("primary_paired_difference", {})
            for key in ("mean", "video_macro_mean", "ci_low", "ci_high", "n_videos", "n_source_groups", "n_boot", "seed"):
                audit.require(finite(declared.get(key)) and np.isclose(declared[key], paired[key], atol=1e-12), f"主比较bootstrap字段{key}复核失败")
        except (KeyError, TypeError, ValueError) as e:
            audit.errors.append(f"主比较无法复核: {e}")
    optional = {name: audit.json(index.get(name), name, required=False)
                for name in ("proxy_diagnostics", "storage_audit", "sync_receipt", "qualitative_failures")}
    archive = audit.json(index.get("archive_receipt"), "已完成端点归档")
    optional["archive_receipt"] = archive
    if archive:
        audit.require(archive.get("status") == "COMPLETED" and archive.get("local_sources_deleted") is False
                      and archive.get("files"), "端点归档未完成逐文件核验或本地原件保留状态不明确")
        audit.file(archive.get("archive_manifest"), "归档文件清单")
        if archive.get("files"):
            sizes = [row.get("bytes") for row in archive["files"]]
            audit.require(all(finite(size) and size >= 0 for size in sizes) and
                          archive.get("total_bytes") == sum(sizes), "归档字节数与已核验清单不一致")
        audit.require(finite(archive.get("wall_seconds")), "归档代价尚未观测，不补0")
    historical_merge_failure = None
    if protocol:
        registered = protocol.get("protocol", {}).get("merge_numeric_acceptance", {})
        initial = registered.get("initial_failure", {}).get("comparison")
        if initial:
            historical_merge_failure = audit.json(initial, "预冻结原初合并容差失败")
    preparation = None
    if audited_predictions:
        prepared = list(records(audited_predictions["manifest_path"]))
        times = [r.get("seconds") for r in prepared]
        preparation = {"n_videos": len(prepared),
                       "sum_per_video_wall_seconds": sum(times) if times and all(finite(t) for t in times) else None,
                       "scope": "每视频真实解码和CLIP墙钟之和；不含共享模型加载、UMAP和技术训练；不能解释为GPU核函数时间"}
    result = {"status": "INCOMPLETE" if audit.errors else "COMPLETED",
              "created_hong_kong": dt.datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "index_path": str(audit.index_path), "index_sha256": digest(audit.index_path),
              "reporter_sha256": digest(__file__), "errors": list(dict.fromkeys(audit.errors)),
              "warnings": audit.warnings, "verified_files": audit.verified_files,
              "protocol": "V0-strict-median-noninterpolated-AP-v1", "V1": (evaluation or {}).get("V1"),
              "git_sync_policy": "ON_DEMAND_USER_REQUEST_ONLY_NO_AUTOMATION",
              "common_s0_sha256": common_hash, "runs": run_results,
              "common_s0": common,
              "preparation": preparation,
              "historical_merge_failure": historical_merge_failure,
              "archive_controller_wall_seconds": index.get("archive_controller_wall_seconds"),
              "primary_paired_difference": paired, "optional": optional}
    result["recommendations"] = recommendations(result)
    return result


def recommendations(report):
    if report["status"] != "COMPLETED":
        return ["暂缓成效判断与扩大投入。先补齐下列证据缺口，完成预登记六组固定终点；技术smoke、计划曝光和缺失成本均不能代替正式结果。"]
    runs = report["runs"]
    ap = {k: r["evaluation"]["video_macro_ap"] for k, r in runs.items()}
    paired = report["primary_paired_difference"]
    advice = []
    if ap["B0"] >= max(ap[k] for k in TRAINED):
        advice.append("直接代理达到或超过全部学生端点：先查输入细节、局部输入能否预测全视频百分位、尺度绑定及训练有效性，暂不扩大GRPO预算。")
    if ap["S0"] > ap["B0"] and paired["ci_low"] <= 0 <= paired["ci_high"]:
        advice.append("共同S0超过直接代理，而G-RANK相对继续SFT的配对区间包含0：学生学习有用，GRPO额外价值尚未成立，优先保留成本更低的训练配方。")
    rank = runs["G-RANK"]["inference"].get("proxy_mean_reward")
    plus = runs["S+"]["inference"].get("proxy_mean_reward")
    if finite(rank) and finite(plus) and rank > plus and ap["G-RANK"] < ap["S+"]:
        advice.append("G-RANK代理奖励上升而人类V0下降：存在代理错位候选。重点复核常见背景、簇典型性、去并列和局部观测缺失，不用更长训练掩盖背离。")
    if paired["ci_low"] > 0:
        advice.append("单seed上G-RANK相对S+的来源组bootstrap区间高于0：可有限续投，下一阶段先做3个完整训练seed（各自训练S0）、独立来源和直接排序/回归学习对照，再决定扩大规模。")
    elif paired["ci_high"] < 0:
        advice.append("本轮G-RANK在配对来源组分析中落后S+：当前证据不支持追加GRPO计算；保留失败结果，先定位奖励与输入问题。")
    else:
        advice.append("主比较区间包含0，尚不足以确认GRPO收益。先进行低成本误差归因；是否增加完整seed应结合效果幅度及下表增量成本，不能只依据代理奖励。")
    h_rank = runs["G-RANK"]["training"]["summary"].get("single_gpu_allocation_wall_hours")
    h_plus = runs["S+"]["training"]["summary"].get("single_gpu_allocation_wall_hours")
    if finite(h_rank) and finite(h_plus) and h_plus > 0:
        advice.append(f"G-RANK续训占用GPU墙钟为S+的{h_rank/h_plus:.2f}倍；来源组平均V0差为{100*paired['mean']:.3f}个百分点。收益与成本需共同决定投入，不把相同曝光解释为相同计算。")
    return advice


def markdown(report):
    lines = ["# 无人工高光标注首轮实验报告", "", f"状态：**{report['status']}**。生成时间：{report['created_hong_kong']}（香港）。", ""]
    lines += report["recommendations"] + [""]
    common = report.get("common_s0") or {}
    gap = common.get("merge_logp_gap") or {}
    if common:
        within = common.get("merge_logp_within_initial_tolerance")
        lines += ["## 共同S0的实际定义", "",
                  "共同起点是完整逐位验证后的实际BF16量化合并模型。S0推理与后三个新LoRA分支使用完全相同的这个模型；合并前SFT adapter端点另行保留。",
                  f"合并前后固定答案token logp平均绝对差{number(gap.get('mean_abs'),6)}、最大绝对差{number(gap.get('max_abs'),6)}。原0.03/0.25初拟容差诊断{'通过' if within is True else '未通过' if within is False else '未观测'}，原始比较状态为{gap.get('original_compare_status','未观测')}，原FAILED文件不修改、不抬高阈值。",
                  "正式验收依据为全部适配权重精确等于BF16(base+2BA)、非LoRA张量逐位不变、processor恢复一致及更新新adapter后固定参考完全不变；不声称量化合并前后策略函数等价。", ""]
        historical = report.get("historical_merge_failure") or {}
        numeric = historical.get("merge_logp_comparison") or {}
        if historical:
            lines += [f"冻结前最初技术比较仍为{historical.get('status','未观测')}：平均绝对差{number(numeric.get('mean_abs'),6)}、最大差{number(numeric.get('max_abs'),6)}。这份历史证据支持预冻结时明确共同端点定义，不能代替当前正式S0自身的逐位验收。", ""]
    if report["errors"]:
        lines += ["## 未完成的证据", ""] + [f"- {x}" for x in report["errors"]] + [""]
    lines += ["## 六组结果与实际成本", "", "V0是内部诊断：真实区间重叠映射到原生帧，每位标注者严格高于自身中位数二值化，非插值AP先按人、再按视频平均。它不是可跨论文比较的TVSum top-5 mAP。缺失值写“未观测”。", "",
              "|方法|审计状态|V0视频宏AP|合法率|代理奖励（含非法）|训练成功更新/曝光/rollout|独特窗口|续训GPU墙钟h|推理GPU墙钟h|推理峰值GiB|",
              "|---|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for run in RUN_IDS:
        row = report["runs"][run]; e = row.get("evaluation") or {}; i = row.get("inference") or {}
        training = row.get("training") or {}; s = training.get("summary") or {}; c = training.get("observed_counts")
        consumed = f"{c.get('successful_optimizer_updates', '未观测')}/{c.get('prompt_exposures_in_successful_updates', '未观测')}/{c.get('rollout_completions_consumed_in_successful_updates', 0)}" if c else ("0/0/0（无训练）" if run in ("B0", "B1") else "未观测")
        peak = i.get("peak_cuda_allocated_bytes")
        lines.append(f"|{run}|{row['status']}|{percent(e.get('video_macro_ap'))}|{percent(i.get('proxy_legal_rate')) if run!='B0' else '不适用'}|{number(i.get('proxy_mean_reward')) if run!='B0' else '不适用'}|{consumed}|{training.get('unique_windows','不适用' if run in ('B0','B1') else '未观测')}|{number(s.get('single_gpu_allocation_wall_hours')) if run in TRAINED else '0（无训练）'}|{number(i.get('gpu_hours'))}|{number(peak/1024**3 if finite(peak) else None,3)}|")
    lines += ["", "S0是共同训练成本，后三分支的训练小时仅为各自增量，不含S0；比较完整成本时应分别加上同一S0。GPU小时表示单卡占用墙钟，不代表GPU核函数利用率。有效token、重复曝光顺序、逐阶段内存/时间和端点逐文件哈希见JSON审计产物。", ""]
    lines += ["|训练组|实测阶段|调用数|墙钟秒|峰值已分配GiB|峰值保留GiB|", "|---|---|---:|---:|---:|---:|"]
    for run in TRAINED:
        summary = (report["runs"][run].get("training") or {}).get("summary") or {}
        for stage, cost in sorted(summary.get("stage_costs", {}).items()):
            allocated, reserved = cost.get("peak_allocated_bytes"), cost.get("peak_reserved_bytes")
            lines.append(f"|{run}|{stage}|{cost.get('calls','未观测')}|{number(cost.get('wall_seconds'),2)}|{number(allocated/1024**3 if finite(allocated) else None,3)}|{number(reserved/1024**3 if finite(reserved) else None,3)}|")
    prep = report.get("preparation") or {}
    lines += ["", f"共享真实视频预处理：{prep.get('n_videos','未观测')}个视频，逐视频墙钟之和{number(prep.get('sum_per_video_wall_seconds'),2)}秒（含解码和CLIP，不含共享加载）。"]
    proxy = report["optional"].get("proxy_diagnostics") or {}
    lines += [f"训练域UMAP/聚类及代理构建墙钟：{number(proxy.get('elapsed_seconds'),2)}秒。技术检查、安装及下载成本不包含在续训表中，应查看各自独立收据。", ""]
    paired = report.get("primary_paired_difference")
    if paired:
        lines += ["## 主比较与逐视频结果", "", f"G-RANK − S+：来源组等权平均差 **{100*paired['mean']:.3f}个百分点**，95%百分位bootstrap区间 [{100*paired['ci_low']:.3f}, {100*paired['ci_high']:.3f}]。{paired['n_videos']}个视频、{paired['n_source_groups']}个来源组，bootstrap {paired['n_boot']}次，seed={paired['seed']}。", "", "单训练seed的来源组bootstrap只反映测试来源抽样，不包含训练随机性；窗口不是独立统计样本。", "",
                  "|视频|来源组|B0|B1|S0|S+|G-CAL|G-RANK|G-RANK − S+（百分点）|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        maps = {k: {v["video_id"]: v for v in ((r.get("evaluation") or {}).get("per_video") or [])} for k, r in report["runs"].items()}
        for vid in sorted(maps["G-RANK"]):
            r = maps["G-RANK"][vid]
            values = [maps[k].get(vid, {}).get("mean_ap") for k in RUN_IDS]
            diff = values[-1] - values[3] if finite(values[-1]) and finite(values[3]) else None
            lines.append("|" + "|".join([vid, str(r["source_group"]), *[percent(v) for v in values], number(100*diff if finite(diff) else None, 3)]) + "|")
    lines += ["", "## 格式、稳定性与失败候选", "", "|方法|合法回答排序奖励|合法回答校准奖励|合法回答常数输出率|重复片段跨窗平均分差|推理达到长度上限比例|训练零方差组比例|", "|---|---:|---:|---:|---:|---:|---:|"]
    for run in RUN_IDS[1:]:
        row = report["runs"][run]; i = row.get("inference") or {}; d = (row.get("training") or {}).get("rollout_diagnostics") or {}
        lines.append(f"|{run}|{number(i.get('proxy_mean_ranking_among_legal'))}|{number(i.get('proxy_mean_calibration_among_legal'))}|{percent(i.get('constant_output_fraction_among_legal'))}|{number(i.get('cross_window_mean_range'))}|{percent(i.get('truncated_window_fraction'))}|{percent(d.get('zero_variance_group_fraction')) if run.startswith('G-') else '不适用'}|")
    lines += ["", "合法回答上的奖励是条件诊断，不能替代含非法窗口的主指标。非法窗口以全零分计入重叠平均分母。排序上涨可能来自去并列、校准或格式修复；本轮S+与GRPO同时改变目标和训练方式，不能据此证明强化学习必不可少。", ""]
    for run, row in report["runs"].items():
        i = row.get("inference") or {}; t = row.get("training") or {}; d = t.get("rollout_diagnostics") or {}
        if finite(i.get("proxy_legal_rate")) and i["proxy_legal_rate"] < 1:
            lines.append(f"- {run}有{100*(1-i['proxy_legal_rate']):.2f}%非法窗口，已纳入主指标；应查看独立windows.jsonl中的实际回答。")
        if t and t.get("parameter_changed_updates") == 0:
            if t.get("negative_outcome") == "ALL_GROUP_ADVANTAGES_ZERO_NO_POLICY_UPDATE":
                lines.append(f"- {run}已完成固定预算，但经专项原账本验收确认为全程零组优势、零梯度/KL、参数未变的合法负结果；保留其实际成本与评价，不把它称作有用策略更新。")
            else:
                lines.append(f"- {run}未观测到任何参数哈希变化且缺少专项零信号验收；不能只看optimizer调用次数。")
        if d.get("truncation_unknown", 0):
            lines.append(f"- {run}有{d['truncation_unknown']}条训练回答的截断元数据未知；不把未知写成未截断。")
    qualitative = report["optional"].get("qualitative_failures")
    if qualitative:
        lines.append("定性复核文件已绑定SHA256，具体案例见JSON产物及索引中的qualitative_failures文件。")
    else:
        lines.append("未提供已核验画面的定性案例文件；上述为数值触发的失败候选，不能冒称已确认背景偏好或遗漏了某个具体动作。")
    lines += ["", "## 可复核性与限制", "", f"产物索引SHA256：`{report['index_sha256']}`；共同S0身份文件SHA256：`{report.get('common_s0_sha256') or '未提供'}`。所有已验证输入及端点文件哈希在JSON报告中，训练曝光来自真实optimizer提交账本。", "", "V1的真实benchmark片段时间轴和参考split对齐尚未通过，因此不作跨论文数值比较。测试标签只在六组预测冻结后进入独立评价；若以后根据本轮测试结果改方法，本集合应视为开发证据，下一阶段须保留独立来源。", "", "未验证优化器、scheduler、RNG与数据游标的精确恢复；重新加载adapter只属于权重恢复，不是精确续训。", ""]
    sync = report["optional"].get("sync_receipt")
    lines += [f"最近Git核验/同步状态：{sync.get('status') if sync else '未提供实际收据'}。用户已取消自动每日同步，仅在每次明确要求时执行按需同步；未创建任何同步定时器，也不因训练完成自行提交/推送。身份缺失仅记录，不反复催问。", ""]
    storage = report["optional"].get("storage_audit")
    archive = report["optional"].get("archive_receipt")
    if archive:
        lines += [f"端点归档：{archive.get('archive_root','路径见归档收据')}；{len(archive.get('files',[]))}个逐SHA256核验文件，{number(archive.get('total_bytes')/1024**3 if finite(archive.get('total_bytes')) else None,3)} GiB。复制与核验墙钟{number(archive.get('wall_seconds'),2)}秒，控制器归档阶段（含再次核验）{number(report.get('archive_controller_wall_seconds'),2)}秒；该成本单列，不计入方法训练成本。本地原件保留，原视频、人工标注内容与环境不复制。", ""]
    if storage:
        for name, entry in storage.get("filesystems", {}).items():
            lines.append(f"- {name}盘：总量{entry['total_bytes']/1024**3:.2f} GiB，可用{entry['free_bytes']/1024**3:.2f} GiB；审计状态{'通过' if storage.get('ok') else '未通过'}。")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()
    if args.out_md.exists() or args.out_json.exists():
        p.error("use new report paths; existing reports are immutable")
    try:
        report = build(args.index)
    except Exception as e:
        report = {"status": "INCOMPLETE", "error": f"{type(e).__name__}: {e}"}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(f"# 首轮实验报告\n\n状态：INCOMPLETE。无法完成证据审计：{report['error']}\n")
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(2)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown(report))
    print(json.dumps({"status": report["status"], "n_errors": len(report["errors"]), "out_md": str(args.out_md)}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "COMPLETED" else 2)


if __name__ == "__main__":
    main()
