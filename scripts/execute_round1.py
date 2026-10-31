#!/usr/bin/env python3
"""Sequential, fail-closed execution of the frozen formal six-arm experiment.

Default is --plan: no training, state creation, label extraction, or Git action.
--execute requires real FROZEN GPU acceptance. --resume resumes this controller's
completed stages only; it never resumes/restarts a failed or interrupted trainer.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("UNSGRPO_WORK_DIR", "/root/autodl-tmp/unsgrpo-work"))
STORE = Path(os.environ.get("UNSGRPO_STORAGE_DIR", "/root/autodl-fs/unsgrpo"))
METHODS = ("B0", "B1", "S0", "S+", "G-CAL", "G-RANK")
TRAINED = ("S0", "S+", "G-CAL", "G-RANK")
SLUG = {"S0": "S0", "S+": "SPLUS", "G-CAL": "G_CAL", "G-RANK": "G_RANK"}
CODE = ("execute_round1.py", "run.sh", "training_telemetry.py", "reward_plugin.py",
        "accept_training_run.py", "verify_checkpoint.py", "restore_processor.py",
        "verify_bf16_merge.py", "diagnose_merge.py",
        "inference.py", "freeze_predictions.py", "convert_tvsum_labels.py",
        "evaluate.py", "report_round1.py", "storage_audit.py", "archive_round1.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def now():
    value = datetime.now(timezone.utc)
    return {"utc": value.isoformat(), "hong_kong": value.astimezone(ZoneInfo("Asia/Hong_Kong")).isoformat()}


def atomic(path, value, *, fresh=False):
    path = Path(path)
    require(not fresh or not path.exists(), f"Refusing to overwrite an existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def inventory(root):
    root = Path(root).resolve()
    require(root.is_dir(), f"Model/endpoint directory missing: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            require(not path.is_symlink(), f"Unexpected symlink in exported endpoint: {path}")
            result[str(path.relative_to(root))] = sha(path)
    require(result, f"Empty model/endpoint inventory: {root}")
    return result


def final_checkpoint(output):
    """Only one actual version-0 final checkpoint is accepted, never 'latest'."""
    output = Path(output).resolve()
    candidates = sorted(output.glob("*/checkpoint-256"))
    require(len(candidates) == 1, f"Expected exactly one versioned checkpoint-256 under {output}")
    checkpoint = candidates[0]
    require(not checkpoint.is_symlink() and not checkpoint.parent.is_symlink()
            and checkpoint.resolve().is_relative_to(output), "Final checkpoint must remain inside its actual run directory")
    require(re.fullmatch(r"v0(?:-\d{8}-\d{6})?", checkpoint.parent.name),
            f"Unexpected output version; no automatic v1/latest selection: {checkpoint}")
    require((checkpoint / "adapter_model.safetensors").is_file() and
            (checkpoint / "adapter_config.json").is_file(), "Final checkpoint lacks LoRA weights/config")
    require(read(checkpoint / "trainer_state.json").get("global_step") == 256, "Final checkpoint is not update 256")
    return checkpoint


class Driver:
    def __init__(self, args):
        self.args = args
        self.root = args.state_dir.resolve()
        self.prefix = self.root.name
        require(re.fullmatch(r"[A-Za-z0-9_-]+", self.prefix), "State directory name must be a simple unique run label")
        require(self.root.is_relative_to((WORK / "runs").resolve()), "Controller state must be under external work/runs")
        # Resolving a venv's python symlink would select the system interpreter
        # and silently lose the independent environment's installed packages.
        self.python = os.path.abspath(args.python)
        self.common = PROJECT / "reports/common_s0.json"
        self.index = PROJECT / "reports/round1_artifacts.json"
        self.markdown = PROJECT / "reports" / f"ROUND1_RESULTS_{self.prefix}.md"
        self.result_json = PROJECT / "reports" / f"round1_results_{self.prefix}.json"
        self.runs = {m: WORK / "runs" / f"{self.prefix}__{SLUG[m]}" for m in TRAINED}
        self.merge_run = WORK / "runs" / f"{self.prefix}__MERGE"
        self.output_root = args.checkpoints.resolve()
        self.merged = self.output_root / self.merge_run.name
        self.inference = {m: self.root / "inference" / m.replace("+", "PLUS") for m in METHODS}
        self.checks = self.root / "checks"
        self.prediction_freeze = self.root / "predictions.freeze.json"
        self.annotations = STORE / "datasets/labels_eval_only" / self.prefix
        self.evaluation = self.root / "evaluation_V0.json"
        self.state_path = self.root / "state.json"
        self.state, self.frozen = None, {}
        self.active_process = None
        self.storage_final = self.root / "storage_final.json"
        self.archive_root = (STORE / "archives" / self.prefix).resolve()
        self.archive_receipt = self.root / "archive_receipt.json"

    def digest(self, path):
        # Real file content is rehashed. Timestamp caches can miss same-size
        # changes on mounts whose mtime/ctime resolution is coarse.
        return sha(Path(path).resolve())

    def validate_freeze(self):
        require(self.args.freeze.is_file(), "Real reports/freeze.json is missing; technical smoke is insufficient")
        freeze = read(self.args.freeze)
        require(freeze.get("status") == "FROZEN" and freeze.get("human_annotations_read_for_training") is False,
                "Formal protocol is not FROZEN with the no-human-training-label boundary")
        self.frozen = {}
        for name, expected in freeze.get("files", {}).items():
            path = Path(name); path = path if path.is_absolute() else PROJECT / path
            path = path.resolve()
            require(path.is_file() and self.digest(path) == expected, f"Frozen file changed/missing: {path}")
            self.frozen[str(path)] = expected
        required = [PROJECT / "scripts" / name for name in CODE]
        required += [PROJECT / "storage-env.sh", PROJECT / "src/unsgrpo/core.py", PROJECT / "src/unsgrpo/data.py",
                     PROJECT / "src/unsgrpo/evaluation.py", PROJECT / "configs/environment.freeze.txt",
                     PROJECT / "configs/round1.candidate.json", self.args.input_audit,
                     self.args.manifest, self.args.pseudo, self.args.annotations_zip]
        required += [self.args.data / name for name in ("train_sft.jsonl", "train_grpo.jsonl", "eval.jsonl")]
        require(all(str(p.resolve()) in self.frozen for p in required),
                "Freeze does not contain every executed source/config/data file and original annotation ZIP")
        technical = Path(freeze["technical_acceptance"])
        technical = technical if technical.is_absolute() else PROJECT / technical
        require(str(technical.resolve()) in self.frozen, "Technical acceptance receipt is not frozen")
        report = read(technical)
        require(report.get("status") == "COMPLETED" and report.get("gpu_training_verified") is True
                and report.get("checks") and all(v is True for v in report["checks"].values()),
                "Actual GPU technical loop and formal rollout concurrency have not passed")
        require(freeze.get("protocol", {}).get("merge_numeric_acceptance", {}).get("acceptance_basis")
                == "verified_bf16_tensor_merge_common_s0", "Frozen common S0 must be the explicitly verified actual BF16 merged endpoint")
        for method, prompts, group in (("S0", 16, 1), ("S+", 4, 1), ("G-CAL", 4, 4), ("G-RANK", 4, 4)):
            spec = freeze.get("matrix", {}).get(method, {})
            require((spec.get("updates"), spec.get("prompt_exposures_per_update"), spec.get("group_size"), spec.get("seed"))
                    == (256, prompts, group, 42), f"Frozen budget differs from original formal matrix: {method}")
        audit = read(self.args.input_audit)
        require(audit.get("status") == "COMPLETED" and audit.get("all_rows_encoded") is True and not audit.get("errors"),
                "Real full-input audit has not passed")
        require(Path(audit["model_path"]).resolve() == self.args.model.resolve(), "Original policy differs from the audited Qwen")
        require(str(Path(audit["rows_path"]).resolve()) in self.frozen, "Full-input audit row ledger was not frozen")
        nominated = report.get("evidence", {}).get("full_input_and_framework_audit", {})
        require(nominated.get("rows_sha256") == audit.get("rows_sha256"),
                "Selected input audit is not the full formal audit accepted in the technical gate")
        prepared = [json.loads(line) for line in self.args.manifest.read_text().splitlines() if line.strip()]
        expected_pool = sum(row["split"] == "train" for row in prepared) * 32
        for name in ("train_sft.jsonl", "train_grpo.jsonl"):
            path = str((self.args.data / name).resolve())
            require(audit.get("input_sha256", {}).get(path) == self.frozen[path]
                    and audit.get("per_file_rows", {}).get(path) == expected_pool,
                    "Actual formal pool must be the audited 32 instances per training video, not a smoke subset")
        self.audit = audit
        return freeze

    def environment(self, method=None, run=None):
        env = os.environ.copy()
        env.update({"PATH": str(Path(self.python).parent) + os.pathsep + env.get("PATH", ""),
                    "PYTHONPATH": str(PROJECT / "src") + os.pathsep + str(PROJECT / "scripts"),
                    "HL_DATA": str(self.args.data.resolve()), "HL_OUTPUT": str(self.output_root),
                    "HL_INPUT_AUDIT": str(self.args.input_audit.resolve()), "HL_FREEZE": str(self.args.freeze.resolve()),
                    "HL_INPUT_AUDIT_ROWS": str(Path(self.audit["rows_path"]).resolve()),
                    "HL_SEED": "42", "HL_STEPS": "256", "HL_PROMPTS": "4", "HL_SFT_BATCH": "16",
                    "HL_GROUP_SIZE": "4", "HL_RANK_WEIGHT": "0" if method == "G-CAL" else "0.8",
                    "NPROC_PER_NODE": "1", "CUDA_VISIBLE_DEVICES": "0",
                    "HF_HOME": str(WORK / "cache/huggingface"), "HF_HUB_CACHE": str(STORE / "models/hub"),
                    "HF_DATASETS_CACHE": str(WORK / "cache/datasets"), "TORCH_HOME": str(WORK / "cache/torch"),
                    "TMPDIR": str(WORK / "tmp"), "OMP_NUM_THREADS": "4"})
        env.update(self.audit["visual_environment"])
        require(all(isinstance(v, str) for v in env.values()), "Environment contains non-string values")
        if run is not None:
            env["HL_RUN_ID"] = Path(run).name
        return env

    def save(self):
        self.state["updated"] = now()
        atomic(self.state_path, self.state)

    def journal(self, event, **payload):
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps({"event": event, **now(), **payload}, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())

    def command(self, argv, name, *, env=None, allowed_diagnostic_failure=None):
        stage = self.state["stages"][name]
        command = list(map(str, argv))
        stage.setdefault("commands", []).append(command)
        self.save()
        self.journal("command_started", stage=name, command=command)
        log_path = self.root / "logs" / f"{name}.{len(stage['commands']):02d}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(command, env=env or self.environment(), cwd=PROJECT,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.active_process = process
            stage["active_child_pid"] = process.pid
            self.save()
            try:
                code = process.wait()
            except BaseException:
                self.stop_child()
                raise
            finally:
                self.active_process = None
            stage.pop("active_child_pid", None)
        stage.setdefault("logs", []).append(str(log_path))
        self.journal("command_exited", stage=name, exit_code=code, log=str(log_path))
        if code != 0 and allowed_diagnostic_failure is not None:
            require(allowed_diagnostic_failure(code) is True, f"Stage {name} had an unrecognized diagnostic failure")
            stage.setdefault("registered_diagnostic_failures", []).append({
                "command": command, "exit_code": code, "log": str(log_path),
                "reason": "Original fixed logp tolerance exceeded; preserve FAILED report. Independent bitwise/processor/reference checks remain mandatory."})
            self.journal("registered_diagnostic_failure", stage=name, exit_code=code, log=str(log_path))
            self.save()
        else:
            require(code == 0, f"Stage {name} command exited {code}; no automatic retry: {log_path}")
        return log_path

    def stop_child(self):
        process = self.active_process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def checkpoint(self, method):
        receipt = read(self.runs[method] / "receipt.json")
        require(receipt.get("status") == "COMPLETED" and receipt.get("method_id") == method,
                f"Formal training not accepted: {method}")
        path = final_checkpoint(receipt["output_dir"])
        require(path.resolve() == Path(receipt["endpoint"]["path"]).resolve(), "Accepted endpoint differs from unique final checkpoint")
        return path

    def preflight(self, stage, peak):
        path = self.root / "storage" / f"{stage}.json"
        path.parent.mkdir(exist_ok=True)
        self.command([self.python, PROJECT / "scripts/storage_audit.py", "--output", path,
                      "--reserve-path", self.output_root, "--peak-extra-bytes", peak], stage)
        require(read(path).get("ok") is True, "Registered storage budget/peak check did not pass")
        return path

    def train(self, method, stage):
        require(not self.runs[method].exists() and not (self.output_root / self.runs[method].name).exists(),
                "Training run/output already exists; refuse a hidden version increment or restart")
        storage = self.preflight(stage, 2 * 1024**3)
        action = "sft0" if method == "S0" else "sft-continue" if method == "S+" else "grpo"
        model = self.args.model if method == "S0" else self.merged
        self.command(["bash", PROJECT / "scripts/run.sh", action, model], stage,
                     env=self.environment(method, self.runs[method]))
        receipt_path = self.runs[method] / "receipt.json"
        receipt = read(receipt_path)
        require(receipt.get("status") == "TRAINER_EXITED_SUCCESS_REQUIRES_AUDIT" and receipt.get("mode") == "formal",
                "Successful launcher exit did not produce an unaudited formal trainer receipt")
        checkpoint = final_checkpoint(receipt["output_dir"])
        immutable = self.root / "trainer_exit_receipts" / f"{SLUG[method]}.json"
        immutable.parent.mkdir(exist_ok=True)
        require(not immutable.exists(), "Immutable trainer-exit copy already exists")
        immutable.write_bytes(receipt_path.read_bytes())
        return [storage, immutable, self.runs[method] / "training_summary.json",
                self.runs[method] / "training_ledger.jsonl", checkpoint / "adapter_model.safetensors",
                checkpoint / "adapter_config.json", checkpoint / "trainer_state.json"]

    def accept(self, method, stage):
        out = self.runs[method] / "acceptance.json"
        command = [self.python, PROJECT / "scripts/accept_training_run.py", "--run-dir", self.runs[method],
                   "--method", method, "--freeze", self.args.freeze, "--out", out]
        if method != "S0": command += ["--common-s0", self.common]
        if method.startswith("G-"):
            command += ["--allow-formal-zero-signal"]
        self.command(command, stage)
        require(read(out).get("status") == "COMPLETED", "Formal endpoint acceptance did not pass")
        self.checkpoint(method)
        paths = [out, self.runs[method] / "receipt.json", self.runs[method] / "trainer_exit_receipt.json"]
        root = self.checkpoint(method)
        paths += [root / name for name in read(out)["endpoint"]["files"]]
        if method.startswith("G-"):
            ledgers = list(self.runs[method].glob("rollouts.rank*.jsonl"))
            require(len(ledgers) == 1, "Expected one single-GPU reward ledger")
            paths += ledgers
        return paths

    def snapshot(self, stage):
        path = self.checks / "S0_snapshot.json"
        self.command([self.python, PROJECT / "scripts/verify_checkpoint.py", "snapshot", "--model", self.args.model,
                      "--adapter", self.checkpoint("S0"), "--inputs", self.args.data / "train_sft.jsonl", "--out", path], stage)
        obj = read(path)
        require(obj.get("status") == "COMPLETED" and obj.get("gpu_checkpoint_verification") is True, "Formal S0 GPU snapshot failed")
        return [path]

    def merge(self, stage):
        require(not self.merge_run.exists() and not self.merged.exists(), "Merge run/output already exists")
        storage = self.preflight(stage, 5 * 1024**3)
        self.command(["bash", PROJECT / "scripts/run.sh", "merge", self.args.model, self.checkpoint("S0")], stage,
                     env=self.environment("S0", self.merge_run))
        receipt = read(self.merge_run / "receipt.json")
        require(receipt.get("exit_code") == 0 and receipt.get("processor_restore_exit_code") == 0,
                "Merge/export or processor restoration failed")
        require(Path(receipt["output_dir"]).resolve() == self.merged, "Unexpected merged-model output path")
        restore = read(self.merge_run / "processor_restore.json")
        require(restore.get("status") == "COMPLETED" and restore.get("fixed_train_row_encode_verified") is True
                and restore.get("restored_entire_processor_package_byte_identical") is True
                and restore.get("model_weights_modified") is False, "Processor restoration did not preserve the pinned visual interface")
        return [storage, self.merge_run / "receipt.json", self.merge_run / "processor_restore.json",
                *[p for p in self.merged.rglob("*") if p.is_file()]]

    def compare(self, stage):
        path = self.checks / "S0_merge_compare.json"
        def only_initial_tolerance_failure(code):
            require(code == 1, "Only the measured compare RuntimeError may continue to independent verification")
            gap = self.measured_gap(path)
            require(gap["within_initial_tolerance"] is False, "Nonzero comparison exit did not correspond to the original threshold failure")
            return True
        self.command([self.python, PROJECT / "scripts/verify_checkpoint.py", "compare-merged", "--model", self.merged,
                      "--inputs", self.args.data / "train_sft.jsonl", "--snapshot", self.checks / "S0_snapshot.json", "--out", path],
                     stage, allowed_diagnostic_failure=only_initial_tolerance_failure)
        self.measured_gap(path)  # Finite vectors, original thresholds and real GPU stages must reproduce.
        return [path]

    def measured_gap(self, path):
        sys.path.insert(0, str(PROJECT / "scripts"))
        from verify_bf16_merge import logp_gap
        return logp_gap(read(self.checks / "S0_snapshot.json"), read(path))

    def verify_bitwise_merge(self, stage):
        path = self.checks / "S0_bf16_merge_verification.json"
        self.command([self.python, PROJECT / "scripts/verify_bf16_merge.py", "--base", self.args.model,
                      "--adapter", self.checkpoint("S0"), "--merged", self.merged,
                      "--snapshot", self.checks / "S0_snapshot.json", "--compare", self.checks / "S0_merge_compare.json",
                      "--threads", "8", "--out", path], stage)
        obj = read(path)
        require(obj.get("status") == "COMPLETED" and obj.get("merge_validation_passed") is True
                and obj.get("acceptance_basis") == "verified_bf16_tensor_merge_common_s0"
                and obj.get("merge_formula_verified_all_tensors") is True
                and obj.get("aggregate", {}).get("different_from_exact_bf16_elements") == 0
                and obj.get("premerge_functional_equivalence_claimed") is False,
                "Full bitwise BF16 merge validation failed; logp gap alone cannot authorize a common endpoint")
        return [path]

    def verify_reference(self, stage):
        storage = self.preflight(stage, 1024**3)
        path = self.checks / "S0_reference_integrity.json"
        probe = self.checks / "reference_integrity_probe_adapter"
        self.command([self.python, PROJECT / "scripts/verify_checkpoint.py", "reference-update", "--model", self.merged,
                      "--inputs", self.args.data / "train_sft.jsonl", "--adapter-out", probe, "--out", path], stage)
        obj = read(path)
        require(obj.get("status") == "COMPLETED" and obj.get("gpu_checkpoint_verification") is True
                and obj.get("gpu_reference_update_verified") is True
                and obj.get("reference_invariance", {}).get("exact_equal") is True,
                "Actual merged S0 reference drifted or could not be independently verified")
        return [storage, path, *[p for p in probe.rglob("*") if p.is_file()]]

    def register_common(self, stage):
        require(not self.common.exists(), "Existing common_s0.json cannot be silently replaced")
        parent = read(self.runs["S0"] / "receipt.json")
        verification = read(self.checks / "S0_bf16_merge_verification.json")
        reference_check = read(self.checks / "S0_reference_integrity.json")
        processor = read(self.merge_run / "processor_restore.json")
        gap = self.measured_gap(self.checks / "S0_merge_compare.json")
        require(parent.get("status") == "COMPLETED" and parent.get("mode") == "formal"
                and parent.get("gpu_training_verified") is True and parent.get("method_id") == "S0"
                and verification.get("status") == "COMPLETED" and verification.get("merge_validation_passed") is True
                and verification.get("acceptance_basis") == "verified_bf16_tensor_merge_common_s0"
                and verification.get("merge_formula_verified_all_tensors") is True
                and verification.get("aggregate", {}).get("different_from_exact_bf16_elements") == 0
                and reference_check.get("status") == "COMPLETED"
                and reference_check.get("gpu_reference_update_verified") is True
                and reference_check.get("reference_invariance", {}).get("exact_equal") is True
                and processor.get("status") == "COMPLETED" and processor.get("model_weights_modified") is False
                and processor.get("restored_entire_processor_package_byte_identical") is True
                and processor.get("fixed_train_row_encode_verified") is True,
                "Cannot register common S0 without actual tensor, processor and fixed-reference acceptance")
        common = {"status": "COMPLETED", "created": now(), "seed": 42,
                  "merge_validation_passed": True, "acceptance_basis": "verified_bf16_tensor_merge_common_s0",
                  "merge_logp_within_initial_tolerance": gap["within_initial_tolerance"],
                  "merge_logp_gap": gap, "premerge_functional_equivalence_claimed": False,
                  "definition": "Common S0 is the actual verified BF16 quantized merged endpoint; all S0 inference and continuations use this identical model.",
                  "merged_model": str(self.merged),
                  "merged_model_files": inventory(self.merged), "sft_endpoint": str(self.checkpoint("S0")),
                  "sft_endpoint_files": parent["endpoint"]["files"],
                  "parent_sft_receipt": reference(self.runs["S0"] / "receipt.json"),
                  "parent_sft_acceptance": reference(self.runs["S0"] / "acceptance.json"),
                  "snapshot": reference(self.checks / "S0_snapshot.json"),
                  "merge_comparison": reference(self.checks / "S0_merge_compare.json"),
                  "merge_bitwise_verification": reference(self.checks / "S0_bf16_merge_verification.json"),
                  "reference_verification": reference(self.checks / "S0_reference_integrity.json"),
                  "processor_restore": reference(self.merge_run / "processor_restore.json"),
                  "protocol_freeze": reference(self.args.freeze), "exact_resume_verified": False,
                  "new_identical_lora_spec_for_each_continuation": {"rank": 16, "alpha": 32, "dropout": 0}}
        atomic(self.common, common, fresh=True)
        immutable = self.root / "common_s0.json"
        immutable.write_bytes(self.common.read_bytes())
        return [self.common, immutable]

    def infer(self, method, stage):
        storage = self.preflight(stage, 512 * 1024**2)
        command = [self.python, PROJECT / "scripts/inference.py", "--run-id", method,
                   "--inputs", self.args.data / "eval.jsonl", "--manifest", self.args.manifest,
                   "--pseudo", self.args.pseudo, "--input-audit", self.args.input_audit, "--out", self.inference[method]]
        if method == "B1": command += ["--model", self.args.model]
        elif method in TRAINED:
            command += ["--model", self.merged, "--common-s0", self.common,
                        "--training-receipt", self.runs[method] / "receipt.json"]
            if method != "S0": command += ["--adapter", self.checkpoint(method)]
        self.command(command, stage)
        receipt = read(self.inference[method] / "receipt.json")
        require(receipt.get("status") == "COMPLETED" and receipt.get("run_id") == method, "Inference did not actually complete")
        paths = [storage, self.inference[method] / "receipt.json", self.inference[method] / "videos.jsonl"]
        if method != "B0": paths.append(self.inference[method] / "windows.jsonl")
        return paths

    def freeze_predictions(self, stage):
        path = self.root / "inference_receipts.json"
        atomic(path, {method: str(self.inference[method] / "receipt.json") for method in METHODS}, fresh=True)
        self.command([self.python, PROJECT / "scripts/freeze_predictions.py", "--manifest", self.args.manifest,
                      "--protocol-freeze", self.args.freeze, "--receipts-index", path, "--out", self.prediction_freeze], stage)
        require(read(self.prediction_freeze).get("require_run_provenance") is True, "Predictions lack real six-run provenance")
        return [path, self.prediction_freeze]

    def extract_mat(self, stage):
        # This is the first annotation extraction. It occurs only after six real
        # predictions and all formal endpoint identities have been frozen.
        sys.path.insert(0, str(PROJECT / "src"))
        from unsgrpo.evaluation import validate_six_run_freeze
        require(validate_six_run_freeze(self.prediction_freeze)["require_run_provenance"] is True,
                "Annotation access requires the verified six-run prediction freeze")
        require(not self.annotations.exists(), "Use a new annotation-conversion directory")
        with zipfile.ZipFile(self.args.annotations_zip) as archive:
            matches = [i for i in archive.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() == ".mat"]
            require(len(matches) == 1, "Official annotation archive must contain exactly one MAT; do not guess")
            info = matches[0]
            require(0 < info.file_size < 100 * 1024**2, "Unexpected annotation MAT extraction size")
            self.annotations.mkdir(parents=True)
            path = self.annotations / "official_tvsum.mat"
            with archive.open(info) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
        receipt = self.annotations / "extraction_receipt.json"
        atomic(receipt, {"status": "COMPLETED", "created": now(), "archive": reference(self.args.annotations_zip),
                         "member": info.filename, "mat": reference(path),
                         "prediction_freeze": reference(self.prediction_freeze), "values_parsed": False}, fresh=True)
        return [path, receipt]

    def convert(self, stage):
        self.command([self.python, PROJECT / "scripts/convert_tvsum_labels.py", "--mat", self.annotations / "official_tvsum.mat",
                      "--freeze", self.prediction_freeze, "--out", self.annotations / "converted"], stage)
        return [p for p in (self.annotations / "converted").iterdir() if p.is_file()]

    def evaluate(self, stage):
        self.command([self.python, PROJECT / "scripts/evaluate.py", "--freeze", self.prediction_freeze,
                      "--annotations", self.annotations / "converted", "--out", self.evaluation], stage)
        evaluation = read(self.evaluation)
        require(evaluation.get("status") == "COMPLETED" and evaluation.get("require_run_provenance") is True
                and evaluation.get("synthetic_fixture_only") is False, "Only real six-arm V0 evaluation is acceptable")
        return [self.evaluation]

    def archive(self, stage):
        require(not self.archive_root.exists(), "Archive already exists; no silent recopy/retry")
        metadata = []
        destinations = set()
        def add(path, destination):
            path = Path(path).resolve()
            require(path.is_file(), f"Missing archive metadata source: {path}")
            require(destination not in destinations, f"Duplicate archive metadata destination: {destination}")
            destinations.add(destination)
            metadata.append({**reference(path), "destination": destination})
        for path in self.frozen:
            source = Path(path)
            if source.is_relative_to(PROJECT):
                add(source, "source/" + str(source.relative_to(PROJECT)))
            elif source.name in {"_asset_receipt.json", "asset_receipt.json"}:
                add(source, f"provenance/assets/{source.parent.name}__{source.name}")
        for path, destination in (
            (self.args.freeze, "provenance/protocol.freeze.json"),
            (self.args.manifest, "manifests/prepared.jsonl"),
            (self.args.pseudo, "proxy/pseudo.json"),
            (self.args.input_audit, "provenance/input_audit.json"),
            (self.audit["rows_path"], "provenance/input_audit_rows.jsonl"),
            (self.prediction_freeze, "predictions/freeze.json"),
            (self.evaluation, "evaluation/V0.json"),
            (self.annotations / "converted/conversion_receipt.json", "evaluation/conversion_receipt.json"),
        ):
            add(path, destination)
        for name in ("train_sft.jsonl", "train_grpo.jsonl", "eval.jsonl"):
            add(self.args.data / name, "trainer_inputs/" + name)
        for method in METHODS:
            for path in sorted(self.inference[method].iterdir()):
                if path.is_file() and path.suffix in {".json", ".jsonl"}:
                    add(path, "inference/" + method.replace("+", "PLUS") + "/" + path.name)
        for name, previous in self.state["stages"].items():
            if previous["status"] == "COMPLETED":
                for number, path in enumerate(previous.get("logs", []), 1):
                    add(path, f"controller_logs/{name}.{number:02d}.log")
        for path in sorted(self.checks.glob("*.json")):
            add(path, "checks/" + path.name)
        # The live controller state/journal will change when the copy command
        # starts. Archive immutable snapshots, not files that are being appended.
        snapshot_dir = self.root / "prearchive_snapshot"
        snapshot_dir.mkdir()
        for path in (self.state_path, self.root / "events.jsonl"):
            destination = snapshot_dir / path.name
            destination.write_bytes(path.read_bytes())
            add(destination, "controller_snapshot/" + path.name)
        source_index = self.root / "archive_sources.json"
        sources = {"schema_version": 1, "prefix": self.prefix, "common_s0": reference(self.common),
                   "runs": {method: reference(self.runs[method] / "receipt.json") for method in TRAINED},
                   "metadata": metadata, "excluded": "original videos, decoded frames, human annotation contents, environments, original base-model copies"}
        atomic(source_index, sources, fresh=True)
        self.command([self.python, PROJECT / "scripts/archive_round1.py", "--sources-index", source_index,
                      "--out", self.archive_root, "--receipt", self.archive_receipt], stage)
        receipt = read(self.archive_receipt)
        require(receipt.get("status") == "COMPLETED" and receipt.get("files")
                and receipt.get("local_sources_deleted") is False
                and Path(receipt["archive_root"]).resolve() == self.archive_root,
                "Archive must be copied, SHA-verified and local sources retained before proceeding")
        targets = []
        for item in receipt["files"]:
            destination = (self.archive_root / item["destination"]).resolve()
            require(destination.is_relative_to(self.archive_root) and destination.is_file()
                    and self.digest(destination) == item["sha256"], "Archived file missing/changed after copy")
            targets.append(destination)
        manifest = receipt["archive_manifest"]
        require(Path(manifest["path"]).resolve().is_relative_to(self.archive_root)
                and self.digest(manifest["path"]) == manifest["sha256"], "Archive manifest changed")
        # Include generated archive receipts/manifests as well as every payload
        # file in future completed-stage revalidation.
        all_archive_files = [p for p in self.archive_root.rglob("*") if p.is_file()]
        require(all(not p.is_symlink() and p.resolve().is_relative_to(self.archive_root) for p in all_archive_files),
                "Archive contains an unexpected symlink/path escape")
        return [source_index, self.archive_receipt, *all_archive_files]

    def artifact_index(self, stage):
        self.command([self.python, PROJECT / "scripts/storage_audit.py", "--output", self.storage_final], stage)
        result = {"schema_version": 1, "protocol_freeze": reference(self.args.freeze),
                  "prediction_freeze": reference(self.prediction_freeze), "evaluation": reference(self.evaluation),
                  "common_s0": reference(self.common), "storage_audit": reference(self.storage_final), "runs": {},
                  "sync_policy": "ON_DEMAND_USER_REQUEST_ONLY_NO_AUTOMATION"}
        result["archive_receipt"] = reference(self.archive_receipt)
        result["archive_controller_wall_seconds"] = self.state["stages"]["archive_endpoints"]["wall_seconds"]
        for method in METHODS:
            entry = {"inference_receipt": reference(self.inference[method] / "receipt.json")}
            if method in TRAINED:
                receipt = read(self.runs[method] / "receipt.json")
                entry.update(training_receipt=reference(self.runs[method] / "receipt.json"),
                             telemetry_summary=reference(self.runs[method] / "training_summary.json"),
                             training_ledger=reference(self.runs[method] / "training_ledger.jsonl"), endpoint=receipt["endpoint"])
                if method.startswith("G-"):
                    paths = list(self.runs[method].glob("rollouts.rank*.jsonl"))
                    require(len(paths) == 1, "Expected one reward ledger")
                    entry["rollout_ledger"] = reference(paths[0])
            result["runs"][method] = entry
        proxy = self.args.pseudo.parent / "diagnostics.json"
        if proxy.is_file(): result["proxy_diagnostics"] = reference(proxy)
        sync = WORK / "runs/sync/last_sync.json"
        if sync.is_file():
            immutable_sync = self.root / "sync_status_at_reporting.json"
            require(not immutable_sync.exists(), "Sync status snapshot already exists")
            immutable_sync.write_bytes(sync.read_bytes())
            result["sync_receipt"] = reference(immutable_sync)
        atomic(self.index, result, fresh=True)
        duplicate = self.root / "round1_artifacts.json"
        duplicate.write_bytes(self.index.read_bytes())
        return [self.storage_final, self.index, duplicate]

    def report(self, stage):
        self.command([self.python, PROJECT / "scripts/report_round1.py", "--index", self.index,
                      "--out-md", self.markdown, "--out-json", self.result_json], stage)
        require(read(self.result_json).get("status") == "COMPLETED", "Scientific comparison report is INCOMPLETE")
        return [self.markdown, self.result_json]

    def stages(self):
        stages = OrderedDict()
        stages["S0_train"] = lambda s: self.train("S0", s)
        stages["S0_accept"] = lambda s: self.accept("S0", s)
        stages["S0_snapshot"] = self.snapshot
        stages["S0_merge_restore"] = self.merge
        stages["S0_compare"] = self.compare
        stages["S0_bitwise_verify"] = self.verify_bitwise_merge
        stages["S0_reference_verify"] = self.verify_reference
        stages["common_S0_register"] = self.register_common
        for method in TRAINED[1:]:
            stages[f"{SLUG[method]}_train"] = lambda s, m=method: self.train(m, s)
            stages[f"{SLUG[method]}_accept"] = lambda s, m=method: self.accept(m, s)
        for method in METHODS:
            stages["infer_" + method.replace("+", "PLUS")] = lambda s, m=method: self.infer(m, s)
        stages.update(predictions_freeze=self.freeze_predictions, annotation_extract=self.extract_mat,
                      annotation_convert=self.convert, evaluation_V0=self.evaluate,
                      archive_endpoints=self.archive,
                      artifact_index=self.artifact_index, scientific_report=self.report)
        return stages

    def plan(self):
        return {"status": "PLANNED", "executes_commands": False, "label_values_read": False,
                "freeze_required": str(self.args.freeze), "state_dir": str(self.root),
                "protocol_budget": {"S0": "256 successful updates, 4096 input exposures",
                                    "S+/G-CAL/G-RANK": "each 256 updates, 1024 exposures; each GRPO 4096 completions"},
                "gpu_scheduling": "strictly sequential subprocesses; no background GPU tasks",
                "resume_scope": "verified COMPLETED controller stages only; no trainer restart/resume",
                "sync_policy": "manual explicit user request only; this driver never invokes Git",
                "stages": list(self.stages()), "runs": {k: str(v) for k, v in self.runs.items()},
                "common_s0": str(self.common), "artifact_index": str(self.index),
                "archive": str(self.archive_root), "local_files_deleted_after_archive": False,
                "final_report": str(self.markdown), "evaluation_protocol": "registered V0; no label-based selection"}

    def verify_completed(self, stage):
        require(stage.get("status") == "COMPLETED" and stage.get("artifacts"), "Only completed stages with output hashes can be skipped")
        for path, expected in stage["artifacts"].items():
            require(Path(path).is_file() and self.digest(path) == expected, f"Completed-stage artifact changed/missing: {path}")

    def execution_config(self):
        result = {name: str(getattr(self.args, name).resolve()) for name in
                ("state_dir", "freeze", "python", "model", "data", "manifest", "pseudo",
                 "input_audit", "checkpoints", "annotations_zip")}
        result["python"] = self.python
        return result

    def execute(self):
        self.validate_freeze()
        require(self.args.python.is_file(), "Pinned environment interpreter is missing")
        require(self.output_root.is_relative_to((WORK / "checkpoints").resolve()), "Checkpoints must use the registered external local directory")
        require(os.environ.get("NPROC_PER_NODE", "1") == "1", "Driver supports exactly one GPU")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0", "Unexpected external GPU selection; do not silently change it")
        self.root.mkdir(parents=True, exist_ok=True)
        # A shared lock also prevents two labels/controllers from scheduling this
        # project's one GPU concurrently. It does not alter system GPU settings.
        with (WORK / "runs/.round1_gpu_driver.lock").open("a") as gpu_lock, (self.root / ".driver.lock").open("a") as lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stages = self.stages()
            if self.state_path.exists():
                require(self.args.resume, "State exists; use --resume for verified completed stages only")
                self.state = read(self.state_path)
                require(self.state.get("freeze_sha256") == self.digest(self.args.freeze), "Controller protocol freeze identity changed")
                require(self.state.get("execution_config") == self.execution_config(), "Controller paths/model/data changed during state resume")
                require(list(self.state.get("stages", {})) == list(stages), "Controller stage plan changed")
                require(self.state.get("status") not in {"FAILED", "INTERRUPTED"}, "Failed controller requires explicit investigation; no automatic trainer restart")
                for stage in self.state["stages"].values():
                    require(stage.get("status") in {"PLANNED", "COMPLETED"}, "Interrupted/running/failed stage cannot be automatically resumed")
                    if stage["status"] == "COMPLETED": self.verify_completed(stage)
                if self.state.get("status") == "COMPLETED":
                    print(json.dumps({"status": "COMPLETED", "already_verified": True, "report": str(self.markdown)})); return
            else:
                require(not self.args.resume, "No controller state exists to resume")
                require(not [p for p in self.root.iterdir() if p.name != ".driver.lock"],
                        "New controller state directory must be empty; choose a new comparison label")
                require(not self.common.exists() and not self.index.exists(), "A previous common S0/index exists; do not overwrite another comparison")
                training_outputs = [self.output_root / p.name for p in self.runs.values()]
                require(all(not p.exists() for p in [*self.runs.values(), *training_outputs, self.merge_run, self.merged, self.markdown,
                                                     self.result_json, self.annotations]), "Existing run/endpoint/output conflicts with this new comparison")
                require(not self.archive_root.exists(), "Archive label already exists; choose a new registered comparison")
                self.state = {"status": "RUNNING", "created": now(), "driver_pid": os.getpid(),
                              "freeze_path": str(self.args.freeze.resolve()), "freeze_sha256": self.digest(self.args.freeze),
                              "driver_source_sha256": self.digest(Path(__file__)), "exact_training_resume_verified": False,
                              "execution_config": self.execution_config(),
                              "sync_policy": "ON_DEMAND_USER_REQUEST_ONLY_NO_AUTOMATION",
                              "stages": {s: {"status": "PLANNED", "commands": []} for s in stages}}
                self.save()
            self.checks.mkdir(exist_ok=True)
            for name, action in stages.items():
                stage = self.state["stages"][name]
                if stage["status"] == "COMPLETED":
                    self.verify_completed(stage); continue
                started = time.monotonic()
                stage.update(status="RUNNING", started=now())
                stage["controller_action"] = {"script": str(Path(__file__).resolve()),
                                              "source_sha256": self.digest(Path(__file__)), "stage": name}
                self.state.update(status="RUNNING", active_stage=name, driver_pid=os.getpid()); self.save()
                self.journal("stage_started", stage=name)
                try:
                    self.validate_freeze()  # Frozen sources/assets rechecked before every stage.
                    for previous in self.state["stages"].values():
                        if previous["status"] == "COMPLETED": self.verify_completed(previous)
                    paths = action(name)
                    artifacts = {str(Path(p).resolve()): self.digest(p) for p in paths}
                    require(artifacts, f"Stage {name} produced no hash-bound artifacts")
                    stage.update(status="COMPLETED", ended=now(), wall_seconds=time.monotonic()-started, artifacts=artifacts)
                    self.journal("stage_completed", stage=name, wall_seconds=stage["wall_seconds"], artifacts=artifacts)
                    self.save()
                    print(json.dumps({"stage": name, "status": "COMPLETED", "seconds": stage["wall_seconds"]}), flush=True)
                except BaseException as exc:
                    self.stop_child()
                    stage.update(status="FAILED", ended=now(), wall_seconds=time.monotonic()-started,
                                 error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                    self.state.update(status="FAILED", failed_stage=name); self.save()
                    self.journal("stage_failed", stage=name, error=stage["error"])
                    raise
            self.state.update(status="COMPLETED", active_stage=None, completed=now(), report=reference(self.markdown),
                              report_json=reference(self.result_json), common_s0=reference(self.common))
            self.save()
            print(json.dumps({"status": "COMPLETED", "report": str(self.markdown), "state": str(self.state_path)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--state-dir", type=Path, default=WORK / "runs/round1_seed42_v1")
    p.add_argument("--freeze", type=Path, default=PROJECT / "reports/freeze.json")
    p.add_argument("--python", type=Path, default=WORK / "envs/highlights/bin/python")
    p.add_argument("--model", type=Path, default=STORE / "models/Qwen3-VL-2B-Instruct")
    p.add_argument("--data", type=Path, default=STORE / "datasets/trainer_inputs")
    p.add_argument("--manifest", type=Path, default=WORK / "prepared/prepared.jsonl")
    p.add_argument("--pseudo", type=Path, default=STORE / "datasets/pseudo/r0_visual_v1/pseudo.json")
    p.add_argument("--input-audit", type=Path, default=PROJECT / "reports/input_audit.json")
    p.add_argument("--checkpoints", type=Path, default=WORK / "checkpoints")
    p.add_argument("--annotations-zip", type=Path, default=STORE / "datasets/tvsum/labels_eval_only/source/ydata-tvsum50-matlab.zip")
    args = p.parse_args()
    require(not args.resume or args.execute, "--resume requires --execute")
    driver = Driver(args)
    if not args.execute:
        print(json.dumps(driver.plan(), ensure_ascii=False, indent=2)); return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Driver interrupted by signal {signum}; no trainer restart will be attempted")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(signum, interrupted)
    driver.execute()


if __name__ == "__main__":
    main()
