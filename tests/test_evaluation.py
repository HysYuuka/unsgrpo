import numpy as np
import pytest
import json
import runpy
from pathlib import Path

from unsgrpo.evaluation import (
    average_precision_noninterpolated,
    evaluate_top5_umt,
    evaluate_v0,
    file_sha256,
    interval_average,
    midranks,
    paired_bootstrap,
    spearman_with_status,
    validate_prediction_records,
    validate_six_run_freeze,
)


def test_interval_overlap_hand_calculation_and_last_short_clip():
    got = interval_average([[0, 2], [2, 4], [4, 4.5]], [10, 40, 100], [[0, 3], [3, 4.5]])
    assert got == pytest.approx([20, 60])


@pytest.mark.parametrize("bounds", [
    [[0, 1], [2, 3]],  # uncovered gap
    [[0, 2], [1, 3]],  # overlap
    [[2, 3], [0, 2]],  # nonchronological
    [[0, 1], [1, 2.5]],  # incomplete final interval
])
def test_interval_rejects_partial_or_duplicate_coverage(bounds):
    with pytest.raises(ValueError):
        interval_average(bounds, [1, 2], [[0, 3]])


def test_ap_hand_calculation():
    # Positive ranks 1 and 3: AP = (1 + 2/3) / 2.
    assert average_precision_noninterpolated([1, 0, 1], [3, 2, 1]) == pytest.approx(5 / 6)


def test_ap_tied_thresholds_do_not_choose_favorable_order():
    assert average_precision_noninterpolated([1, 0], [1, 1]) == 0.5
    assert average_precision_noninterpolated([0, 1], [1, 1]) == 0.5
    assert average_precision_noninterpolated([0, 0], [2, 1]) == 0.0


def test_v0_user_threshold_strict_and_average_order():
    bounds = [[0, 2], [2, 4], [4, 6]]
    result = evaluate_v0(bounds, [3, 2, 1], bounds, [[3, 1], [2, 1], [1, 1]])
    assert result["ap_by_annotator"] == [1.0, 0.0]
    assert result["mean_ap"] == 0.5
    assert result["n_no_positive_annotators"] == 1


def test_spearman_midranks_and_explicit_undefined():
    assert midranks([4, 2, 2]) == pytest.approx([3, 1.5, 1.5])
    assert spearman_with_status([1, 1, 2], [3, 3, 1])["value"] == pytest.approx(-1)
    assert spearman_with_status([1, 1], [2, 3]) == {
        "value": None, "defined": False, "reason": "constant_or_singleton"}


def test_prediction_record_reordering():
    got = validate_prediction_records(["a", "b"], [{"request_id": "b"}, {"request_id": "a"}])
    assert list(got) == ["a", "b"]


@pytest.mark.parametrize("records", [
    [{"request_id": "a"}],
    [{"request_id": "a"}, {"request_id": "a"}],
    [{"request_id": "a"}, {"request_id": "b"}, {"request_id": "c"}],
    [{"id": "a"}, {"request_id": "b"}],
])
def test_prediction_missing_duplicate_extra_rejected(records):
    with pytest.raises(ValueError):
        validate_prediction_records(["a", "b"], records)


def test_paired_bootstrap_source_group_not_window_unit():
    result = paired_bootstrap([1, 1, -1], n_boot=500, source_groups=["same", "same", "other"])
    assert result["mean"] == 0.0
    assert result["video_macro_mean"] == pytest.approx(1 / 3)
    assert result["n_source_groups"] == 2
    assert result["ci_low"] == -1
    assert result["ci_high"] == 1
    assert result["training_seed_uncertainty_included"] is False


def test_paired_bootstrap_deterministic():
    assert paired_bootstrap([0.2, 0.1, -0.1], n_boot=101) == paired_bootstrap([0.2, 0.1, -0.1], n_boot=101)


def test_umt_top5_hand_calculation_is_not_sklearn_ap():
    pytest.importorskip("torch")
    human = np.tile(np.array([5, 1, 5, 1, 1, 1])[:, None], (1, 20))
    result = evaluate_top5_umt([6, 5, 4, 3, 2, 1], human)
    # Top-5 positive ranks 1 and 3; trapezoidal AP = 1/2 + (1/2+2/3)/4.
    assert result["mean_ap"] == pytest.approx(19 / 24)
    assert result["benchmark_alignment_verified"] is False


def test_umt_zero_positive_top5_and_source_tie_order_recorded():
    pytest.importorskip("torch")
    human = np.tile(np.array([1, 1, 1, 1, 1, 5])[:, None], (1, 20))
    result = evaluate_top5_umt([6, 5, 4, 3, 2, 1], human)
    assert result["mean_ap"] == 0
    assert result["n_no_positive_topk_annotators"] == 20
    assert "stable-false" in result["tie_backend"]


@pytest.fixture
def synthetic_freeze(tmp_path):
    np.save(tmp_path / "frames.npy", np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=float))
    video = {"video_id": "synthetic", "split": "test", "source_group": "independent",
             "duration": 4, "n_frames": 4, "bounds": [[0, 2], [2, 4]],
             "frame_bounds_path": str(tmp_path / "frames.npy")}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(video) + "\n")
    freeze = {"manifest": {"path": str(manifest), "sha256": file_sha256(manifest)}, "runs": {}}
    for run in ["B0", "B1", "S0", "S+", "G-CAL", "G-RANK"]:
        path = tmp_path / f"{run}.jsonl"
        path.write_text(json.dumps({"video_id": "synthetic", "bounds": video["bounds"], "scores": [100, 0]}) + "\n")
        freeze["runs"][run] = {"path": str(path), "sha256": file_sha256(path)}
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    return path, freeze


def test_six_run_freeze_rejects_tampered_predictions_before_labels(synthetic_freeze, tmp_path):
    freeze_path, freeze = synthetic_freeze
    audited = validate_six_run_freeze(freeze_path)
    assert len(audited["runs"]) == 6
    Path(freeze["runs"]["B1"]["path"]).write_text("[]")
    converter = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "convert_tvsum_labels.py"))
    with pytest.raises(ValueError, match="hash mismatch"):
        converter["convert"](tmp_path / "deliberately_absent.mat", freeze_path, tmp_path / "labels")
    assert not (tmp_path / "labels").exists()


def test_synthetic_frozen_mat_conversion_and_evaluation(synthetic_freeze, tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    freeze_path, _ = synthetic_freeze
    mat_path = tmp_path / "labels.mat"
    scipy_io.savemat(mat_path, {"tvsum50": {"video": "synthetic", "nframes": 4,
                       "user_anno": np.tile(np.array([5, 1, 5, 1])[:, None], (1, 20))}})
    scripts = Path(__file__).parents[1] / "scripts"
    converter = runpy.run_path(str(scripts / "convert_tvsum_labels.py"))
    converted = converter["convert"](mat_path, freeze_path, tmp_path / "labels", allow_synthetic_fixture=True)
    assert converted["evaluation_unit"] == "native_frame"
    evaluator = runpy.run_path(str(scripts / "evaluate.py"))
    report = evaluator["evaluate"](freeze_path, tmp_path / "labels", tmp_path / "report.json", allow_synthetic_fixture=True)
    assert report["V1"]["status"] == "BLOCKED"
    assert report["runs"]["B0"]["video_macro_ap"] == 0.5
    assert report["primary_paired_difference"]["mean"] == 0
    assert report["primary_paired_difference"]["n_source_groups"] == 1
    assert report["synthetic_fixture_only"] is True


def test_freeze_missing_branch_fails(synthetic_freeze):
    path, freeze = synthetic_freeze
    del freeze["runs"]["S+"]
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="six runs"):
        validate_six_run_freeze(path)


def test_formal_provenance_flag_requires_real_inference_receipts(synthetic_freeze):
    path, freeze = synthetic_freeze
    freeze["require_run_provenance"] = True
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="path and sha256"):
        validate_six_run_freeze(path)


def test_real_label_conversion_rejects_hash_only_freeze_before_reading_mat(synthetic_freeze, tmp_path):
    path, _ = synthetic_freeze
    converter = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "convert_tvsum_labels.py"))
    with pytest.raises(ValueError, match="require_run_provenance"):
        converter["convert"](tmp_path / "deliberately_absent.mat", path, tmp_path / "labels")
    assert not (tmp_path / "labels").exists()
