from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sac_qutab.config import load_config
from sac_qutab.detector import evaluate_nested_detectors, fit_nested_detectors
from sac_qutab.metrics import (
    MetricUndefinedError,
    apply_control_ecdf,
    brier_score,
    detection_metrics,
    expected_calibration_error,
    fit_control_ecdf,
    js_divergence,
    linear_cka,
)
from sac_qutab.runtime import atomic_write_json, sha256_file, write_jsonl
from sac_qutab.statistics import benjamini_hochberg, cluster_bootstrap_difference, clustered_detector_metric, generate_primary_statistics, holm_bonferroni, summarize_seed_estimates, wilcoxon_sensitivity

CONFIG = Path("configs/core.yaml")


def test_probability_metrics_hand_fixtures() -> None:
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    js = js_divergence(p, q)
    assert js[0] == pytest.approx(0.0)
    assert 0 < js[1] <= np.log(2)
    assert brier_score(p, [0, 1]) == pytest.approx(0.25)
    ece = expected_calibration_error(np.array([[0.8, 0.2], [0.4, 0.6]]), [0, 0], 2)
    assert sum(row["count"] for row in ece["bins"]) == 2
    assert 0 <= ece["ece"] <= 1


def test_cka_and_ecdf_scale_contract() -> None:
    x = np.arange(24, dtype=float).reshape(6, 4)
    assert linear_cka(x, x * 10 + 3) == pytest.approx(1.0)
    ecdf_a = fit_control_ecdf([1, 2, 3, 4])
    ecdf_b = fit_control_ecdf([10, 20, 30, 40])
    assert np.array_equal(apply_control_ecdf([1.5, 3.5], ecdf_a), apply_control_ecdf([15, 35], ecdf_b))
    with pytest.raises(MetricUndefinedError):
        linear_cka(np.ones((3, 2)), np.ones((3, 2)))


def test_detection_undefined_and_underpowered() -> None:
    result = detection_metrics([0, 0], [0.1, 0.2], 2)
    assert not result["defined"] and result["reason"] == "single target class"
    result = detection_metrics([0, 1, 0], [0.1, 0.9, 0.2], 2)
    assert not result["defined"] and result["underpowered"]
    assert result["exploratory_estimates"]["average_precision"] == pytest.approx(1.0)


def _detector_rows(role: str, independent_signal: bool) -> list[dict]:
    rows = []
    for index in range(80):
        failure = int(index % 3 == 0)
        output = (index % 7) / 20
        representation = (0.9 if failure else 0.1) if independent_signal else 0.5
        severity = "mild" if index % 2 else "severe"
        rows.append({
            "pair_id": f"{role}-{index}", "source_id": f"source-{index}", "source_group_id": f"group-{index // 2}",
            "role": role, "family": "resolution", "severity": severity, "audit_condition": f"resolution:{severity}",
            "class_name": "Forest", "audit_qualified": True, "clean_correct": True, "counterfactual_failure": failure,
            "transformed_uncertainty": output, "transformed_entropy": output * 0.8 + 0.1,
            "normalized_representation_instability": representation,
        })
    rows.append({
        **rows[0], "pair_id": f"{role}-sham", "severity": "sham", "audit_condition": "resolution:sham",
        "counterfactual_failure": 1,
    })
    return rows


def _evaluate_detector(tmp_path: Path, independent_signal: bool) -> dict:
    cfg = load_config(CONFIG)
    detector_path = tmp_path / f"detector-{independent_signal}.json"
    calibration_provenance = {
        "split_role": "calibration", "model_name": "resnet50", "training_seed": 17,
        "config_sha256": "config", "condition_signature": "conditions", "pair_manifest_sha256": "calibration-pairs",
        "checkpoint_sha256": "checkpoint", "resolved_run_sha256": "run", "review_freeze_sha256": "review",
        "control_ecdf_sha256": "ecdf",
    }
    detector = fit_nested_detectors(
        _detector_rows("calibration", independent_signal), cfg.detector, calibration_provenance,
        detector_path, cfg.evaluation.rq3_min_positives_per_model_seed,
    )
    protocol_path = tmp_path / f"protocol-{independent_signal}.json"
    atomic_write_json(protocol_path, {
        "schema_version": "sac-experiment-protocol-v4",
        "detector_sha256": {"resnet50-seed17": sha256_file(detector_path)},
        "control_ecdf_sha256": {"resnet50-seed17": "ecdf"},
    })
    final_provenance = {**calibration_provenance, "split_role": "final_test", "pair_manifest_sha256": "final-pairs", "protocol_sha256": sha256_file(protocol_path)}
    return evaluate_nested_detectors(
        _detector_rows("final_test", independent_signal), detector, final_provenance,
        cfg.evaluation.rq3_min_positives_per_model_seed, detector_path=detector_path, protocol_path=protocol_path,
    )


def test_nested_detector_isolation_sham_exclusion_and_increment(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    result = _evaluate_detector(tmp_path, True)
    assert result["population"]["sham_count"] == 0
    assert result["population"]["count"] == 80
    assert result["models"]["output_plus_representation"]["average_precision"] > result["models"]["output_only"]["average_precision"]
    bad_provenance = {
        "split_role": "final_test", "model_name": "resnet50", "training_seed": 17,
        "config_sha256": "config", "condition_signature": "conditions", "pair_manifest_sha256": "final-pairs",
        "checkpoint_sha256": "checkpoint", "resolved_run_sha256": "run", "review_freeze_sha256": "review",
        "control_ecdf_sha256": "ecdf",
    }
    with pytest.raises(ValueError, match="calibration"):
        fit_nested_detectors(_detector_rows("final_test", True), cfg.detector, bad_provenance, tmp_path / "bad.json", 10)
    constant = _evaluate_detector(tmp_path, False)
    assert constant["models"]["output_plus_representation"]["average_precision"] == pytest.approx(constant["models"]["output_only"]["average_precision"])

def test_cluster_bootstrap_seed_summary_bh_and_wilcoxon() -> None:
    left = [{"pair_id": str(index), "source_group_id": f"g{index // 2}", "value": float(index + 1)} for index in range(8)]
    right = [{"pair_id": str(index), "source_group_id": f"g{index // 2}", "value": float(index)} for index in range(8)]
    result = cluster_bootstrap_difference(left, right, "value", 100, 0.95, 3)
    assert result["effect"] == pytest.approx(1.0)
    assert summarize_seed_estimates({1: 1.0, 2: 3.0})["sd"] == pytest.approx(np.sqrt(2))
    bh = benjamini_hochberg([0.01, 0.04, 0.2], 0.05)
    assert bh["rejected"][0]
    sensitivity = wilcoxon_sensitivity([2, 3, 4], [1, 1, 1])
    assert sensitivity["rank_biserial"] == pytest.approx(1.0)
    with pytest.raises(MetricUndefinedError):
        wilcoxon_sensitivity([1, 1], [1, 1])
    with pytest.raises(ValueError, match="identical pair IDs"):
        cluster_bootstrap_difference(left, right[:-1], "value", 10, 0.95, 1)


def test_absolute_detector_intervals_use_original_image_clusters() -> None:
    targets = [value for _ in range(10) for value in (0, 1)]
    scores = [0.1 if value == 0 else 0.9 for value in targets]
    groups = [f"group-{index}" for index in range(10) for _ in range(2)]
    result = clustered_detector_metric(targets, scores, groups, "average_precision", 200, 0.95, 19)
    assert result["defined"] and result["clusters"] == 10
    assert result["estimate"] == pytest.approx(1.0)
    assert result["lower"] == pytest.approx(1.0) and result["upper"] == pytest.approx(1.0)


def test_primary_statistics_clusters_sources_and_aggregates_all_three_seeds(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, ["statistics.bootstrap_samples=200", "evaluation.rq3_min_positives_per_model_seed=4"])
    digest = __import__("sac_qutab.config", fromlist=["config_digest"]).config_digest(cfg)
    run_names = [f"{preset.name}-seed{seed}" for preset in cfg.experiment.model_presets for seed in cfg.experiment.seeds]
    protocol_path = tmp_path / "protocol.json"
    atomic_write_json(protocol_path, {"schema_version": "sac-experiment-protocol-v4", "config_sha256": digest, "condition_signature": "conditions", "resolved_run_sha256": {name: f"run:{name}" for name in run_names}, "checkpoint_sha256": {name: f"checkpoint:{name}" for name in run_names}, "pair_manifest_sha256": {"final_test": "pairs"}, "review_freeze_sha256": "review"})
    protocol_sha = sha256_file(protocol_path)
    directories = []
    for preset in cfg.experiment.model_presets:
        for seed in cfg.experiment.seeds:
            run = f"{preset.name}-seed{seed}"
            directory = tmp_path / run
            directory.mkdir()
            rows = []
            for family in ("spectral_appearance", "texture", "resolution"):
                for source in range(8):
                    for severity, label, js, instability in (("mild", 1.0, 0.1, 0.2), ("severe", 0.75, 0.3, 0.5)):
                        rows.append({"pair_id": f"{run}-{family}-{source}-{severity}", "source_id": f"source-{source}", "source_group_id": f"group-{source // 2}", "family": family, "severity": severity, "latent_seed": source, "label_consistency": label, "js_divergence": js, "normalized_representation_instability": instability, "audit_qualified": True})
            write_jsonl(directory / "pair_metrics.jsonl", rows)
            atomic_write_json(directory / "provenance.json", {"schema_version": "sac-evaluation-v2", "split_role": "final_test", "model_name": preset.name, "training_seed": seed, "protocol_sha256": protocol_sha, "config_sha256": digest, "resolved_run_sha256": f"run:{run}", "checkpoint_sha256": f"checkpoint:{run}", "pair_manifest_sha256": "pairs", "review_freeze_sha256": "review", "condition_signature": "conditions"})
            atomic_write_json(directory / "metrics.json", {"conditions": {}})
            np.savez_compressed(directory / "representations.npz", values=np.zeros((1, 1)))
            population_rows = [{"pair_id": f"failure-{i}", "source_id": f"source-{i}", "source_group_id": f"group-{i // 2}", "family": "resolution", "severity": "mild", "class_name": "Forest", "target": int(i % 2), "scores": {"confidence_only": float(i % 2)}} for i in range(8)]
            atomic_write_json(directory / "detector_metrics.json", {"schema_version": "sac-detector-evaluation-v2", "model_name": preset.name, "training_seed": seed, "protocol_sha256": protocol_sha, "population": {"count": 8, "positives": 4, "prevalence": 0.5, "sham_count": 0}, "models": {}, "population_rows": population_rows, "omission_reason": "learned detectors unavailable"})
            atomic_write_json(directory / "evaluation_index.json", {"schema_version": "sac-evaluation-index-v1", "protocol_sha256": protocol_sha, "provenance_sha256": sha256_file(directory / "provenance.json"), "pair_metrics_sha256": sha256_file(directory / "pair_metrics.jsonl"), "metrics_sha256": sha256_file(directory / "metrics.json"), "representations_sha256": sha256_file(directory / "representations.npz")})
            directories.append(directory)
    output = tmp_path / "primary.json"
    result = generate_primary_statistics(cfg, directories, output, protocol_path=protocol_path)
    label_summaries = [row for row in result["primary_seed_summaries"] if row["contrast"] == "rq1_label_inconsistency_severe_minus_mild"]
    assert label_summaries and all(row["defined"] and row["mean"] == pytest.approx(0.25) for row in label_summaries)
    assert result["protocol_sha256"] == protocol_sha
    assert result["rq3_absolute_clustered"] and all(row["defined"] for row in result["rq3_absolute_clustered"])
    assert all(row["bootstrap_state"] == "defined" and "bootstrap_seed" in row for row in result["rq3_absolute_clustered"])
    pairwise = [row for row in result["primary_per_seed"] if row["contrast"] == "rq2_cross_model_difference_in_differences"]
    assert len(pairwise) == 10 * 3 * 3
    assert all(row["effect_direction"] == "right_minus_left" for row in pairwise)
    assert all(row["primary_test_defined"] is False and row["holm_rejected"] is False for row in pairwise)
    assert len(result["pairwise_model_summaries"]) == 10
    assert result["schema_version"] == "sac-primary-statistics-v3"


def test_holm_bonferroni_preserves_input_order_and_controls_family() -> None:
    result = holm_bonferroni([0.01, 0.04, 0.03, 0.20], 0.05)
    assert result["adjusted_p_values"] == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert result["rejected"] == [True, False, False, False]
