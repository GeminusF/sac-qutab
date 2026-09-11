from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sac_qutab.config import REPORT_ARTIFACT_PLAN, Config, config_digest, load_config, materialize_run
from sac_qutab.evaluate import _infer_stream
from sac_qutab.interventions import pair_condition_signature
from sac_qutab.models import ModelOutput
from sac_qutab.reporting import generate_report_artifacts
from sac_qutab.runtime import atomic_write_json, sha256_file, write_jsonl


def test_inference_stream_is_bounded_and_order_preserving() -> None:
    state = {"resident": 0, "maximum": 0}

    class TrackingModel(nn.Module):
        def forward(self, batch: torch.Tensor) -> ModelOutput:
            state["maximum"] = max(state["maximum"], state["resident"])
            state["resident"] = 0
            values = batch[:, 0]
            return ModelOutput(logits=torch.stack((values, -values), dim=1), representation=values[:, None] * 2)

    def load_item(value: int) -> torch.Tensor:
        state["resident"] += 1
        return torch.tensor([float(value)])

    items = list(range(10_000))
    probabilities, features = _infer_stream(TrackingModel(), items, load_item, torch.device("cpu"), 37)
    expected_logits = np.stack((np.asarray(items, dtype=float), -np.asarray(items, dtype=float)), axis=1)
    expected = np.exp(expected_logits - expected_logits.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    assert state["maximum"] <= 37
    assert np.allclose(probabilities, expected, atol=1e-7)
    assert np.array_equal(features[:, 0], np.asarray(items) * 2)


def _publication_fixture(tmp_path: Path) -> tuple[Config, list[Path], Path, Path, Path, list[Path]]:
    cfg = load_config("configs/core.yaml")
    condition_signature = pair_condition_signature(cfg)
    panel = tmp_path / "panel.png"
    panel.write_bytes(b"panel")
    review = tmp_path / "review.json"
    cell = {"accepted": cfg.audit.samples_per_class_condition, "total": cfg.audit.samples_per_class_condition, "rate": 1.0}
    conditions = {
        f"{family}:{severity}": {
            "accepted": len(cfg.data.class_names) * cfg.audit.samples_per_class_condition,
            "total": len(cfg.data.class_names) * cfg.audit.samples_per_class_condition,
            "rate": 1.0, "class_rates": {name: dict(cell) for name in cfg.data.class_names},
            "class_veto": False, "eligible": True,
        }
        for family in ("spectral_appearance", "texture", "resolution")
        for severity in ("mild", "moderate", "severe")
    }
    included = sorted(conditions)
    atomic_write_json(review, {
        "schema_version": "sac-review-freeze-v2", "config_sha256": config_digest(cfg),
        "condition_signature": condition_signature, "cohen_kappa": 1.0, "kappa_scope": cfg.audit.kappa_scope,
        "rules": {"exclusion_granularity": cfg.audit.exclusion_granularity, "min_kappa": cfg.audit.min_kappa,
                  "min_condition_acceptance_rate": cfg.audit.min_condition_acceptance_rate,
                  "min_class_acceptance_rate": cfg.audit.min_class_acceptance_rate,
                  "disagreement_policy": cfg.audit.disagreement_policy,
                  "downstream_description": "whole family×severity conditions only"},
        "conditions": conditions, "included_conditions": included,
        "qualified_primary_families": ["spectral_appearance", "texture", "resolution"],
        "qualitative_panel_path": str(panel), "qualitative_panel_sha256": sha256_file(panel),
    })
    review_sha = sha256_file(review)
    run_names = [f"{preset.name}-seed{seed}" for preset in cfg.experiment.model_presets for seed in cfg.experiment.seeds]
    resolved_hashes = {
        f"{preset.name}-seed{seed}": config_digest(materialize_run(cfg, preset.name, seed))
        for preset in cfg.experiment.model_presets for seed in cfg.experiment.seeds
    }
    protocol = tmp_path / "protocol.json"
    atomic_write_json(protocol, {
        "schema_version": "sac-experiment-protocol-v4", "config_sha256": config_digest(cfg), "condition_signature": condition_signature,
        "resolved_run_sha256": resolved_hashes,
        "checkpoint_sha256": {name: f"checkpoint:{name}" for name in run_names},
        "detector_sha256": {name: f"detector:{name}" for name in run_names},
        "control_ecdf_sha256": {name: f"ecdf:{name}" for name in run_names},
        "pair_manifest_sha256": {"validation": "validation", "calibration": "calibration", "final_test": "pairs"},
        "review_freeze_sha256": review_sha,
        "metric_contract": {"ece_bins": cfg.evaluation.ece_bins, "rq3_population": "selected non-sham pairs whose clean prediction is correct, per model seed", "rq3_min_positives_calibration_and_test": cfg.evaluation.rq3_min_positives_per_model_seed, "pr_summary": cfg.evaluation.pr_summary, "normalization": cfg.evaluation.representation_normalization},
        "statistics_contract": {"primary_contrasts": list(cfg.statistics.primary_contrasts), "model_pair_contrasts": [{"left": pair.left, "right": pair.right} for pair in cfg.statistics.model_pair_contrasts], "cluster_unit": "source_group_id", "bootstrap_samples": cfg.statistics.bootstrap_samples, "confidence_level": cfg.statistics.confidence_level, "primary_familywise_alpha": cfg.statistics.primary_familywise_alpha, "seed_summary": "mean_and_sample_sd", "exploratory_fdr_q": cfg.statistics.fdr_q},
        "report_plan": list(REPORT_ARTIFACT_PLAN),
    })
    protocol_sha = sha256_file(protocol)
    conditions = {}
    pair_rows = []
    for family in ("spectral_appearance", "texture", "resolution"):
        for severity in ("sham", "mild", "moderate", "severe"):
            condition = f"{family}:{severity}"
            calibration = {"ece": {"ece": 0.0, "bins": [{"count": 1, "mean_confidence": 0.9, "accuracy": 1.0}]}, "brier": 0.01, "nll": 0.1}
            conditions[condition] = {
                "count": 1, "unique_source_count": 1, "transformed_accuracy": 1.0, "label_consistency": 1.0,
                "js_mean": 0.0, "entropy_mean": 0.1, "cosine_mean_within_model": 1.0, "control_margin_mean_within_model": 0.5,
                "normalized_instability_mean": 0.1, "failure_count": 0, "clean_correct_count": 1,
                "failure_prevalence_among_clean_correct": 0.0,
                "confidence_change": {"mean_signed": 0.0, "mean_absolute": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0, "increased_confidence_failure_fraction": 0.0},
                "calibration": calibration, "linear_cka": 1.0,
                "class_cells": {"Forest": {"count": 1, "label_consistency": 1.0, "failure_rate": 0.0, "calibration_ece_clean": 0.0, "calibration_ece_transformed": 0.0, "calibration_ece_change": 0.0}},
            }
            pair_rows.append({"pair_id": condition, "source_id": "source", "source_group_id": "group", "family": family, "severity": severity, "class_name": "Forest", "label": 0, "audit_qualified": True, "original_probabilities": [0.9, 0.1], "transformed_probabilities": [0.9, 0.1], "confidence_change": 0.0, "label_consistency": 1.0, "cosine_similarity": 1.0, "normalized_representation_instability": 0.1})
    directories = []
    sources = []
    rq3 = []
    for run_name in run_names:
        model, seed_text = run_name.rsplit("-seed", 1)
        seed = int(seed_text)
        directory = tmp_path / "evaluations" / run_name
        directory.mkdir(parents=True)
        provenance = {"schema_version": "sac-evaluation-v2", "split_role": "final_test", "model_name": model, "training_seed": seed, "config_sha256": config_digest(cfg), "condition_signature": condition_signature, "checkpoint_sha256": f"checkpoint:{run_name}", "resolved_run_sha256": resolved_hashes[run_name], "pair_manifest_sha256": "pairs", "review_freeze_sha256": review_sha, "protocol_sha256": protocol_sha, "inference": {"device_type": "cpu", "forward_images_per_second": 10.0}}
        atomic_write_json(directory / "provenance.json", provenance)
        write_jsonl(directory / "pair_metrics.jsonl", pair_rows)
        atomic_write_json(directory / "metrics.json", {"clean": {"count": 1, "accuracy": 1.0, "macro_f1": 1.0}, "clean_calibration": {"ece": {"ece": 0.0}, "brier": 0.01, "nll": 0.1}, "conditions": conditions})
        np.savez_compressed(directory / "representations.npz", original=np.full((len(pair_rows), 2), run_names.index(run_name)+1.0), transformed=np.full((len(pair_rows), 2), run_names.index(run_name)+1.0))
        atomic_write_json(directory / "detector_metrics.json", {"models": {}, "condition_prevalence": {"resolution:mild": {"count": 1, "positives": 0, "prevalence": 0.0}}})
        atomic_write_json(directory / "evaluation_index.json", {"schema_version": "sac-evaluation-index-v1", "protocol_sha256": protocol_sha, "provenance_sha256": sha256_file(directory / "provenance.json"), "pair_metrics_sha256": sha256_file(directory / "pair_metrics.jsonl"), "metrics_sha256": sha256_file(directory / "metrics.json"), "representations_sha256": sha256_file(directory / "representations.npz")})
        sources.append({"run": run_name, "provenance_sha256": sha256_file(directory / "provenance.json"), "pairs_sha256": sha256_file(directory / "pair_metrics.jsonl"), "metrics_sha256": sha256_file(directory / "metrics.json"), "representations_sha256": sha256_file(directory / "representations.npz"), "evaluation_index_sha256": sha256_file(directory / "evaluation_index.json"), "detector_sha256": sha256_file(directory / "detector_metrics.json")})
        for metric in ("average_precision", "auroc"):
            rq3.append({"model": model, "seed": seed, "score": "confidence_only", "metric": metric, "count": 1, "positives": 0, "prevalence": 0.0, "underpowered": True, "defined": False, "bootstrap_state": "underpowered", "bootstrap_seed": seed + 3000, "reason": "fixture underpowered"})
        directories.append(directory)
    pairwise = [{"left_model": pair.left, "right_model": pair.right, "model": f"{pair.right}_minus_{pair.left}", "effect_direction": "right_minus_left", "effect": 0.0, "ci": [-0.1, 0.1], "holm_rejected": False} for pair in cfg.statistics.model_pair_contrasts]
    detector_summaries = [{"contrast": "rq3_nested_detector_delta_auroc", "model": preset.name, "family": "all_accepted_non_sham", "defined": False, "reason": "fixture"} for preset in cfg.experiment.model_presets]
    statistics = tmp_path / "statistics.json"
    atomic_write_json(statistics, {"schema_version": "sac-primary-statistics-v3", "protocol_source": {"path": str(protocol), "sha256": protocol_sha}, "protocol_sha256": protocol_sha, "sources": sources, "primary_per_seed": [{"contrast": "rq1", "model": "resnet50", "seed": 17, "family": "resolution", "effect": 0.0}], "primary_seed_summaries": detector_summaries, "pairwise_model_summaries": pairwise, "rq3_absolute_clustered": rq3, "exploratory_bh": [{"model": "resnet50", "seed": 17, "family": "resolution", "metric": "js", "defined": False, "reason": "fixture"}]})
    training_dirs = []
    preset_map = {preset.name: preset for preset in cfg.experiment.model_presets}
    for run_name in run_names:
        model, seed_text = run_name.rsplit("-seed", 1)
        run_dir = tmp_path / "runs" / run_name
        run_dir.mkdir(parents=True)
        atomic_write_json(run_dir / "resolved_run.json", materialize_run(cfg, model, int(seed_text)).canonical())
        preset = preset_map[model]
        atomic_write_json(run_dir / "model.json", {"name": model, "checkpoint_sha256": preset.checkpoint_sha256, "adapter_version": preset.adapter_version, "expected_representation_dim": preset.expected_representation_dim, "parameter_count": preset.expected_representation_dim * 1000})
        write_jsonl(run_dir / "events.jsonl", [{"event": "optimizer_step", "epoch": 0, "loss": 1.0, "global_step": 1, "learning_rate": preset.learning_rate}, {"event": "validation", "epoch": 0, "loss": 0.9, "accuracy": 0.8, "macro_f1": 0.75, "best": True}])
        training_dirs.append(run_dir)
    return cfg, directories, protocol, statistics, review, training_dirs


def test_report_rejects_protocol_mismatch_before_publication(tmp_path: Path) -> None:
    cfg, directories, protocol, statistics, review, training_dirs = _publication_fixture(tmp_path)
    provenance_path = directories[0] / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["checkpoint_sha256"] = "tampered"
    atomic_write_json(provenance_path, provenance)
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="real protocol"):
        generate_report_artifacts(cfg, directories, output, protocol_path=protocol, statistics_path=statistics, review_freeze_path=review, training_run_dirs=training_dirs)
    assert not output.exists()


def test_complete_configured_fifteen_run_report_matrix_passes(tmp_path: Path) -> None:
    cfg, directories, protocol, statistics, review, training_dirs = _publication_fixture(tmp_path)
    output = tmp_path / "report"
    result = generate_report_artifacts(cfg, directories, output, protocol_path=protocol, statistics_path=statistics, review_freeze_path=review, training_run_dirs=training_dirs)
    assert result["evaluation_count"] == 15
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["protocol_sha256"] == sha256_file(protocol)
    for source, directory in zip(index["sources"], directories, strict=True):
        assert source["representations_sha256"] == sha256_file(directory / "representations.npz")
    assert (output / "detector_metrics.csv").is_file() and (output / "exploratory_bh.csv").is_file()
    assert "bootstrap_seed" in (output / "detector_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert (output / "learning_curves.png").is_file()
    assert (output / "master_summary.tex").is_file()
    assert (output / "pairwise_forest_plot.png").is_file()
    assert (output / "cross_model_cka_matrix.png").is_file()
    assert (output / "throughput_vs_robustness.png").is_file()
    assert (output / "qualitative_failure_gallery.png").is_file()
    assert (output / "attention_rollout.png").is_file()
    assert (output / "macro_cluster_confusion.png").is_file()


def test_report_rejects_protocol_from_different_config(tmp_path: Path) -> None:
    cfg, directories, protocol, statistics, review, training_dirs = _publication_fixture(tmp_path)
    changed_cfg = load_config("configs/core.yaml", ["train.epochs=49"])
    with pytest.raises(ValueError, match="configured run-matrix"):
        generate_report_artifacts(changed_cfg, directories, tmp_path / "report", protocol_path=protocol, statistics_path=statistics, review_freeze_path=review, training_run_dirs=training_dirs)
