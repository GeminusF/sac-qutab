from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sac_qutab.config import load_config, materialize_run
from sac_qutab.runtime import atomic_write_json
from sac_qutab.tracking import (
    checkpoint_wandb_run_id,
    init_wandb_tracker,
    publish_detector_to_wandb,
    publish_evaluation_to_wandb,
    publish_hardware_preflight_to_wandb,
    publish_protocol_to_wandb,
    publish_report_to_wandb,
    publish_review_freeze_to_wandb,
    publish_statistics_to_wandb,
)


class FakeTable:
    def __init__(self, *, columns, data) -> None:
        self.columns = columns
        self.data = data


class FakeImage:
    def __init__(self, path: str, *, caption: str | None = None) -> None:
        self.path = path
        self.caption = caption


class FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))


class FakeRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.summary: dict[str, object] = {}
        self.logged: list[dict[str, object]] = []
        self.metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.artifacts: list[FakeArtifact] = []
        self.exit_code: int | None = None

    def define_metric(self, *args, **kwargs) -> None:
        self.metrics.append((args, kwargs))

    def log(self, values) -> None:
        self.logged.append(values)

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)


class FakeSdk:
    Table = FakeTable
    Image = FakeImage
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.util = SimpleNamespace(generate_id=lambda: "generated-id")
        self.kwargs = None
        self.run = None

    def init(self, **kwargs):
        self.kwargs = kwargs
        self.run = FakeRun(kwargs["id"])
        return self.run


def test_offline_wandb_initialization_logs_resolved_contract_and_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/core.yaml", ["wandb.mode=offline", "wandb.project=test-project", "wandb.log_interval_steps=1"])
    run = materialize_run(cfg, "resnet50", 17)
    sdk = FakeSdk()
    monkeypatch.setenv("WANDB_RUN_GROUP", "vast-smoke")
    tracker = init_wandb_tracker(run, tmp_path, sdk=sdk)
    assert tracker is not None and tracker.run_id == "generated-id"
    assert sdk.kwargs["mode"] == "offline" and sdk.kwargs["project"] == "test-project"
    assert sdk.kwargs["group"] == "vast-smoke" and sdk.kwargs["resume"] == "allow"
    assert sdk.kwargs["config"]["model"] == "resnet50"
    assert sdk.kwargs["config"]["model_contract"]["timm_model_id"] == "resnet50.tv_in1k"
    assert sdk.kwargs["config"]["resolved_run"]["effective_global_batch"] == 64
    tracker.log({"optimizer_step": 1, "train/loss": 2.0})
    tracker.summarize({"checkpoint/best_sha256": "abc"})
    tracker.finish(0)
    assert sdk.run.logged == [{"optimizer_step": 1, "train/loss": 2.0}]
    assert sdk.run.summary["checkpoint/best_sha256"] == "abc" and sdk.run.exit_code == 0
    metadata = json.loads((tmp_path / "wandb_run.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "sac-wandb-run-v2"
    assert metadata["run_id"] == "generated-id" and metadata["exit_code"] == 0
    assert metadata["job_type"] == "train" and "train/loss" in metadata["logged_keys"]


def test_resume_uses_checkpoint_identity_and_rejects_conflicting_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/core.yaml", ["wandb.mode=offline"])
    run = materialize_run(cfg, "resnet50", 17)
    assert checkpoint_wandb_run_id({"tracking": {"wandb_run_id": "saved-id"}}) == "saved-id"
    sdk = FakeSdk()
    tracker = init_wandb_tracker(run, tmp_path, resume_run_id="saved-id", sdk=sdk)
    assert tracker is not None and tracker.run_id == "saved-id"
    tracker.finish(1)
    monkeypatch.setenv("WANDB_RUN_ID", "different-id")
    with pytest.raises(RuntimeError, match="differs"):
        init_wandb_tracker(run, tmp_path / "conflict", resume_run_id="saved-id", sdk=FakeSdk())


def test_disabled_mode_does_not_import_or_initialize_sdk(tmp_path: Path) -> None:
    cfg = load_config("configs/core.yaml")
    run = materialize_run(cfg, "resnet50", 17)
    assert init_wandb_tracker(run, tmp_path) is None


def _evaluation_files(output: Path) -> None:
    atomic_write_json(output / "provenance.json", {"kind": "provenance"})
    atomic_write_json(output / "metrics.json", {"kind": "metrics"})
    atomic_write_json(output / "evaluation_index.json", {"kind": "index"})
    (output / "pair_metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (output / "representations.npz").write_bytes(b"representations")


def test_evaluation_statistics_and_report_publish_complete_ui_contract(tmp_path: Path) -> None:
    cfg = load_config("configs/core.yaml", ["wandb.mode=offline", "wandb.project=test-project"])
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    _evaluation_files(evaluation_dir)
    sdk = FakeSdk()
    provenance = {
        "split_role": "final_test", "model_name": "resnet50", "training_seed": 17,
        "resolved_run_sha256": "resolved", "protocol_sha256": "protocol",
        "pair_manifest_sha256": "pairs", "training_wandb_run_id": "training-run",
        "inference": {"forward_images_per_second": 321.0},
    }
    metrics = {
        "clean": {"accuracy": .9, "macro_f1": .8},
        "clean_calibration": {"ece": {"ece": .1}, "brier": .2, "nll": .3},
        "conditions": {"texture:mild": {
            "transformed_accuracy": .7, "label_consistency": .75, "js_mean": .2,
            "cosine_mean_within_model": .8, "normalized_instability_mean": .4,
            "failure_prevalence_among_clean_correct": .25,
        }},
    }
    publish_evaluation_to_wandb(cfg, evaluation_dir, provenance, metrics, sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "evaluation-final_test"
    assert any("evaluation/final_test/clean_accuracy" in row for row in sdk.run.logged)
    assert any("evaluation/final_test/condition_metrics" in row for row in sdk.run.logged)
    assert sdk.run.artifacts[0].metadata["large_pair_and_representation_files_uploaded"] is False
    assert {name for _, name in sdk.run.artifacts[0].files} == {"provenance.json", "metrics.json", "evaluation_index.json"}

    statistics_path = tmp_path / "statistics.json"
    statistics = {
        "protocol_sha256": "protocol", "cluster_unit": "source_group_id", "families": ["texture"],
        "primary_per_seed": [{"effect": .2}],
        "primary_seed_summaries": [{"mean": .2}],
        "pairwise_model_summaries": [{
            "left_model": "resnet50", "right_model": "convnext_tiny", "effect": .1,
            "p_value": .02, "holm_adjusted_p": .04, "holm_rejected": True,
        }],
        "rq3_absolute_clustered": [{"defined": False}],
        "exploratory_bh": [{"defined": False}],
    }
    atomic_write_json(statistics_path, statistics)
    sdk = FakeSdk()
    publish_statistics_to_wandb(cfg, statistics_path, statistics, sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "primary-statistics"
    assert any("statistics/pairwise/convnext_tiny_minus_resnet50/effect" in row for row in sdk.run.logged)
    assert len(sdk.run.artifacts) == 1

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "learning_curves.png").write_bytes(b"png")
    (report_dir / "master_summary.csv").write_text("model,accuracy\nresnet50,0.9\n", encoding="utf-8")
    index = {
        "schema_version": "sac-report-artifacts-v4", "protocol_sha256": "protocol",
        "generated": ["learning_curves.png", "master_summary.csv"],
    }
    atomic_write_json(report_dir / "index.json", index)
    sdk = FakeSdk()
    publish_report_to_wandb(cfg, report_dir, index, sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "report-artifacts"
    assert any("report/plots/learning_curves" in row for row in sdk.run.logged)
    assert any("report/tables/master_summary" in row for row in sdk.run.logged)
    assert {name for _, name in sdk.run.artifacts[0].files} == {
        "index.json", "learning_curves.png", "master_summary.csv",
    }


def test_governance_and_detector_stages_publish_metrics_and_lineage(tmp_path: Path) -> None:
    cfg = load_config("configs/core.yaml", ["wandb.mode=offline", "wandb.project=test-project"])

    preflight_path = tmp_path / "preflight.json"
    measurement = {
        "device": {"name": "NVIDIA test", "cuda_runtime": "12.8"},
        "precision": "bf16", "batch_size_per_device": 8, "grad_accum_steps": 8,
        "effective_global_batch": 64, "peak_gpu_reserved_bytes": 1_000,
        "median_optimizer_step_seconds_rank0": .5, "projected_training_validation_hours": 2.0,
        "pretrained_checkpoint_sha256": "weights",
    }
    preflight = {
        "status": "ready_for_training", "config_sha256": "config", "world_size": 1,
        "resolved_run_sha256": {f"run-{index}": str(index) for index in range(15)},
        "projected_training_and_validation_hours": 30.0,
        "summaries": {preset.name: {"measurements": measurement} for preset in cfg.experiment.model_presets},
    }
    atomic_write_json(preflight_path, preflight)
    sdk = FakeSdk()
    publish_hardware_preflight_to_wandb(cfg, preflight_path, preflight, sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "hardware-preflight"
    assert any("hardware/preflight_models" in row for row in sdk.run.logged)

    panel = tmp_path / "panel.png"
    panel.write_bytes(b"panel")
    review_path = tmp_path / "review.json"
    review = {
        "condition_signature": "conditions", "rules": {"minimum": .8},
        "qualified_primary_families": ["texture"], "cohen_kappa": .9, "raw_agreement": .95,
        "included_conditions": ["texture:mild"],
        "conditions": {"texture:mild": {"rate": .9, "eligible": True}},
        "qualitative_panel_path": str(panel), "qualitative_panel_sha256": "panel-sha",
        "pair_manifest_sha256": "pairs",
    }
    atomic_write_json(review_path, review)
    sdk = FakeSdk()
    publish_review_freeze_to_wandb(cfg, review_path, review, sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "semantic-review-freeze"
    assert any("audit/qualitative_panel" in row for row in sdk.run.logged)

    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "config_sha256": "config", "review_freeze_sha256": "review",
        "resolved_run_sha256": {"resnet50-seed17": "resolved"},
        "model_contracts": {"resnet50-seed17": {"model": {"name": "resnet50"}}},
        "metric_contract": {"metric": "frozen"}, "statistics_contract": {"alpha": .05},
        "report_plan": ["learning_curves"],
    }
    atomic_write_json(protocol_path, protocol)
    sdk = FakeSdk()
    publish_protocol_to_wandb(cfg, protocol_path, protocol, sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "experiment-freeze"
    assert any("protocol/model_run_contracts" in row for row in sdk.run.logged)

    fit_path = tmp_path / "detector-fit.json"
    fit = {
        "model_name": "resnet50", "training_seed": 17, "training_wandb_run_id": "training-run",
        "fit_population": {"count": 100, "positives": 25, "prevalence": .25},
        "learned_detectors_defined": True,
        "models": {"output_only": {"features": ["uncertainty"], "coefficient": [.5]}},
    }
    atomic_write_json(fit_path, fit)
    sdk = FakeSdk()
    publish_detector_to_wandb(cfg, fit_path, fit, stage="detector-fit", sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "detector-fit"
    assert any("detector/fitted_models" in row for row in sdk.run.logged)

    evaluation_path = tmp_path / "detector-evaluation.json"
    detector_evaluation = {
        "model_name": "resnet50", "training_seed": 17, "training_wandb_run_id": "training-run",
        "protocol_sha256": "protocol", "population": {"count": 50, "positives": 10, "prevalence": .2},
        "models": {"output_only": {"defined": True, "auroc": .8, "average_precision": .7}},
        "condition_prevalence": {"texture:mild": {"count": 10, "positives": 2, "prevalence": .2}},
    }
    atomic_write_json(evaluation_path, detector_evaluation)
    sdk = FakeSdk()
    publish_detector_to_wandb(cfg, evaluation_path, detector_evaluation, stage="detector-evaluation", sdk=sdk)
    assert sdk.run is not None and sdk.kwargs["job_type"] == "detector-evaluation"
    assert any("detector/final_test/output_only/auroc" in row for row in sdk.run.logged)
    assert sdk.run.summary["lineage/training_wandb_run_id"] == "training-run"
