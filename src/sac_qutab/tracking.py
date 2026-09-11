from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Config, ResolvedRun, WandbConfig, as_dict, config_digest
from .runtime import atomic_write_json, canonical_json, environment_metadata, runtime_path, sha256_file


def _wandb_mode(cfg: WandbConfig) -> str:
    mode = os.environ.get("WANDB_MODE", cfg.mode).strip().lower()
    if mode not in {"online", "offline", "disabled"}:
        raise RuntimeError("WANDB_MODE/wandb.mode must be online, offline, or disabled")
    return mode


def _wandb_sdk(sdk: Any | None) -> Any:
    if sdk is not None:
        return sdk
    try:
        import wandb as imported_sdk
    except ImportError as exc:
        raise RuntimeError("W&B logging is enabled but the pinned 'wandb' dependency is not installed") from exc
    return imported_sdk


def _wandb_coordinates(cfg: WandbConfig) -> tuple[str, str | None, str | None]:
    project = os.environ.get("WANDB_PROJECT") or cfg.project
    entity = os.environ.get("WANDB_ENTITY") or cfg.entity
    group = os.environ.get("WANDB_RUN_GROUP") or cfg.group
    return project, entity, group


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "sac-qutab"


def _table_cell(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return value


class WandbTracker:
    """One failure-isolated W&B adapter shared by every pipeline stage."""

    def __init__(
        self, sdk: Any, run: Any, mode: str, metadata_path: Path, metadata: Mapping[str, Any],
    ) -> None:
        self._sdk = sdk
        self._run = run
        self.mode = mode
        self.metadata_path = metadata_path
        self.error: str | None = None
        self._metadata = dict(metadata)
        self._logged_keys: set[str] = set()
        self._logged_artifacts: list[dict[str, Any]] = []
        self._write_metadata(exit_code=None)

    @property
    def run_id(self) -> str:
        return str(self._run.id)

    def _write_metadata(self, *, exit_code: int | None) -> None:
        atomic_write_json(self.metadata_path, {
            "schema_version": "sac-wandb-run-v2",
            **self._metadata,
            "mode": self.mode,
            "run_id": self.run_id,
            "error": self.error,
            "exit_code": exit_code,
            "logged_keys": sorted(self._logged_keys),
            "logged_artifacts": self._logged_artifacts,
        })

    def _record_failure(self, operation: str, exc: Exception) -> None:
        self.error = f"{operation}: {type(exc).__name__}: {exc}"
        print(f"warning: W&B integration disabled after failure: {self.error}", file=sys.stderr)

    def log(self, values: dict[str, Any]) -> None:
        if self.error is not None:
            return
        try:
            self._run.log(values)
            self._logged_keys.update(values)
        except Exception as exc:  # W&B must not hide or replace a pipeline exception.
            self._record_failure("log", exc)

    def define_metric(self, name: str, **kwargs: Any) -> None:
        if self.error is not None:
            return
        try:
            self._run.define_metric(name, **kwargs)
        except Exception as exc:
            self._record_failure(f"define_metric:{name}", exc)

    def summarize(self, values: dict[str, Any]) -> None:
        if self.error is not None:
            return
        try:
            self._run.summary.update(values)
            self._logged_keys.update(values)
        except Exception as exc:
            self._record_failure("summary", exc)

    def log_table(self, key: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.error is not None or not rows:
            return
        try:
            columns = sorted({column for row in rows for column in row})
            data = [[_table_cell(row.get(column)) for column in columns] for row in rows]
            self._run.log({key: self._sdk.Table(columns=columns, data=data)})
            self._logged_keys.add(key)
        except Exception as exc:
            self._record_failure(f"table:{key}", exc)

    def log_image(self, key: str, path: str | Path, *, caption: str | None = None) -> None:
        if self.error is not None:
            return
        try:
            self._run.log({key: self._sdk.Image(str(Path(path)), caption=caption)})
            self._logged_keys.add(key)
        except Exception as exc:
            self._record_failure(f"image:{key}", exc)

    def log_artifact_files(
        self,
        name: str,
        artifact_type: str,
        files: Sequence[str | Path],
        *,
        metadata: Mapping[str, Any],
        root: str | Path | None = None,
    ) -> None:
        if self.error is not None:
            return
        root_path = Path(root).resolve() if root is not None else None
        try:
            artifact = self._sdk.Artifact(
                name=_slug(name), type=_slug(artifact_type), metadata=dict(metadata),
            )
            file_records: list[dict[str, str]] = []
            for value in sorted((Path(item) for item in files), key=lambda item: str(item)):
                resolved = value.resolve()
                logical_name = (
                    str(resolved.relative_to(root_path)).replace("\\", "/")
                    if root_path is not None else resolved.name
                )
                artifact.add_file(str(resolved), name=logical_name)
                file_records.append({"name": logical_name, "sha256": sha256_file(resolved)})
            self._run.log_artifact(artifact)
            self._logged_artifacts.append({
                "name": _slug(name), "type": _slug(artifact_type), "files": file_records,
            })
        except Exception as exc:
            self._record_failure(f"artifact:{name}", exc)

    def finish(self, exit_code: int) -> None:
        effective_exit_code = 1 if self.error is not None and exit_code == 0 else exit_code
        try:
            self._run.finish(exit_code=effective_exit_code)
        except Exception as exc:
            message = f"finish: {type(exc).__name__}: {exc}"
            self.error = self.error or message
            print(f"warning: W&B cleanup failed: {message}", file=sys.stderr)
        finally:
            self._write_metadata(exit_code=effective_exit_code)


def _open_tracker(
    cfg: WandbConfig,
    run_dir: Path,
    *,
    run_id: str,
    name: str,
    stage: str,
    resume: str,
    run_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    metadata_filename: str,
    sdk: Any | None,
) -> WandbTracker | None:
    mode = _wandb_mode(cfg)
    if mode == "disabled":
        return None
    sdk = _wandb_sdk(sdk)
    project, entity, group = _wandb_coordinates(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        wandb_run = sdk.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=stage,
            id=run_id,
            resume=resume,
            mode=mode,
            dir=str(run_dir),
            config=dict(run_config),
        )
        if wandb_run is None:
            raise RuntimeError("wandb.init returned no run")
    except Exception as exc:
        raise RuntimeError(f"W&B initialization failed in {mode} mode for {stage}: {exc}") from exc
    return WandbTracker(sdk, wandb_run, mode, run_dir / metadata_filename, {
        **metadata,
        "stage": stage,
        "project": project,
        "entity": entity,
        "name": name,
        "group": group,
        "job_type": stage,
    })


def checkpoint_wandb_run_id(checkpoint: dict[str, Any]) -> str | None:
    tracking = checkpoint.get("tracking")
    if not isinstance(tracking, dict):
        return None
    value = tracking.get("wandb_run_id")
    return value if isinstance(value, str) and value.strip() else None


def _generate_wandb_run_id(sdk: Any | None = None) -> str:
    if sdk is not None:
        util = getattr(sdk, "util", None)
        if util is not None and callable(getattr(util, "generate_id", None)):
            return str(util.generate_id())
        sdk_mod = getattr(sdk, "sdk", None)
        if sdk_mod is not None:
            lib = getattr(sdk_mod, "lib", None)
            if lib is not None:
                runid = getattr(lib, "runid", None)
                if runid is not None and callable(getattr(runid, "generate_id", None)):
                    return str(runid.generate_id())
    import secrets
    import string
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))


def init_wandb_tracker(
    run: ResolvedRun,
    run_dir: Path,
    *,
    resume_run_id: str | None = None,
    sdk: Any | None = None,
) -> WandbTracker | None:
    cfg = run.config.wandb
    mode = _wandb_mode(cfg)
    if mode == "disabled":
        return None
    loaded_sdk = _wandb_sdk(sdk)
    name = os.environ.get("WANDB_NAME") or cfg.run_name or f"{run.model.name}-seed{run.seed}"
    env_run_id = os.environ.get("WANDB_RUN_ID")
    selected_id = resume_run_id or env_run_id or _generate_wandb_run_id(loaded_sdk)
    if resume_run_id and env_run_id and resume_run_id != env_run_id:
        raise RuntimeError("WANDB_RUN_ID differs from the run identity stored in the training checkpoint")
    tracker = _open_tracker(
        cfg,
        run_dir,
        run_id=selected_id,
        name=name,
        stage="train",
        resume="allow",
        run_config={
            "pipeline_stage": "train",
            "resolved_run": as_dict(run),
            "config_sha256": config_digest(run.config),
            "resolved_run_sha256": config_digest(run),
            "campaign_name": os.environ.get("SAC_QUTAB_CAMPAIGN_NAME"),
            "campaign_sha256": os.environ.get("SAC_QUTAB_CAMPAIGN_SHA256"),
            "model": run.model.name,
            "model_contract": {
                "timm_model_id": run.model.timm_model_id,
                "pretraining": run.model.pretraining,
                "checkpoint_sha256": run.model.checkpoint_sha256,
                "source_input_size": run.model.source_input_size,
                "target_input_size": run.config.data.input_size,
                "adapter_version": run.model.adapter_version,
                "representation": run.model.representation,
                "expected_representation_dim": run.model.expected_representation_dim,
                "learning_rate": run.model.learning_rate,
            },
            "seed": run.seed,
            "execution_provider": os.environ.get("SAC_EXECUTION_PROVIDER", "unspecified"),
            "code_and_environment": environment_metadata(),
        },
        metadata={
            "resolved_run_sha256": config_digest(run),
            "model": run.model.name,
            "seed": run.seed,
            "resumed_from_checkpoint": resume_run_id is not None,
            "campaign_sha256": os.environ.get("SAC_QUTAB_CAMPAIGN_SHA256"),
        },
        metadata_filename="wandb_run.json",
        sdk=loaded_sdk,
    )
    assert tracker is not None
    tracker.define_metric("optimizer_step")
    tracker.define_metric("epoch")
    tracker.define_metric("train/*", step_metric="optimizer_step")
    tracker.define_metric("validation/*", step_metric="optimizer_step")
    return tracker


def init_wandb_stage_tracker(
    cfg: Config,
    stage: str,
    output_dir: str | Path,
    *,
    identity: Mapping[str, Any],
    display_name: str,
    stage_config: Mapping[str, Any],
    sdk: Any | None = None,
) -> WandbTracker | None:
    mode = _wandb_mode(cfg.wandb)
    if mode == "disabled":
        return None
    digest = hashlib.sha256(canonical_json({"stage": stage, "identity": identity}).encode("utf-8")).hexdigest()
    base_name = os.environ.get("WANDB_NAME") or cfg.wandb.run_name
    name = f"{base_name}-{display_name}-{stage}" if base_name else f"{display_name}-{stage}"
    return _open_tracker(
        cfg.wandb,
        Path(output_dir),
        run_id=f"sac{digest[:13]}",
        name=name,
        stage=stage,
        resume="allow",
        run_config={
            "pipeline_stage": stage,
            "config_sha256": config_digest(cfg),
            "campaign_name": os.environ.get("SAC_QUTAB_CAMPAIGN_NAME"),
            "campaign_sha256": os.environ.get("SAC_QUTAB_CAMPAIGN_SHA256"),
            "identity": dict(identity),
            "stage_contract": dict(stage_config),
            "execution_provider": os.environ.get("SAC_EXECUTION_PROVIDER", "unspecified"),
            "code_and_environment": environment_metadata(),
        },
        metadata={"identity": dict(identity), "campaign_sha256": os.environ.get("SAC_QUTAB_CAMPAIGN_SHA256")},
        metadata_filename=f"wandb_{_slug(display_name)}_{_slug(stage)}.json",
        sdk=sdk,
    )


def _campaign_display_name() -> str:
    return os.environ.get("SAC_QUTAB_CAMPAIGN_NAME", "five-model")


def publish_hardware_preflight_to_wandb(
    cfg: Config, output_path: str | Path, artifact: Mapping[str, Any], *, sdk: Any | None = None,
) -> None:
    output_path = Path(output_path)
    preflight_sha = sha256_file(output_path)
    tracker = init_wandb_stage_tracker(
        cfg, "hardware-preflight", output_path.parent,
        identity={"preflight_sha256": preflight_sha}, display_name=_campaign_display_name(),
        stage_config={"world_size": artifact["world_size"], "run_count": len(artifact["resolved_run_sha256"])},
        sdk=sdk,
    )
    if tracker is None:
        return
    succeeded = False
    try:
        rows = []
        for model, source in sorted(artifact["summaries"].items()):
            values = source["measurements"]
            rows.append({
                "model": model,
                "gpu_name": values["device"]["name"],
                "cuda_runtime": values["device"]["cuda_runtime"],
                "precision": values["precision"],
                "physical_batch": values["batch_size_per_device"],
                "gradient_accumulation": values["grad_accum_steps"],
                "effective_global_batch": values["effective_global_batch"],
                "peak_reserved_bytes": values["peak_gpu_reserved_bytes"],
                "median_optimizer_step_seconds": values["median_optimizer_step_seconds_rank0"],
                "projected_training_validation_hours": values["projected_training_validation_hours"],
                "weights_sha256": values["pretrained_checkpoint_sha256"],
            })
        tracker.log_table("hardware/preflight_models", rows)
        tracker.log({
            "hardware/projected_campaign_hours": artifact["projected_training_and_validation_hours"],
            "hardware/world_size": artifact["world_size"],
            "hardware/model_count": len(rows),
        })
        tracker.summarize({
            "hardware/status": artifact["status"],
            "hardware/preflight_sha256": preflight_sha,
            "hardware/projected_campaign_hours": artifact["projected_training_and_validation_hours"],
        })
        tracker.log_artifact_files(
            f"sac-qutab-hardware-preflight-{preflight_sha[:12]}", "hardware-preflight", [output_path],
            metadata={"config_sha256": artifact["config_sha256"], "preflight_sha256": preflight_sha},
        )
        succeeded = True
    finally:
        tracker.finish(0 if succeeded else 1)


def publish_review_freeze_to_wandb(
    cfg: Config, output_path: str | Path, artifact: Mapping[str, Any], *, sdk: Any | None = None,
) -> None:
    output_path = Path(output_path)
    freeze_sha = sha256_file(output_path)
    tracker = init_wandb_stage_tracker(
        cfg, "semantic-review-freeze", output_path.parent,
        identity={"review_freeze_sha256": freeze_sha, "condition_signature": artifact["condition_signature"]},
        display_name="counterfactual-audit",
        stage_config={"rules": artifact["rules"], "qualified_primary_families": artifact["qualified_primary_families"]},
        sdk=sdk,
    )
    if tracker is None:
        return
    succeeded = False
    try:
        condition_rows = []
        scalars: dict[str, Any] = {
            "audit/cohen_kappa": artifact["cohen_kappa"],
            "audit/raw_agreement": artifact["raw_agreement"],
            "audit/included_condition_count": len(artifact["included_conditions"]),
        }
        for condition, values in sorted(artifact["conditions"].items()):
            family, severity = condition.split(":", 1)
            condition_rows.append({"family": family, "severity": severity, **values})
            scalars[f"audit/{family}/{severity}/acceptance_rate"] = values["rate"]
            scalars[f"audit/{family}/{severity}/eligible"] = int(bool(values["eligible"]))
        tracker.log(scalars)
        tracker.log_table("audit/condition_acceptance", condition_rows)
        panel_path = runtime_path(artifact["qualitative_panel_path"])
        tracker.log_image("audit/qualitative_panel", panel_path, caption="Frozen blinded semantic-preservation audit panel")
        tracker.summarize({
            **scalars,
            "audit/review_freeze_sha256": freeze_sha,
            "audit/qualitative_panel_sha256": artifact["qualitative_panel_sha256"],
        })
        tracker.log_artifact_files(
            f"sac-qutab-semantic-review-{freeze_sha[:12]}", "semantic-review",
            [output_path, panel_path],
            metadata={
                "review_freeze_sha256": freeze_sha,
                "pair_manifest_sha256": artifact["pair_manifest_sha256"],
                "condition_signature": artifact["condition_signature"],
            },
        )
        succeeded = True
    finally:
        tracker.finish(0 if succeeded else 1)


def publish_protocol_to_wandb(
    cfg: Config, output_path: str | Path, artifact: Mapping[str, Any], *, sdk: Any | None = None,
) -> None:
    output_path = Path(output_path)
    protocol_sha = sha256_file(output_path)
    tracker = init_wandb_stage_tracker(
        cfg, "experiment-freeze", output_path.parent,
        identity={"protocol_sha256": protocol_sha}, display_name=_campaign_display_name(),
        stage_config={
            "run_count": len(artifact["resolved_run_sha256"]),
            "metric_contract": artifact["metric_contract"],
            "statistics_contract": artifact["statistics_contract"],
            "report_plan": artifact["report_plan"],
        },
        sdk=sdk,
    )
    if tracker is None:
        return
    succeeded = False
    try:
        tracker.log({
            "protocol/run_count": len(artifact["resolved_run_sha256"]),
            "protocol/model_count": len(artifact.get("campaign", {}).get("active_models", cfg.experiment.model_presets)),
            "protocol/seed_count": len(artifact.get("campaign", {}).get("seeds", cfg.experiment.seeds)),
        })
        tracker.log_table("protocol/model_run_contracts", [
            {"run": run, "resolved_run_sha256": digest, **artifact["model_contracts"][run]}
            for run, digest in sorted(artifact["resolved_run_sha256"].items())
        ])
        tracker.summarize({
            "protocol/sha256": protocol_sha,
            "protocol/run_count": len(artifact["resolved_run_sha256"]),
            "protocol/review_freeze_sha256": artifact["review_freeze_sha256"],
        })
        tracker.log_artifact_files(
            f"sac-qutab-experiment-protocol-{protocol_sha[:12]}", "experiment-protocol", [output_path],
            metadata={"protocol_sha256": protocol_sha, "config_sha256": artifact["config_sha256"]},
        )
        succeeded = True
    finally:
        tracker.finish(0 if succeeded else 1)


def publish_detector_to_wandb(
    cfg: Config,
    output_path: str | Path,
    artifact: Mapping[str, Any],
    *,
    stage: str,
    sdk: Any | None = None,
) -> None:
    if stage not in {"detector-fit", "detector-evaluation"}:
        raise ValueError("detector W&B stage must be detector-fit or detector-evaluation")
    if _wandb_mode(cfg.wandb) == "disabled":
        return
    output_path = Path(output_path)
    run_name = f"{artifact['model_name']}-seed{artifact['training_seed']}"
    artifact_sha = sha256_file(output_path)
    protocol_sha = artifact.get("protocol_sha256")
    tracker = init_wandb_stage_tracker(
        cfg, stage, output_path.parent,
        identity={"run": run_name, "artifact_sha256": artifact_sha, "protocol_sha256": protocol_sha},
        display_name=run_name,
        stage_config={
            "model": artifact["model_name"], "seed": artifact["training_seed"],
            "training_wandb_run_id": artifact.get("training_wandb_run_id"),
        },
        sdk=sdk,
    )
    if tracker is None:
        return
    succeeded = False
    try:
        scalars: dict[str, Any] = {}
        if stage == "detector-fit":
            population = artifact["fit_population"]
            scalars = {
                "detector/calibration/count": population["count"],
                "detector/calibration/positives": population["positives"],
                "detector/calibration/prevalence": population["prevalence"],
                "detector/learned_detectors_defined": int(bool(artifact["learned_detectors_defined"])),
            }
            tracker.log_table("detector/fitted_models", [
                {"detector": name, **values} for name, values in sorted(artifact["models"].items())
            ])
        else:
            population = artifact["population"]
            scalars = {
                "detector/final_test/count": population["count"],
                "detector/final_test/positives": population["positives"],
                "detector/final_test/prevalence": population["prevalence"],
            }
            score_rows = []
            for name, values in sorted(artifact["models"].items()):
                score_rows.append({"score": name, **values})
                for metric in ("average_precision", "auroc"):
                    value = values.get(metric)
                    if isinstance(value, (int, float)):
                        scalars[f"detector/final_test/{name}/{metric}"] = value
            tracker.log_table("detector/final_test_scores", score_rows)
            tracker.log_table("detector/condition_prevalence", [
                {"condition": condition, **values}
                for condition, values in sorted(artifact["condition_prevalence"].items())
            ])
        tracker.log(scalars)
        tracker.summarize({
            **scalars,
            "detector/artifact_sha256": artifact_sha,
            "lineage/training_wandb_run_id": artifact.get("training_wandb_run_id"),
            "lineage/protocol_sha256": protocol_sha,
        })
        tracker.log_artifact_files(
            f"sac-qutab-{run_name}-{stage}", stage, [output_path],
            metadata={
                "artifact_sha256": artifact_sha,
                "protocol_sha256": protocol_sha,
                "training_wandb_run_id": artifact.get("training_wandb_run_id"),
            },
        )
        succeeded = True
    finally:
        tracker.finish(0 if succeeded else 1)


def publish_evaluation_to_wandb(
    cfg: Config,
    output_dir: str | Path,
    provenance: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    sdk: Any | None = None,
) -> None:
    output_dir = Path(output_dir)
    role = str(provenance["split_role"])
    run_name = f"{provenance['model_name']}-seed{provenance['training_seed']}"
    identity = {
        "run": run_name,
        "role": role,
        "resolved_run_sha256": provenance["resolved_run_sha256"],
        "protocol_sha256": provenance.get("protocol_sha256"),
        "pair_manifest_sha256": provenance["pair_manifest_sha256"],
    }
    tracker = init_wandb_stage_tracker(
        cfg, f"evaluation-{role}", output_dir, identity=identity, display_name=run_name,
        stage_config={
            "model": provenance["model_name"], "seed": provenance["training_seed"],
            "training_wandb_run_id": provenance.get("training_wandb_run_id"),
        },
        sdk=sdk,
    )
    if tracker is None:
        return
    succeeded = False
    try:
        clean = metrics["clean"]
        calibration = metrics["clean_calibration"]
        scalar_values: dict[str, Any] = {
            f"evaluation/{role}/clean_accuracy": clean["accuracy"],
            f"evaluation/{role}/clean_macro_f1": clean["macro_f1"],
            f"evaluation/{role}/clean_ece": calibration["ece"]["ece"],
            f"evaluation/{role}/clean_brier": calibration["brier"],
            f"evaluation/{role}/clean_nll": calibration["nll"],
        }
        throughput = provenance.get("inference", {}).get("forward_images_per_second")
        if isinstance(throughput, (int, float)):
            scalar_values[f"evaluation/{role}/forward_images_per_second"] = throughput
        condition_rows = []
        for condition, values in sorted(metrics["conditions"].items()):
            family, severity = condition.split(":", 1)
            condition_rows.append({"family": family, "severity": severity, **values})
            prefix = f"evaluation/{role}/{family}/{severity}"
            for source_key, metric_name in (
                ("transformed_accuracy", "accuracy"),
                ("label_consistency", "s_label"),
                ("js_mean", "js_divergence"),
                ("cosine_mean_within_model", "cosine_similarity"),
                ("normalized_instability_mean", "normalized_instability"),
                ("failure_prevalence_among_clean_correct", "failure_prevalence"),
            ):
                value = values.get(source_key)
                if isinstance(value, (int, float)):
                    scalar_values[f"{prefix}/{metric_name}"] = value
        tracker.log(scalar_values)
        tracker.log_table(f"evaluation/{role}/condition_metrics", condition_rows)
        tracker.summarize({
            **scalar_values,
            "lineage/training_wandb_run_id": provenance.get("training_wandb_run_id"),
            "lineage/protocol_sha256": provenance.get("protocol_sha256"),
            "lineage/evaluation_index_sha256": sha256_file(output_dir / "evaluation_index.json"),
        })
        tracker.log_artifact_files(
            f"sac-qutab-{run_name}-{role}-evaluation", "evaluation-metrics",
            [output_dir / "provenance.json", output_dir / "metrics.json", output_dir / "evaluation_index.json"],
            metadata={
                **identity,
                "pair_metrics_sha256": sha256_file(output_dir / "pair_metrics.jsonl"),
                "representations_sha256": sha256_file(output_dir / "representations.npz"),
                "large_pair_and_representation_files_uploaded": False,
            },
            root=output_dir,
        )
        succeeded = True
    finally:
        tracker.finish(0 if succeeded else 1)


def publish_statistics_to_wandb(
    cfg: Config, output_path: str | Path, artifact: Mapping[str, Any], *, sdk: Any | None = None,
) -> None:
    output_path = Path(output_path)
    protocol_sha = str(artifact["protocol_sha256"])
    tracker = init_wandb_stage_tracker(
        cfg, "primary-statistics", output_path.parent,
        identity={"protocol_sha256": protocol_sha, "statistics_sha256": sha256_file(output_path)},
        display_name=_campaign_display_name(),
        stage_config={"cluster_unit": artifact["cluster_unit"], "families": artifact["families"]},
        sdk=sdk,
    )
    if tracker is None:
        return
    succeeded = False
    try:
        table_specs = {
            "statistics/primary_per_seed": artifact["primary_per_seed"],
            "statistics/primary_seed_summaries": artifact["primary_seed_summaries"],
            "statistics/pairwise_model_summaries": artifact["pairwise_model_summaries"],
            "statistics/rq3_absolute_clustered": artifact["rq3_absolute_clustered"],
            "statistics/exploratory_bh": artifact["exploratory_bh"],
        }
        for key, rows in table_specs.items():
            tracker.log_table(key, rows)
        scalars: dict[str, Any] = {}
        for row in artifact["pairwise_model_summaries"]:
            pair = f"{row['right_model']}_minus_{row['left_model']}"
            prefix = f"statistics/pairwise/{pair}"
            for field in ("effect", "p_value", "holm_adjusted_p"):
                if isinstance(row.get(field), (int, float)):
                    scalars[f"{prefix}/{field}"] = row[field]
            scalars[f"{prefix}/holm_rejected"] = int(bool(row.get("holm_rejected")))
        tracker.log(scalars)
        tracker.summarize({
            "statistics/protocol_sha256": protocol_sha,
            "statistics/artifact_sha256": sha256_file(output_path),
            "statistics/pairwise_count": len(artifact["pairwise_model_summaries"]),
            "statistics/holm_rejection_count": sum(bool(row.get("holm_rejected")) for row in artifact["pairwise_model_summaries"]),
        })
        tracker.log_artifact_files(
            f"sac-qutab-primary-statistics-{protocol_sha[:12]}", "statistics", [output_path],
            metadata={"protocol_sha256": protocol_sha, "statistics_sha256": sha256_file(output_path)},
        )
        succeeded = True
    finally:
        tracker.finish(0 if succeeded else 1)


def publish_report_to_wandb(
    cfg: Config,
    output_dir: str | Path,
    index: Mapping[str, Any],
    *,
    sdk: Any | None = None,
) -> None:
    output_dir = Path(output_dir)
    protocol_sha = str(index["protocol_sha256"])
    tracker = init_wandb_stage_tracker(
        cfg, "report-artifacts", output_dir,
        identity={"protocol_sha256": protocol_sha, "report_schema": index["schema_version"]},
        display_name=_campaign_display_name(),
        stage_config={"generated_files": index["generated"], "report_schema": index["schema_version"]},
        sdk=sdk,
    )
    if tracker is None:
        return
    succeeded = False
    try:
        png_paths = sorted(output_dir.glob("*.png"))
        csv_paths = sorted(output_dir.glob("*.csv"))
        for path in png_paths:
            tracker.log_image(f"report/plots/{path.stem}", path, caption=path.stem.replace("_", " "))
        for path in csv_paths:
            with path.open(newline="", encoding="utf-8") as handle:
                tracker.log_table(f"report/tables/{path.stem}", list(csv.DictReader(handle)))
        tracker.log({
            "report/png_count": len(png_paths),
            "report/table_count": len(csv_paths),
            "report/generated_file_count": len(index["generated"]),
        })
        tracker.summarize({
            "report/protocol_sha256": protocol_sha,
            "report/index_sha256": sha256_file(output_dir / "index.json"),
            "report/png_count": len(png_paths),
            "report/table_count": len(csv_paths),
        })
        files = [
            path for path in output_dir.iterdir()
            if path.is_file() and not path.name.startswith("wandb_")
        ]
        tracker.log_artifact_files(
            f"sac-qutab-publication-report-{protocol_sha[:12]}", "publication-report", files,
            metadata={
                "protocol_sha256": protocol_sha,
                "index_sha256": sha256_file(output_dir / "index.json"),
                "png_count": len(png_paths),
                "table_count": len(csv_paths),
            },
            root=output_dir,
        )
        succeeded = True
    finally:
        tracker.finish(0 if succeeded else 1)
