from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import Config, ResolvedRun, as_dict, config_digest, materialize_run
from .data import validate_split_contract
from .runtime import atomic_write_json, runtime_path, sha256_file
from .tracking import publish_hardware_preflight_to_wandb

_PREFLIGHT_SCHEMA = "sac-hardware-preflight-v1"
_CAMPAIGN_PREFLIGHT_SCHEMA = "sac-hardware-preflight-v2"


def _validate_campaign_device(summary: Mapping[str, Any], model: str) -> None:
    device_name = str(summary.get("device", {}).get("name", ""))
    total_memory = int(summary.get("device", {}).get("total_memory_bytes", 0))
    if "A100" not in device_name.upper() or not 18 * 1024**3 <= total_memory <= 22 * 1024**3:
        raise ValueError(f"campaign timing for {model} must come from the expected A100 20 GB MIG device")
    if summary.get("precision") != "bf16":
        raise ValueError(f"campaign timing for {model} must prove BF16 execution")


def _validate_cuda_measurement(summary: Mapping[str, Any]) -> None:
    device = summary.get("device")
    if not isinstance(device, dict) or device.get("type") != "cuda":
        raise ValueError("CPU/local summaries are ineligible for the CUDA hardware preflight")
    name, capability = device.get("name"), device.get("capability")
    total_memory, multiprocessors = device.get("total_memory_bytes"), device.get("multi_processor_count")
    cuda_ordinal = device.get("cuda_ordinal")
    if (not isinstance(name, str) or not name.strip() or not isinstance(capability, list)
            or len(capability) != 2 or not all(isinstance(item, int) for item in capability)
            or not device.get("cuda_runtime")):
        raise ValueError("compute timing must identify a visible CUDA GPU and CUDA runtime")
    if (not isinstance(total_memory, int) or total_memory <= 0
            or not isinstance(multiprocessors, int) or multiprocessors <= 0
            or not isinstance(cuda_ordinal, int) or cuda_ordinal < 0):
        raise ValueError("CUDA timing has incomplete device capability or memory metadata")
    if int(summary.get("peak_gpu_reserved_bytes", 0)) <= 0:
        raise ValueError("eligible CUDA timing must record positive reserved memory")


def _validate_rank_memory(
    summary: Mapping[str, Any], world_size: int, model: str, *, max_memory_fraction: float = 1.0,
) -> None:
    rank_memory = summary.get("peak_gpu_memory_by_rank")
    if not isinstance(rank_memory, list) or len(rank_memory) != world_size:
        raise ValueError(f"timing summary for {model} omits per-rank memory measurements")
    try:
        by_rank = {int(row["rank"]): int(row["reserved_bytes"]) for row in rank_memory}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"timing summary for {model} has invalid per-rank memory measurements") from exc
    if set(by_rank) != set(range(world_size)) or any(value <= 0 for value in by_rank.values()):
        raise ValueError(f"timing summary for {model} has invalid or duplicate rank memory measurements")
    if int(summary.get("peak_gpu_reserved_bytes", 0)) != max(by_rank.values()):
        raise ValueError(f"timing summary for {model} misstates maximum per-rank reserved memory")
    total_memory = int(summary["device"]["total_memory_bytes"])
    if any(value >= total_memory * max_memory_fraction for value in by_rank.values()):
        raise ValueError(f"timing summary for {model} exceeds the allowed CUDA memory fraction and memory headroom gate")


def _validate_full_split_projection(
    summary: Mapping[str, Any], run: ResolvedRun, expected_train_count: int, expected_validation_count: int,
) -> None:
    measured_samples = int(summary.get("measured_sample_count", 0))
    measured_steps = int(summary.get("measured_optimizer_steps_rank0", 0))
    warmup_steps = int(summary.get("timing_warmup_steps_rank0", 0))
    timed_steps = int(summary.get("timed_optimizer_steps_rank0", 0))
    full_samples = int(summary.get("projected_full_train_sample_count", 0))
    projected_steps = int(summary.get("projected_optimizer_steps_per_epoch", 0))
    median_step = summary.get("median_optimizer_step_seconds_rank0")
    expected_steps = expected_train_count // run.effective_global_batch
    if (measured_samples <= 0 or full_samples != expected_train_count or measured_samples > full_samples
            or expected_steps <= 0 or projected_steps != expected_steps):
        raise ValueError(f"timing summary for {run.model.name} does not project from the full frozen train split")
    if warmup_steps < 5 or timed_steps < 20 or measured_steps != warmup_steps + timed_steps:
        raise ValueError(f"timing summary for {run.model.name} lacks five warm-up and twenty post-warm optimizer steps")
    expected_hours = float(median_step) * expected_steps * run.config.train.epochs / 3600 if isinstance(median_step, (int, float)) else None
    projected_hours = summary.get("projected_epoch_cap_hours_from_median_rank0")
    if expected_hours is None or not isinstance(projected_hours, (int, float)) or abs(float(projected_hours) - expected_hours) > max(1e-9, expected_hours * 1e-9):
        raise ValueError(f"timing summary for {run.model.name} has inconsistent full-run projection arithmetic")
    measured_validation_samples = int(summary.get("measured_validation_sample_count", 0))
    full_validation_samples = int(summary.get("projected_full_validation_sample_count", 0))
    measured_validation_seconds = summary.get("measured_validation_seconds_rank0")
    projected_validation_seconds = summary.get("projected_validation_seconds_per_epoch_rank0")
    projected_combined = summary.get("projected_training_validation_hours")
    if (measured_validation_samples <= 0 or measured_validation_samples > expected_validation_count
            or full_validation_samples != expected_validation_count
            or not isinstance(measured_validation_seconds, (int, float)) or float(measured_validation_seconds) <= 0):
        raise ValueError(f"timing summary for {run.model.name} lacks bounded validation timing")
    expected_validation_seconds = float(measured_validation_seconds) * expected_validation_count / measured_validation_samples
    expected_combined = expected_hours + expected_validation_seconds * run.config.train.epochs / 3600
    if (not isinstance(projected_validation_seconds, (int, float))
            or abs(float(projected_validation_seconds) - expected_validation_seconds) > max(1e-9, expected_validation_seconds * 1e-9)
            or not isinstance(projected_combined, (int, float))
            or abs(float(projected_combined) - expected_combined) > max(1e-9, expected_combined * 1e-9)):
        raise ValueError(f"timing summary for {run.model.name} has inconsistent validation projection arithmetic")


def _read_weight_contract(path: Path, model: str, expected_digest: str, summary_digest: Any) -> dict[str, str]:
    weight = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_path = runtime_path(weight.get("checkpoint_path", ""))
    digest = weight.get("sha256")
    if (weight.get("model") != model or digest != expected_digest
            or not checkpoint_path.is_file() or sha256_file(checkpoint_path) != digest or summary_digest != digest):
        raise ValueError(f"weight preflight or timing initialization for {model} does not match the exact local checkpoint")
    return {"preflight_path": str(path), "preflight_sha256": sha256_file(path),
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": digest}


def create_hardware_preflight(
    cfg: Config, split_path: str | Path, summary_paths: Mapping[str, str | Path],
    weight_preflight_paths: Mapping[str, str | Path], output: str | Path, *,
    model_names: tuple[str, ...] | None = None, seeds: tuple[int, ...] | None = None,
    batch_caps: Mapping[str, int] | None = None, campaign_name: str | None = None,
    campaign_sha256: str | None = None, max_memory_fraction: float = 1.0,
    max_projected_hours_per_seed: float | None = None,
) -> dict[str, Any]:
    manifest_path = cfg.output_path(cfg.data.manifest_relpath)
    _, split = validate_split_contract(cfg, manifest_path, split_path)
    train_count, validation_count = int(split["counts"]["train"]), int(split["counts"]["validation"])
    selected_models = model_names or tuple(preset.name for preset in cfg.experiment.model_presets)
    selected_seeds = seeds or tuple(cfg.experiment.seeds)
    preset_by_name = {preset.name: preset for preset in cfg.experiment.model_presets}
    if any(model not in preset_by_name for model in selected_models) or any(seed not in cfg.experiment.seeds for seed in selected_seeds):
        raise ValueError("hardware preflight selection is outside the frozen base Config")
    expected_models = set(selected_models)
    if set(summary_paths) != expected_models or set(weight_preflight_paths) != expected_models:
        raise ValueError("hardware preflight requires one CUDA timing summary and weight preflight for each core model")
    summaries: dict[str, Any] = {}
    weights: dict[str, Any] = {}
    run_hashes: dict[str, str] = {}
    run_contracts: dict[str, Any] = {}
    total_hours, campaign_world_size = 0.0, None
    manifest_sha256, split_sha256 = sha256_file(manifest_path), sha256_file(split_path)
    for model in selected_models:
        preset = preset_by_name[model]
        summary_path = runtime_path(summary_paths[model])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        required = {"status", "model_name", "training_seed", "resolved_run_sha256", "manifest_sha256", "split_sha256",
                    "pretrained_checkpoint_sha256", "device", "world_size", "batch_size_per_device", "grad_accum_steps",
                    "effective_global_batch", "peak_gpu_reserved_bytes", "peak_gpu_memory_by_rank",
                    "projected_epoch_cap_hours_from_median_rank0", "measured_optimizer_steps_rank0", "measured_sample_count",
                    "projected_full_train_sample_count", "projected_optimizer_steps_per_epoch", "median_optimizer_step_seconds_rank0",
                    "timing_warmup_steps_rank0", "timed_optimizer_steps_rank0", "measured_validation_sample_count",
                    "projected_full_validation_sample_count", "measured_validation_seconds_rank0",
                    "projected_validation_seconds_per_epoch_rank0", "projected_training_validation_hours", "precision"}
        if not required <= set(summary) or summary["status"] != "dry_run_complete" or summary["model_name"] != model:
            raise ValueError(f"invalid bounded timing summary for {model}")
        if summary["manifest_sha256"] != manifest_sha256 or summary["split_sha256"] != split_sha256:
            raise ValueError(f"timing summary for {model} is not linked to the current manifest and split")
        if campaign_name:
            _validate_campaign_device(summary, model)
        world_size = int(summary["world_size"])
        if world_size <= 0 or (campaign_world_size is not None and campaign_world_size != world_size):
            raise ValueError("all model timing summaries must use the same positive world size")
        campaign_world_size = world_size
        timing_seed = int(summary["training_seed"])
        expected_batch = (batch_caps or {}).get(model, int(summary["batch_size_per_device"]))
        measured_run = materialize_run(cfg, model, timing_seed, world_size, expected_batch)
        if (summary["resolved_run_sha256"] != config_digest(measured_run)
                or int(summary["grad_accum_steps"]) != measured_run.grad_accum_steps
                or int(summary["effective_global_batch"]) != measured_run.effective_global_batch):
            raise ValueError(f"timing summary for {model} does not match its exact resolved run")
        _validate_cuda_measurement(summary)
        _validate_full_split_projection(summary, measured_run, train_count, validation_count)
        if int(summary["batch_size_per_device"]) != measured_run.batch_size_per_device:
            raise ValueError(f"timing summary for {model} does not use its selected physical batch")
        _validate_rank_memory(summary, world_size, model, max_memory_fraction=max_memory_fraction)
        if (max_projected_hours_per_seed is not None
                and float(summary["projected_training_validation_hours"]) > max_projected_hours_per_seed):
            raise ValueError(f"timing projection for {model} exceeds the campaign per-seed runtime gate")
        weights[model] = _read_weight_contract(runtime_path(weight_preflight_paths[model]), model, preset.checkpoint_sha256,
                                               summary.get("pretrained_checkpoint_sha256"))
        summaries[model] = {"path": str(summary_path), "sha256": sha256_file(summary_path), "measurements": summary}
        total_hours += float(summary["projected_training_validation_hours"]) * len(selected_seeds)
        for seed in selected_seeds:
            key = f"{model}-seed{seed}"
            resolved = materialize_run(cfg, model, seed, world_size, measured_run.batch_size_per_device)
            run_hashes[key], run_contracts[key] = config_digest(resolved), as_dict(resolved)
    artifact = {"schema_version": _CAMPAIGN_PREFLIGHT_SCHEMA if campaign_name else _PREFLIGHT_SCHEMA, "status": "ready_for_training",
                "config_sha256": config_digest(cfg), "manifest_sha256": manifest_sha256, "split_sha256": split_sha256,
                "world_size": campaign_world_size, "summaries": summaries, "weight_preflights": weights,
                "resolved_run_sha256": run_hashes, "resolved_run_contracts": run_contracts,
                "projected_training_and_validation_hours": total_hours,
                "note": "Measured on this selected GPU; not a provider SLA or a fit guarantee for another GPU."}
    if campaign_name:
        artifact["campaign"] = {
            "name": campaign_name, "sha256": campaign_sha256, "active_models": list(selected_models),
            "seeds": list(selected_seeds), "max_device_batch": dict(batch_caps or {}),
            "max_memory_fraction": max_memory_fraction,
            "max_projected_hours_per_seed": max_projected_hours_per_seed,
        }
    atomic_write_json(output, artifact)
    artifact["preflight_sha256"] = sha256_file(output)
    publish_hardware_preflight_to_wandb(cfg, output, artifact)
    return artifact


def validate_hardware_preflight(
    cfg: Config, split_path: str | Path, preflight_path: str | Path, *, world_size: int,
    max_device_batch: int | None = None, expected_runs: Mapping[str, ResolvedRun] | None = None,
    campaign_sha256: str | None = None, max_memory_fraction: float | None = None,
    max_projected_hours_per_seed: float | None = None,
) -> dict[str, Any]:
    manifest_path = cfg.output_path(cfg.data.manifest_relpath)
    _, split = validate_split_contract(cfg, manifest_path, split_path)
    artifact = json.loads(runtime_path(preflight_path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") not in {_PREFLIGHT_SCHEMA, _CAMPAIGN_PREFLIGHT_SCHEMA} or artifact.get("status") != "ready_for_training":
        raise ValueError("unsupported or incomplete hardware preflight")
    if campaign_sha256 is not None and artifact.get("campaign", {}).get("sha256") != campaign_sha256:
        raise ValueError("hardware preflight mismatch: campaign_sha256")
    expected = {"config_sha256": config_digest(cfg), "manifest_sha256": sha256_file(manifest_path),
                "split_sha256": sha256_file(split_path), "world_size": world_size}
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"hardware preflight mismatch: {key}")
    if expected_runs is None:
        expected_runs = {f"{p.name}-seed{s}": materialize_run(cfg, p.name, s, world_size, max_device_batch)
                         for p in cfg.experiment.model_presets for s in cfg.experiment.seeds}
    expected_keys = set(expected_runs)
    if not expected_keys or any(item.world_size != world_size for item in expected_runs.values()):
        raise ValueError("hardware preflight expected-run matrix is incomplete or has a different world size")
    run_hashes = {key: config_digest(item) for key, item in expected_runs.items()}
    artifact_run_hashes = artifact.get("resolved_run_sha256", {})
    artifact_run_contracts = artifact.get("resolved_run_contracts", {})
    legacy_subset = artifact.get("schema_version") == _PREFLIGHT_SCHEMA and set(artifact_run_hashes) != expected_keys
    compared_hashes = {key: artifact_run_hashes.get(key) for key in expected_keys} if legacy_subset else artifact_run_hashes
    compared_contracts = {key: artifact_run_contracts.get(key) for key in expected_keys} if legacy_subset else artifact_run_contracts
    if compared_hashes != run_hashes:
        raise ValueError("hardware preflight mismatch: resolved_run_sha256")
    if compared_contracts != {key: as_dict(item) for key, item in expected_runs.items()}:
        raise ValueError("hardware preflight mismatch: resolved_run_contracts")
    projected = 0.0
    train_count, validation_count = int(split["counts"]["train"]), int(split["counts"]["validation"])
    selected_models: dict[str, Any] = {}
    for run in expected_runs.values():
        selected_models[run.model.name] = run.model
    selected_seed_count = {model: len({run.seed for run in expected_runs.values() if run.model.name == model}) for model in selected_models}
    memory_gate = max_memory_fraction if max_memory_fraction is not None else float(artifact.get("campaign", {}).get("max_memory_fraction", 1.0))
    runtime_gate = max_projected_hours_per_seed
    if runtime_gate is None:
        runtime_gate = artifact.get("campaign", {}).get("max_projected_hours_per_seed")
    for model, preset in selected_models.items():
        summary_entry = artifact.get("summaries", {}).get(model)
        weight_entry = artifact.get("weight_preflights", {}).get(model)
        if not isinstance(summary_entry, dict) or not isinstance(weight_entry, dict):
            raise ValueError(f"hardware preflight omits {model}")
        summary_path = runtime_path(summary_entry["path"])
        weight_path = runtime_path(weight_entry["preflight_path"])
        checkpoint_path = runtime_path(weight_entry["checkpoint_path"])
        if (not summary_path.is_file() or sha256_file(summary_path) != summary_entry["sha256"]
                or not weight_path.is_file() or sha256_file(weight_path) != weight_entry["preflight_sha256"]
                or not checkpoint_path.is_file() or sha256_file(checkpoint_path) != weight_entry["checkpoint_sha256"]):
            raise ValueError(f"hardware preflight source changed for {model}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = f"{model}-seed{int(summary.get('training_seed', -1))}"
        timing_run = expected_runs.get(key)
        if timing_run is None or summary.get("resolved_run_sha256") != run_hashes[key]:
            raise ValueError(f"hardware preflight resolved run invalid for {model}")
        _validate_cuda_measurement(summary)
        if campaign_sha256 is not None:
            _validate_campaign_device(summary, model)
        _validate_full_split_projection(summary, timing_run, train_count, validation_count)
        _validate_rank_memory(summary, world_size, model, max_memory_fraction=memory_gate)
        if runtime_gate is not None and float(summary["projected_training_validation_hours"]) > float(runtime_gate):
            raise ValueError(f"hardware preflight runtime gate exceeded for {model}")
        weight = json.loads(weight_path.read_text(encoding="utf-8"))
        if (weight.get("checkpoint_path") != weight_entry["checkpoint_path"]
                or weight.get("sha256") != weight_entry["checkpoint_sha256"]
                or summary.get("pretrained_checkpoint_sha256") != weight_entry["checkpoint_sha256"]):
            raise ValueError(f"hardware preflight weight provenance invalid for {model}")
        projected += float(summary["projected_training_validation_hours"]) * selected_seed_count[model]
    artifact_projection = float(artifact.get("projected_training_and_validation_hours", -1))
    if legacy_subset:
        if artifact_projection < projected:
            raise ValueError("hardware preflight subset projection exceeds the frozen campaign projection")
    elif abs(artifact_projection - projected) > max(1e-9, projected * 1e-9):
        raise ValueError("hardware preflight projection changed")
    return artifact
