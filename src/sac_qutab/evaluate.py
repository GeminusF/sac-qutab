from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .campaign import Campaign, campaign_digest
from .config import REPORT_ARTIFACT_PLAN, Config, ResolvedRun, config_digest
from .data import load_manifest, load_native_image, preprocess_image, validate_split_contract
from .governance import validate_hardware_preflight
from .interventions import load_cached_tensor, pair_condition_signature, validate_pair_manifest_contract
from .metrics import (
    apply_control_ecdf,
    brier_score,
    classification_metrics,
    counterfactual_pair_metrics,
    expected_calibration_error,
    fit_control_ecdf,
    linear_cka,
    negative_log_likelihood,
    summarize_confidence_change,
)
from .models import ModelOutput, build_model
from .review import apply_review_freeze, validate_review_freeze_contract
from .runtime import atomic_write_json, read_jsonl, sha256_file, write_jsonl
from .tracking import checkpoint_wandb_run_id, publish_evaluation_to_wandb, publish_protocol_to_wandb


def create_protocol_freeze(
    cfg: Config, split_path: str | Path, pair_manifests: dict[str, str], review_freeze_path: str | Path,
    checkpoint_paths: dict[str, str], detector_paths: dict[str, str], control_ecdf_paths: dict[str, str],
    preflight_path: str | Path | None, output: str | Path, *,
    max_device_batch: int | None = None, resolved_runs: Mapping[str, ResolvedRun] | None = None,
    campaign: Campaign | None = None,
    training_preflight_paths: Mapping[str, str | Path] | None = None,
    inference_preflight_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = cfg.output_path(cfg.data.manifest_relpath)
    validate_split_contract(cfg, manifest_path, split_path)
    manifest_sha256 = sha256_file(manifest_path)
    split_sha256 = sha256_file(split_path)
    if campaign is None:
        if preflight_path is None:
            raise ValueError("legacy experiment freeze requires a hardware preflight")
        preflight_raw = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
        preflight = validate_hardware_preflight(
            cfg, split_path, preflight_path,
            world_size=int(preflight_raw.get("world_size", 0)), max_device_batch=max_device_batch, expected_runs=resolved_runs,
        )
        selected_models = tuple(preset.name for preset in cfg.experiment.model_presets)
        selected_seeds = tuple(cfg.experiment.seeds)
        selected_pairs = tuple(cfg.statistics.model_pair_contrasts)
        preflights_by_model = {model: preflight for model in selected_models}
    else:
        selected_models, selected_seeds = campaign.active_models, campaign.seeds
        selected_pairs = campaign.model_pair_contrasts
        if resolved_runs is None:
            raise ValueError("campaign experiment freeze requires its resolved-run matrix")
        if not training_preflight_paths or set(training_preflight_paths) != set(selected_models):
            raise ValueError("campaign freeze requires one --training-preflight mapping per active model")
        if inference_preflight_path is None:
            raise ValueError("campaign freeze requires the common A100 --inference-preflight")
        campaign_sha = campaign_digest(campaign)
        preflights_by_model: dict[str, dict[str, Any]] = {}
        for model in selected_models:
            path = Path(training_preflight_paths[model])
            raw = json.loads(path.read_text(encoding="utf-8"))
            model_runs = {key: run for key, run in resolved_runs.items() if run.model.name == model}
            if raw.get("schema_version") == "sac-hardware-preflight-v1":
                validated = validate_hardware_preflight(
                    cfg, split_path, path, world_size=int(raw.get("world_size", 0)), expected_runs=model_runs,
                )
            else:
                validation_runs = resolved_runs if set(raw.get("resolved_run_sha256", {})) == set(resolved_runs) else model_runs
                validated = validate_hardware_preflight(
                    cfg, split_path, path, world_size=int(raw.get("world_size", 0)), expected_runs=validation_runs,
                    campaign_sha256=campaign_sha, max_memory_fraction=campaign.max_memory_fraction,
                    max_projected_hours_per_seed=campaign.max_projected_hours_per_seed,
                )
            if model not in validated.get("summaries", {}) or any(key not in validated.get("resolved_run_sha256", {}) for key in model_runs):
                raise ValueError(f"training preflight does not cover campaign model {model}")
            preflights_by_model[model] = validated
        inference_raw = json.loads(Path(inference_preflight_path).read_text(encoding="utf-8"))
        inference_preflight = validate_hardware_preflight(
            cfg, split_path, inference_preflight_path, world_size=int(inference_raw.get("world_size", 0)),
            expected_runs=resolved_runs, campaign_sha256=campaign_sha,
            max_memory_fraction=campaign.max_memory_fraction,
            max_projected_hours_per_seed=campaign.max_projected_hours_per_seed,
        )
        if any("A100" not in str(item["measurements"]["device"]["name"]).upper() for item in inference_preflight["summaries"].values()):
            raise ValueError("campaign final inference preflight must identify an A100 device")
    if set(pair_manifests) != {"validation", "calibration", "final_test"}:
        raise ValueError("experiment freeze requires all three non-training pair manifests")
    pair_contracts: dict[str, dict[str, Any]] = {}
    for role, path in pair_manifests.items():
        contract = validate_pair_manifest_contract(cfg, path, require_cache=True)
        if contract["role"] != role:
            raise ValueError(f"pair manifest role mismatch: {role}")
        pair_contracts[role] = contract
    review = json.loads(Path(review_freeze_path).read_text(encoding="utf-8"))
    validate_review_freeze_contract(cfg, review, validation_pair_manifest=pair_manifests["validation"])
    review_sha256 = sha256_file(review_freeze_path)
    expected_keys = {f"{model}-seed{seed}" for model in selected_models for seed in selected_seeds}
    if set(checkpoint_paths) != expected_keys or set(detector_paths) != expected_keys or set(control_ecdf_paths) != expected_keys:
        raise ValueError(f"experiment freeze requires all {len(expected_keys)} configured checkpoints, detectors, and validation ECDFs")
    if resolved_runs is not None and set(resolved_runs) != expected_keys:
        raise ValueError("experiment freeze received an incomplete resolved-run matrix")
    detector_fit_contract = {
        "solver": cfg.detector.solver, "penalty": cfg.detector.penalty, "regularization_c": cfg.detector.regularization_c,
        "class_weight": cfg.detector.class_weight, "tolerance": cfg.detector.tolerance, "max_iter": cfg.detector.max_iter,
        "min_positives": cfg.evaluation.rq3_min_positives_per_model_seed,
    }
    feature_contract = {
        "output_only": ["transformed_uncertainty", "transformed_entropy"],
        "output_plus_representation": ["transformed_uncertainty", "transformed_entropy", "normalized_representation_instability"],
    }
    resolved_sha256: dict[str, str] = {}
    model_contracts: dict[str, Any] = {}
    preset_by_name = {preset.name: preset for preset in cfg.experiment.model_presets}
    for model in selected_models:
        preset = preset_by_name[model]
        model_preflight = preflights_by_model[model]
        expected_weight_sha256 = model_preflight["weight_preflights"][model]["checkpoint_sha256"]
        for seed in selected_seeds:
            run_key = f"{preset.name}-seed{seed}"
            frozen_run_sha256 = model_preflight["resolved_run_sha256"][run_key]
            if resolved_runs is not None and config_digest(resolved_runs[run_key]) != frozen_run_sha256:
                raise ValueError(f"orchestrated run does not match compute preflight: {run_key}")
            checkpoint_path = Path(checkpoint_paths[run_key])
            checkpoint_sha256 = sha256_file(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (checkpoint.get("schema_version") != "sac-checkpoint-v2"
                    or checkpoint.get("resolved_run_sha256") != frozen_run_sha256
                    or checkpoint.get("dataset_hashes") != {"manifest": manifest_sha256, "splits": split_sha256}
                    or checkpoint.get("pretrained_checkpoint_sha256") != expected_weight_sha256):
                raise ValueError(f"selected checkpoint does not match frozen run/data/pretrained weights: {run_key}")
            detector_path = Path(detector_paths[run_key])
            ecdf_path = Path(control_ecdf_paths[run_key])
            detector = json.loads(detector_path.read_text(encoding="utf-8"))
            ecdf = json.loads(ecdf_path.read_text(encoding="utf-8"))
            ecdf_sha256 = sha256_file(ecdf_path)
            expected_common = {
                "model_name": preset.name, "training_seed": seed, "config_sha256": config_digest(cfg),
                "condition_signature": pair_condition_signature(cfg), "checkpoint_sha256": checkpoint_sha256,
                "resolved_run_sha256": frozen_run_sha256, "review_freeze_sha256": review_sha256,
            }
            if (detector.get("schema_version") != "sac-detector-v3" or detector.get("fit_split") != "calibration"
                    or any(detector.get(key) != value for key, value in expected_common.items())
                    or detector.get("calibration_pair_manifest_sha256") != sha256_file(pair_manifests["calibration"])
                    or detector.get("control_ecdf_sha256") != ecdf_sha256
                    or detector.get("fit_configuration") != detector_fit_contract
                    or detector.get("feature_contract") != feature_contract):
                raise ValueError(f"detector provenance mismatch for {run_key}")
            if (ecdf.get("schema_version") != "sac-control-ecdf-v3" or ecdf.get("role") != "validation"
                    or any(ecdf.get(key) != value for key, value in expected_common.items())
                    or ecdf.get("pair_manifest_sha256") != sha256_file(pair_manifests["validation"])):
                raise ValueError(f"validation ECDF provenance mismatch for {run_key}")
            resolved_sha256[run_key] = frozen_run_sha256
            model_contracts[run_key] = {**model_preflight["resolved_run_contracts"][run_key], "pretrained_checkpoint_sha256": expected_weight_sha256}
    artifact = {
        "schema_version": "sac-experiment-protocol-v5" if campaign else "sac-experiment-protocol-v4",
        "config_sha256": config_digest(cfg),
        "manifest_sha256": manifest_sha256,
        "resolved_run_sha256": resolved_sha256,
        "split_sha256": split_sha256,
        "pair_manifest_sha256": {role: sha256_file(path) for role, path in sorted(pair_manifests.items())},
        "condition_signature": pair_condition_signature(cfg),
        "review_freeze_sha256": review_sha256,
        "condition_selection_policy": {
            "schema_version": review["schema_version"],
            "policy": review.get("policy", "human_semantic_review"),
            "human_semantic_review_performed": review["schema_version"] == "sac-review-freeze-v2",
            "limitation": review.get("limitation"),
        },
        "checkpoint_sha256": {name: sha256_file(path) for name, path in sorted(checkpoint_paths.items())},
        "detector_sha256": {name: sha256_file(path) for name, path in sorted(detector_paths.items())},
        "control_ecdf_sha256": {name: sha256_file(path) for name, path in sorted(control_ecdf_paths.items())},
        "hardware_preflight_sha256": sha256_file(preflight_path) if preflight_path is not None else None,
        "model_contracts": model_contracts,
        "metric_contract": {"ece_bins": cfg.evaluation.ece_bins, "rq3_population": "selected non-sham pairs whose clean prediction is correct, per model seed", "rq3_min_positives_calibration_and_test": cfg.evaluation.rq3_min_positives_per_model_seed, "pr_summary": cfg.evaluation.pr_summary, "normalization": cfg.evaluation.representation_normalization},
        "statistics_contract": {
            "primary_contrasts": list(cfg.statistics.primary_contrasts),
            "model_pair_contrasts": [
                {"left": pair.left, "right": pair.right} for pair in selected_pairs
            ],
            "cluster_unit": "source_group_id", "bootstrap_samples": cfg.statistics.bootstrap_samples,
            "confidence_level": cfg.statistics.confidence_level,
            "primary_familywise_alpha": cfg.statistics.primary_familywise_alpha,
            "seed_summary": "mean_and_sample_sd", "exploratory_fdr_q": cfg.statistics.fdr_q,
        },
        "report_plan": list(REPORT_ARTIFACT_PLAN),
        "test_access_rule": f"all {len(expected_keys)} configured inputs frozen before any final inference; score once; exact same-freeze rerun only for a recorded software failure",
    }
    if campaign is not None:
        artifact["campaign"] = {
            "name": campaign.name, "sha256": campaign_digest(campaign),
            "active_models": list(selected_models), "seeds": list(selected_seeds),
            "model_pair_contrasts": [{"left": pair.left, "right": pair.right} for pair in selected_pairs],
            "base_config_sha256": campaign.base_config_sha256,
        }
        artifact["training_hardware_preflight_sha256"] = {
            model: sha256_file(training_preflight_paths[model]) for model in selected_models
        }
        artifact["inference_hardware_preflight_sha256"] = sha256_file(inference_preflight_path)
        artifact["hardware_limitation"] = (
            "Training hardware is confounded with model (legacy ResNet-50 on H100; ViT-Small/16 on A100 MIG). "
            "Training throughput is not interpreted as an architecture comparison; all final inference uses the frozen A100 preflight."
        )
    atomic_write_json(output, artifact)
    artifact["protocol_sha256"] = sha256_file(output)
    publish_protocol_to_wandb(cfg, output, artifact)
    return artifact


def _load_checkpoint_model(
    run: ResolvedRun, checkpoint_path: str | Path, device: torch.device,
) -> tuple[nn.Module, str | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest_path = run.config.output_path(run.config.data.manifest_relpath)
    split_path = run.config.output_path(run.config.data.split_relpath)
    validate_split_contract(run.config, manifest_path, split_path)
    if (checkpoint.get("schema_version") != "sac-checkpoint-v2"
            or checkpoint.get("resolved_run_sha256") != config_digest(run)
            or checkpoint.get("dataset_hashes") != {"manifest": sha256_file(manifest_path), "splits": sha256_file(split_path)}):
        raise ValueError("checkpoint does not match the resolved run and current frozen data")
    model = build_model(run.model, len(run.config.data.class_names))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint_wandb_run_id(checkpoint)


def _infer_stream(
    model: nn.Module, items: Sequence[Any], load_item: Callable[[Any], torch.Tensor],
    device: torch.device, batch_size: int, *, forward_timings: list[tuple[int, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not items:
        raise ValueError("inference stream cannot be empty")
    probabilities: list[np.ndarray] = []
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = torch.stack([load_item(item) for item in items[start : start + batch_size]]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            output: ModelOutput = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if forward_timings is not None:
                forward_timings.append((len(batch), time.perf_counter() - forward_started))
            probabilities.append(torch.softmax(output.logits.float(), dim=1).cpu().numpy())
            features.append(output.representation.float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(features)


def _evaluation_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cpu, or cuda")


def evaluate_pairs(run: ResolvedRun, checkpoint_path: str | Path, pair_manifest: str | Path, review_freeze_path: str | Path, output_dir: str | Path, *, control_ecdf_path: str | Path | None = None, protocol_path: str | Path | None = None, software_rerun_reason: str | None = None) -> dict[str, Any]:
    contract = validate_pair_manifest_contract(run.config, pair_manifest, require_cache=True)
    all_rows = read_jsonl(pair_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    role = contract["role"]
    review_freeze = json.loads(Path(review_freeze_path).read_text(encoding="utf-8"))
    validate_review_freeze_contract(run.config, review_freeze)
    included_conditions = set(review_freeze["included_conditions"])
    excluded_condition_counts: dict[str, int] = defaultdict(int)
    for candidate in all_rows:
        if candidate["severity"] != "sham" and candidate["audit_condition"] not in included_conditions:
            excluded_condition_counts[candidate["audit_condition"]] += 1
    rows = apply_review_freeze(all_rows, review_freeze)
    if not rows:
        raise ValueError("condition-selection policy leaves no evaluable conditions")
    human_review_performed = review_freeze.get("schema_version") == "sac-review-freeze-v2"
    if role == "final_test":
        if protocol_path is None or control_ecdf_path is None:
            raise ValueError("final-test evaluation requires the experiment protocol and validation ECDF")
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        run_key = f"{run.model.name}-seed{run.seed}"
        if (protocol.get("schema_version") not in {"sac-experiment-protocol-v4", "sac-experiment-protocol-v5"} or protocol.get("config_sha256") != config_digest(run.config)
                or protocol.get("resolved_run_sha256", {}).get(run_key) != config_digest(run)):
            raise ValueError("invalid or mismatched experiment-level final-test protocol")
        if (protocol["pair_manifest_sha256"].get("final_test") != sha256_file(pair_manifest)
                or protocol["review_freeze_sha256"] != sha256_file(review_freeze_path)
                or protocol["checkpoint_sha256"].get(run_key) != sha256_file(checkpoint_path)
                or protocol["control_ecdf_sha256"].get(run_key) != sha256_file(control_ecdf_path)
                or protocol.get("condition_signature") != pair_condition_signature(run.config)):
            raise ValueError("final-test inputs differ from the experiment-level freeze")
        if software_rerun_reason is not None and not software_rerun_reason.strip():
            raise ValueError("software rerun reason cannot be blank")
        prior = out / "provenance.json"
        if prior.exists() and software_rerun_reason is None:
            raise ValueError("final-test output already exists; exact software rerun requires --software-rerun-reason")
        if software_rerun_reason is not None and not prior.exists():
            raise ValueError("software rerun reason supplied but no prior final-test output exists")
    device = _evaluation_device(run.config.train.device)
    model, training_wandb_run_id = _load_checkpoint_model(run, checkpoint_path, device)
    samples = {sample.sample_id: sample for sample in load_manifest(run.config.output_path(run.config.data.manifest_relpath))}
    batch_size = min(run.batch_size_per_device, 64)
    native_ids = list(dict.fromkeys([row["source_id"] for row in rows] + [row["control_id"] for row in rows]))
    forward_timings: list[tuple[int, float]] = []
    native_probabilities, native_features = _infer_stream(
        model, native_ids,
        lambda sample_id: preprocess_image(load_native_image(run.config.data.root, samples[sample_id]), run.config.data.input_size),
        device, batch_size, forward_timings=forward_timings,
    )
    native_index = {sample_id: index for index, sample_id in enumerate(native_ids)}
    source_indices = np.asarray([native_index[row["source_id"]] for row in rows], dtype=int)
    control_indices = np.asarray([native_index[row["control_id"]] for row in rows], dtype=int)
    p, z = native_probabilities[source_indices], native_features[source_indices]
    z_control = native_features[control_indices]
    q, z_cf = _infer_stream(
        model, rows,
        lambda row: preprocess_image(load_cached_tensor(run.config, row), run.config.data.input_size),
        device, batch_size, forward_timings=forward_timings,
    )
    control_dissimilarity = 1 - np.sum(z * z_control, axis=1) / (np.linalg.norm(z, axis=1) * np.linalg.norm(z_control, axis=1))
    first_source_indices: dict[str, int] = {}
    for index, row in enumerate(rows):
        first_source_indices.setdefault(row["source_id"], index)
    if role == "validation":
        source_idx = np.asarray(list(first_source_indices.values()))
        ecdf = fit_control_ecdf(control_dissimilarity[source_idx])
        atomic_write_json(out / "control_ecdf.json", {
            **ecdf, "schema_version": "sac-control-ecdf-v3", "model_name": run.model.name,
            "training_seed": run.seed, "config_sha256": config_digest(run.config),
            "checkpoint_sha256": sha256_file(checkpoint_path), "resolved_run_sha256": config_digest(run),
            "condition_signature": pair_condition_signature(run.config), "pair_manifest_sha256": sha256_file(pair_manifest),
            "review_freeze_sha256": sha256_file(review_freeze_path), "source_count": len(source_idx), "role": role,
        })
    else:
        if control_ecdf_path is None:
            raise ValueError("calibration/final-test evaluation requires validation-fitted control ECDF")
        ecdf = json.loads(Path(control_ecdf_path).read_text(encoding="utf-8"))
        if (ecdf.get("schema_version") != "sac-control-ecdf-v3" or ecdf.get("model_name") != run.model.name
                or ecdf.get("training_seed") != run.seed or ecdf.get("config_sha256") != config_digest(run.config)
                or ecdf.get("checkpoint_sha256") != sha256_file(checkpoint_path)
                or ecdf.get("resolved_run_sha256") != config_digest(run)
                or ecdf.get("condition_signature") != pair_condition_signature(run.config)
                or ecdf.get("pair_manifest_sha256") != review_freeze.get("pair_manifest_sha256")
                or ecdf.get("review_freeze_sha256") != sha256_file(review_freeze_path)
                or ecdf.get("role") != "validation"):
            raise ValueError("control ECDF provenance mismatch")
    normalized = apply_control_ecdf(1 - np.sum(z * z_cf, axis=1) / (np.linalg.norm(z, axis=1) * np.linalg.norm(z_cf, axis=1)), ecdf)
    targets = np.asarray([row["label"] for row in rows], dtype=int)
    pair_values = counterfactual_pair_metrics(p, q, targets, z, z_cf, z_control, normalized)
    pair_output = []
    for index, row in enumerate(rows):
        record = {
            "pair_id": row["pair_id"], "source_id": row["source_id"], "source_path": row["source_path"],
            "source_group_id": row["source_group_id"], "role": role, "family": row["family"], "severity": row["severity"],
            "latent_seed": row["latent_seed"], "audit_condition": row["audit_condition"], "condition_signature": row["condition_signature"],
            "cache_relpath": row["cache_relpath"], "cache_sha256": row["cache_sha256"], "control_id": row["control_id"],
            "label": row["label"], "class_name": row["class_name"],
            "condition_selected": row["severity"] == "sham" or row["audit_condition"] in included_conditions,
            "audit_qualified": row["severity"] == "sham" or row["audit_condition"] in included_conditions,
            "original_probabilities": p[index].tolist(), "transformed_probabilities": q[index].tolist(),
            "transformed_uncertainty": float(1 - q[index].max()),
        }
        for key, values in pair_values.items():
            value = values[index]
            record[key] = bool(value) if np.issubdtype(np.asarray(value).dtype, np.bool_) else float(value)
        pair_output.append(record)
    write_jsonl(out / "pair_metrics.jsonl", pair_output)
    np.savez_compressed(out / "representations.npz", original=z, transformed=z_cf, negative_control=z_control, pair_ids=np.asarray([row["pair_id"] for row in rows]))
    unique_source_indices = {}
    for index, row in enumerate(rows):
        unique_source_indices.setdefault(row["source_id"], index)
    clean_idx = np.asarray(list(unique_source_indices.values()))
    clean_targets = targets[clean_idx]
    clean_prob = p[clean_idx]
    aggregates: dict[str, Any] = {
        "clean": classification_metrics(clean_prob, clean_targets, len(run.config.data.class_names)),
        "clean_calibration": {"ece": expected_calibration_error(clean_prob, clean_targets, run.config.evaluation.ece_bins), "brier": brier_score(clean_prob, clean_targets), "nll": negative_log_likelihood(clean_prob, clean_targets)},
        "all_included_pair_calibration_descriptive": {"ece": expected_calibration_error(q, targets, run.config.evaluation.ece_bins), "brier": brier_score(q, targets), "nll": negative_log_likelihood(q, targets)},
        "excluded_conditions_diagnostic": [{
            "condition": condition,
            "pair_count": count,
            "reason": (
                "failed frozen semantic-preservation audit"
                if human_review_performed else "excluded by frozen condition-selection policy"
            ) + "; not inferred or included in primary aggregates",
        } for condition, count in sorted(excluded_condition_counts.items())],
        "conditions": {},
    }
    by_condition: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_condition[f"{row['family']}:{row['severity']}"].append(i)
    for condition, indices in sorted(by_condition.items()):
        idx = np.asarray(indices)
        clean_correct_count = int(pair_values["clean_correct"][idx].sum())
        values: dict[str, Any] = {
            "count": len(idx), "unique_source_count": len({rows[i]["source_id"] for i in indices}),
            "transformed_accuracy": float(pair_values["transformed_correct"][idx].mean()),
            "label_consistency": float(pair_values["label_consistency"][idx].mean()),
            "js_mean": float(pair_values["js_divergence"][idx].mean()),
            "entropy_mean": float(pair_values["transformed_entropy"][idx].mean()),
            "cosine_mean_within_model": float(pair_values["cosine_similarity"][idx].mean()),
            "control_margin_mean_within_model": float(pair_values["control_margin"][idx].mean()),
            "normalized_instability_mean": float(normalized[idx].mean()),
            "failure_count": int(pair_values["counterfactual_failure"][idx].sum()),
            "clean_correct_count": clean_correct_count,
            "failure_prevalence_among_clean_correct": float(pair_values["counterfactual_failure"][idx].sum() / clean_correct_count) if clean_correct_count else None,
            "confidence_change": summarize_confidence_change(pair_values["confidence_change"][idx], pair_values["counterfactual_failure"][idx]),
            "calibration": {"ece": expected_calibration_error(q[idx], targets[idx], run.config.evaluation.ece_bins), "brier": brier_score(q[idx], targets[idx]), "nll": negative_log_likelihood(q[idx], targets[idx])},
            "class_cells": {},
        }
        for class_index, class_name in enumerate(run.config.data.class_names):
            class_idx = idx[targets[idx] == class_index]
            class_clean = int(pair_values["clean_correct"][class_idx].sum())
            clean_class_calibration = expected_calibration_error(p[class_idx], targets[class_idx], run.config.evaluation.ece_bins)["ece"]
            transformed_class_calibration = expected_calibration_error(q[class_idx], targets[class_idx], run.config.evaluation.ece_bins)["ece"]
            values["class_cells"][class_name] = {
                "count": len(class_idx), "unique_source_count": len({rows[i]["source_id"] for i in class_idx.tolist()}),
                "transformed_accuracy": float(pair_values["transformed_correct"][class_idx].mean()),
                "label_consistency": float(pair_values["label_consistency"][class_idx].mean()),
                "normalized_instability_mean": float(normalized[class_idx].mean()),
                "failure_count": int(pair_values["counterfactual_failure"][class_idx].sum()),
                "failure_prevalence_among_clean_correct": float(pair_values["counterfactual_failure"][class_idx].sum() / class_clean) if class_clean else None,
                "calibration_ece_clean": clean_class_calibration,
                "calibration_ece_transformed": transformed_class_calibration,
                "calibration_ece_change": transformed_class_calibration - clean_class_calibration,
            }
        if run.config.evaluation.cka_enabled and len(idx) >= 2:
            try:
                values["linear_cka"] = linear_cka(z[idx], z_cf[idx])
            except ValueError as exc:
                values["linear_cka"] = {"defined": False, "reason": str(exc)}
        aggregates["conditions"][condition] = values
    if role == "validation":
        control_ecdf_sha256 = sha256_file(out / "control_ecdf.json")
    else:
        if control_ecdf_path is None:
            raise ValueError("non-validation evaluation requires a validation-fitted control ECDF")
        control_ecdf_sha256 = sha256_file(control_ecdf_path)
    provenance = {
        "schema_version": "sac-evaluation-v2", "split_role": role, "model_name": run.model.name, "training_seed": run.seed,
        "config_sha256": config_digest(run.config), "condition_signature": pair_condition_signature(run.config),
        "checkpoint_sha256": sha256_file(checkpoint_path), "resolved_run_sha256": config_digest(run), "pair_manifest_sha256": sha256_file(pair_manifest),
        "review_freeze_sha256": sha256_file(review_freeze_path), "protocol_sha256": sha256_file(protocol_path) if protocol_path else None,
        "condition_selection_policy_schema": review_freeze["schema_version"],
        "human_semantic_review_performed": human_review_performed,
        "control_ecdf_sha256": control_ecdf_sha256,
        "training_wandb_run_id": training_wandb_run_id,
        "software_rerun_reason": software_rerun_reason,
        "inference": {
            "strategy": "bounded_stream_v1", "batch_size": batch_size,
            "unique_native_count": len(native_ids), "transformed_pair_count": len(rows),
            "device_type": device.type,
            "forward_image_count": sum(count for count, _ in forward_timings),
            "forward_seconds": sum(seconds for _, seconds in forward_timings),
            "forward_images_per_second": (
                sum(count for count, _ in forward_timings) / sum(seconds for _, seconds in forward_timings)
                if sum(seconds for _, seconds in forward_timings) > 0 else None
            ),
            "timing_scope": "model_forward_only_after_device_transfer",
        },
    }
    atomic_write_json(out / "provenance.json", provenance)
    atomic_write_json(out / "metrics.json", aggregates)
    atomic_write_json(out / "evaluation_index.json", {
        "schema_version": "sac-evaluation-index-v1",
        "protocol_sha256": provenance["protocol_sha256"],
        "provenance_sha256": sha256_file(out / "provenance.json"),
        "pair_metrics_sha256": sha256_file(out / "pair_metrics.jsonl"),
        "metrics_sha256": sha256_file(out / "metrics.json"),
        "representations_sha256": sha256_file(out / "representations.npz"),
    })
    publish_evaluation_to_wandb(run.config, out, provenance, aggregates)
    return {"provenance": provenance, "metrics": aggregates, "pair_count": len(rows), "evaluation_index_sha256": sha256_file(out / "evaluation_index.json")}
