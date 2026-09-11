from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .campaign import Campaign, campaign_digest, campaign_runs, load_campaign, read_campaign_base_overrides
from .config import Config, ModelPreset, ResolvedRun, config_digest, load_config, materialize_run
from .data import ManifestDataset, build_manifest, create_splits, load_manifest, validate_split_contract
from .detector import evaluate_nested_detectors, fit_nested_detectors, load_detector
from .evaluate import create_protocol_freeze, evaluate_pairs
from .governance import create_hardware_preflight, validate_hardware_preflight
from .interventions import build_pair_manifest, materialize_pair_cache, validate_cache
from .models import SyntheticTinyCNN, validate_local_checkpoint
from .reporting import generate_report_artifacts
from .review import create_no_human_review_policy, export_review, freeze_review
from .runtime import atomic_write_json, read_jsonl, runtime_path, sha256_file
from .statistics import cluster_bootstrap_difference, generate_primary_statistics, wilcoxon_sensitivity
from .tracking import publish_detector_to_wandb
from .train import run_training, train_model

_REVIEW_ATTESTATION = "The two reviewers completed decisions independently, were blind to model outputs and condition identity, and were qualified to judge whether the intended EuroSAT class remained visually defensible."


def _cfg(args: argparse.Namespace) -> Config:
    campaign_path = getattr(args, "campaign", None)
    overrides = [*(read_campaign_base_overrides(campaign_path) if campaign_path else []), *getattr(args, "set", [])]
    cfg = load_config(args.config, overrides)
    if campaign_path:
        campaign = load_campaign(campaign_path, cfg)
        os.environ["SAC_QUTAB_CAMPAIGN_NAME"] = campaign.name
        os.environ["SAC_QUTAB_CAMPAIGN_SHA256"] = campaign_digest(campaign)
    return cfg


def _campaign(args: argparse.Namespace, cfg: Config) -> Campaign | None:
    path = getattr(args, "campaign", None)
    return load_campaign(path, cfg) if path else None


def _matrix(cfg: Config, campaign: Campaign | None) -> tuple[tuple[str, ...], tuple[int, ...]]:
    return (
        campaign.active_models if campaign else tuple(preset.name for preset in cfg.experiment.model_presets),
        campaign.seeds if campaign else tuple(cfg.experiment.seeds),
    )


def _run(args: argparse.Namespace) -> ResolvedRun:
    cfg = _cfg(args)
    campaign = _campaign(args, cfg)
    if campaign and args.model not in campaign.active_models:
        raise ValueError(f"model is outside campaign {campaign.name}: {args.model}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested = getattr(args, "max_device_batch", None)
    frozen_cap = campaign.max_device_batch[args.model] if campaign else requested
    if campaign and requested is not None and requested != frozen_cap and not getattr(args, "dry_run", False):
        raise ValueError("--max-device-batch cannot override the frozen campaign; update and re-hash the campaign before full runs")
    cap = requested if campaign and getattr(args, "dry_run", False) and requested is not None else frozen_cap
    return materialize_run(cfg, args.model, args.seed, world_size, cap)


def _mapping(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in result:
            raise ValueError(f"invalid or duplicate {label}: {value}")
        result[name] = path
    return result


def _require_protocol_campaign(args: argparse.Namespace, cfg: Config, protocol_path: str | Path) -> Campaign | None:
    campaign = _campaign(args, cfg)
    if campaign:
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        if protocol.get("campaign", {}).get("sha256") != campaign_digest(campaign):
            raise ValueError("protocol campaign SHA-256 differs from --campaign")
    return campaign


def _validated_weight_contract(path: str | Path, preset: ModelPreset) -> tuple[str, str]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoint_path = runtime_path(artifact.get("checkpoint_path", ""))
    digest = artifact.get("sha256")
    if (artifact.get("model") != preset.name or not isinstance(digest, str)
            or digest != preset.checkpoint_sha256
            or not checkpoint_path.is_file() or sha256_file(checkpoint_path) != digest):
        raise ValueError(f"weight preflight does not authenticate the exact local checkpoint for {preset.name}")
    return str(checkpoint_path), digest


def _json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def cmd_validate(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    campaign = _campaign(args, cfg)
    runs = list(campaign_runs(campaign, cfg, args.world_size).values()) if campaign else [materialize_run(cfg, preset.name, seed, args.world_size) for preset in cfg.experiment.model_presets for seed in cfg.experiment.seeds]
    _json({"config_sha256": config_digest(cfg), "campaign_sha256": campaign_digest(campaign) if campaign else None, "runs": [{"model": run.model.name, "seed": run.seed, "world_size": run.world_size, "batch_size_per_device": run.batch_size_per_device, "grad_accum_steps": run.grad_accum_steps, "effective_global_batch": run.effective_global_batch, "resolved_sha256": config_digest(run)} for run in runs]})


def cmd_validate_weights(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    preset = next((item for item in cfg.experiment.model_presets if item.name == args.model), None)
    if preset is None:
        raise ValueError(f"unknown model preset: {args.model}")
    _json(validate_local_checkpoint(preset, args.checkpoint, len(cfg.data.class_names)))


def cmd_hardware_preflight(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    campaign = _campaign(args, cfg)
    _json(create_hardware_preflight(
        cfg, args.splits, _mapping(args.summary, "summary"), _mapping(args.weight_preflight, "weight preflight"), args.output,
        model_names=campaign.active_models if campaign else None, seeds=campaign.seeds if campaign else None,
        batch_caps=campaign.max_device_batch if campaign else None,
        campaign_name=campaign.name if campaign else None,
        campaign_sha256=campaign_digest(campaign) if campaign else None,
        max_memory_fraction=campaign.max_memory_fraction if campaign else 1.0,
        max_projected_hours_per_seed=campaign.max_projected_hours_per_seed if campaign else None,
    ))


def cmd_manifest(args: argparse.Namespace) -> None:
    _json(build_manifest(_cfg(args), Path(args.output) if args.output else None, allow_count_mismatch=args.allow_count_mismatch))


def cmd_split(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    _json(create_splits(cfg, Path(args.manifest) if args.manifest else None, Path(args.output) if args.output else None))


def cmd_data_verify(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    manifest = Path(args.manifest) if args.manifest else cfg.output_path(cfg.data.manifest_relpath)
    split_path = Path(args.splits) if args.splits else cfg.output_path(cfg.data.split_relpath)
    samples, split = validate_split_contract(cfg, manifest, split_path)
    _json({"samples": len(samples), "groups": len({sample.group_id for sample in samples}), "manifest_sha256": sha256_file(manifest), "split_sha256": sha256_file(split_path), "counts": split["counts"]})


def cmd_pairs(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    manifest = cfg.output_path(cfg.data.manifest_relpath)
    samples, split = validate_split_contract(cfg, manifest, cfg.output_path(cfg.data.split_relpath))
    _json(build_pair_manifest(cfg, samples, split, args.role, args.output))


def cmd_cache(args: argparse.Namespace) -> None:
    _json(materialize_pair_cache(_cfg(args), args.pairs, args.output))


def cmd_cache_validate(args: argparse.Namespace) -> None:
    _json(validate_cache(_cfg(args), args.pairs))


def cmd_review_export(args: argparse.Namespace) -> None:
    _json(export_review(_cfg(args), args.pairs, args.output_dir))


def cmd_review_freeze(args: argparse.Namespace) -> None:
    _json(freeze_review(_cfg(args), args.review_dir, args.pairs, args.output, args.attestation))


def cmd_condition_policy(args: argparse.Namespace) -> None:
    _json(create_no_human_review_policy(_cfg(args), args.pairs, args.output))


def cmd_train(args: argparse.Namespace) -> None:
    run = _run(args)
    campaign = _campaign(args, run.config)
    if args.dry_run:
        if not args.weight_preflight:
            raise ValueError("bounded CUDA timing requires --weight-preflight")
        weight_path, weight_sha256 = _validated_weight_contract(args.weight_preflight, run.model)
    else:
        if not args.preflight:
            raise ValueError("full training requires --preflight from hardware-preflight")
        raw = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
        if campaign:
            all_campaign_runs = campaign_runs(campaign, run.config, run.world_size)
            expected_runs = (
                {key: item for key, item in all_campaign_runs.items() if item.model.name == run.model.name}
                if raw.get("schema_version") == "sac-hardware-preflight-v1" else all_campaign_runs
            )
            preflight = validate_hardware_preflight(
                run.config, run.config.output_path(run.config.data.split_relpath), args.preflight,
                world_size=run.world_size, expected_runs=expected_runs,
                campaign_sha256=campaign_digest(campaign) if raw.get("schema_version") != "sac-hardware-preflight-v1" else None,
                max_memory_fraction=campaign.max_memory_fraction,
                max_projected_hours_per_seed=campaign.max_projected_hours_per_seed,
            )
        else:
            preflight = validate_hardware_preflight(run.config, run.config.output_path(run.config.data.split_relpath), args.preflight, world_size=run.world_size, max_device_batch=getattr(args, "max_device_batch", None))
        run_key = f"{run.model.name}-seed{run.seed}"
        if preflight.get("resolved_run_sha256", {}).get(run_key) != config_digest(run):
            raise ValueError(f"hardware preflight does not authenticate {run_key}")
        weight = preflight["weight_preflights"][run.model.name]
        weight_path, weight_sha256 = weight["checkpoint_path"], weight["checkpoint_sha256"]
    _json(run_training(run, runtime_path(args.run_dir), resume_path=runtime_path(args.resume) if args.resume else None, dry_run=args.dry_run, pretrained_checkpoint_path=runtime_path(weight_path), pretrained_checkpoint_sha256=weight_sha256))

def cmd_evaluate(args: argparse.Namespace) -> None:
    run = _run(args)
    if args.protocol:
        _require_protocol_campaign(args, run.config, args.protocol)
    _json(evaluate_pairs(run, args.checkpoint, args.pairs, args.review_freeze, args.output_dir, control_ecdf_path=args.control_ecdf, protocol_path=args.protocol, software_rerun_reason=args.software_rerun_reason))


def cmd_detector_fit(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    directory = Path(args.evaluation_dir)
    rows = read_jsonl(directory / "pair_metrics.jsonl")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    result = fit_nested_detectors(rows, cfg.detector, provenance, args.output, cfg.evaluation.rq3_min_positives_per_model_seed)
    publish_detector_to_wandb(cfg, args.output, result, stage="detector-fit")
    _json(result)


def cmd_detector_evaluate(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    directory = Path(args.evaluation_dir)
    rows = read_jsonl(directory / "pair_metrics.jsonl")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    result = evaluate_nested_detectors(rows, load_detector(args.detector), provenance, cfg.evaluation.rq3_min_positives_per_model_seed, detector_path=args.detector, protocol_path=args.protocol)
    atomic_write_json(args.output, result)
    publish_detector_to_wandb(cfg, args.output, result, stage="detector-evaluation")
    _json(result)


def cmd_statistics(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    left, right = read_jsonl(args.left), read_jsonl(args.right)
    result = cluster_bootstrap_difference(left, right, args.metric, cfg.statistics.bootstrap_samples, cfg.statistics.confidence_level, cfg.statistics.seed)
    if args.metric in cfg.statistics.wilcoxon_metrics:
        lmap, rmap = {row["pair_id"]: row for row in left}, {row["pair_id"]: row for row in right}
        ids = sorted(lmap)
        result["wilcoxon_sensitivity"] = wilcoxon_sensitivity([lmap[item][args.metric] for item in ids], [rmap[item][args.metric] for item in ids])
    atomic_write_json(args.output, result)
    _json(result)


def cmd_primary_statistics(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    _require_protocol_campaign(args, cfg, args.protocol)
    _json(generate_primary_statistics(cfg, args.evaluation_dir, args.output, protocol_path=args.protocol))


def cmd_report_inputs(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for path_text in args.metrics:
        path = Path(path_text)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        destination = output / f"{path.parent.name}_{path.name}"
        atomic_write_json(destination, metrics)
        index.append({"source": str(path), "source_sha256": sha256_file(path), "copied": destination.name})
    atomic_write_json(output / "index.json", {"schema_version": "sac-report-inputs-v1", "inputs": index, "warning": "Synthetic inputs are wiring evidence, not experimental results."})
    _json({"inputs": len(index), "output": str(output)})


def cmd_report_artifacts(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    _require_protocol_campaign(args, cfg, args.protocol)
    _json(generate_report_artifacts(
        cfg, args.evaluation_dir, args.output_dir, protocol_path=args.protocol,
        statistics_path=args.statistics, review_freeze_path=args.review_freeze,
        training_run_dirs=args.training_run_dir,
    ))


def cmd_protocol(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    campaign = _campaign(args, cfg)
    resolved_runs = campaign_runs(campaign, cfg) if campaign else None
    training_preflights = _mapping(getattr(args, "training_preflight", []) or [], "training preflight")
    _json(create_protocol_freeze(
        cfg, args.splits, _mapping(args.pair_manifest, "pair manifest"), args.review_freeze,
        _mapping(args.checkpoint, "checkpoint"), _mapping(args.detector, "detector"), _mapping(args.control_ecdf, "control ECDF"),
        getattr(args, "preflight", None), args.output, max_device_batch=args.max_device_batch,
        resolved_runs=resolved_runs, campaign=campaign,
        training_preflight_paths=training_preflights or None,
        inference_preflight_path=getattr(args, "inference_preflight", None),
    ))


def cmd_checkpoint(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _json({key: value for key, value in payload.items() if key not in {"model", "optimizer", "scheduler", "scaler", "rank_states"}})


def _synthetic_images(cfg: Config, root: Path, count_per_class: int = 10) -> None:
    rng = np.random.default_rng(20260831)
    for class_index, class_name in enumerate(cfg.data.class_names):
        folder = root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count_per_class):
            image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            image[:, :, class_index % 3] = np.clip(image[:, :, class_index % 3].astype(int) + class_index * 7, 0, 255)
            Image.fromarray(image).save(folder / f"{class_index:02d}_{index:03d}.png")


def _fill_synthetic_reviews(review_dir: Path) -> Path:
    key = {row["review_id"]: row for row in read_jsonl(review_dir / "review_key.jsonl")}
    for phase in ("pilot", "final"):
        for reviewer in (1, 2):
            path = review_dir / f"{phase}_reviewer_{reviewer}.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            seen: set[tuple[str, str, str]] = set()
            for row in rows:
                if phase == "pilot":
                    row["decision"], row["pilot_usable"] = "accept", "yes"
                else:
                    info = key[row["review_id"]]
                    cell = (info["family"], info["severity"], info["class_name"])
                    row["decision"] = "reject" if cell not in seen else "accept"
                    seen.add(cell)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    adjudication_path = review_dir / "adjudication.csv"
    with adjudication_path.open(newline="", encoding="utf-8") as handle:
        adjudication = list(csv.DictReader(handle))
    for row in adjudication:
        row["decision"] = "accept"
    with adjudication_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=adjudication[0].keys())
        writer.writeheader()
        writer.writerows(adjudication)
    attestation_path = review_dir / "synthetic_attestation.json"
    atomic_write_json(attestation_path, {
        "schema_version": "sac-review-attestation-v1", "statement": _REVIEW_ATTESTATION,
        "reviewer_1": {"name": "synthetic-reviewer-a", "qualified": True, "independent": True},
        "reviewer_2": {"name": "synthetic-reviewer-b", "qualified": True, "independent": True},
        "adjudicator": {"name": "synthetic-adjudicator", "qualified": True, "independent": True},
    })
    return attestation_path


def cmd_synthetic_smoke(args: argparse.Namespace) -> None:
    root = Path(args.work_dir).resolve()
    cfg = load_config(args.config, [
        *getattr(args, "set", []),
        f"project.output_dir={json.dumps(str(root / 'artifacts'))}", f"data.root={json.dumps(str(root / 'data'))}",
        "data.split_fractions.train=0.20", "data.split_fractions.validation=0.60", "data.split_fractions.calibration=0.10", "data.split_fractions.final_test=0.10",
        "data.num_workers=0", "audit.samples_per_class_condition=5", "audit.pilot_rows=30",
        "train.device=cpu", "train.amp_dtype=none",
    ])
    _synthetic_images(cfg, cfg.data.root)
    build_manifest(cfg, allow_count_mismatch=True)
    split = create_splits(cfg)
    samples = load_manifest(cfg.output_path(cfg.data.manifest_relpath))
    pairs = cfg.project.output_dir / "counterfactuals" / "validation_pairs.jsonl"
    build_pair_manifest(cfg, samples, split, "validation", pairs)
    materialize_pair_cache(cfg, pairs)
    review_dir = cfg.project.output_dir / "counterfactuals" / "review"
    export_review(cfg, pairs, review_dir)
    attestation = _fill_synthetic_reviews(review_dir)
    freeze_path = review_dir / "freeze.json"
    freeze_review(cfg, review_dir, pairs, freeze_path, attestation)
    train_samples = [sample for sample in samples if split["sample_roles"][sample.sample_id] == "train"]
    val_samples = [sample for sample in samples if split["sample_roles"][sample.sample_id] == "validation"]
    preset = ModelPreset(
        "synthetic_tiny_cnn", "synthetic_tiny_cnn", "synthetic-only", "none", "0" * 64,
        0, 224, "synthetic", "synthetic-v1", 0.001, "pooled8", 8,
    )
    run = ResolvedRun(cfg, preset, 17, 1, 4, 2, 8)
    train_ds = ManifestDataset(cfg.data.root, train_samples, 224, train=True, seed=17, horizontal_p=.5, vertical_p=.5)
    val_ds = ManifestDataset(cfg.data.root, val_samples[:16], 224, train=False, seed=17, horizontal_p=0, vertical_p=0)
    run_dir = cfg.project.output_dir / "runs" / "synthetic"
    train_model(run, SyntheticTinyCNN(10), train_ds, val_ds, run_dir, dry_run=True)
    evaluation_dir = cfg.project.output_dir / "evaluations" / "synthetic_validation"
    evaluate_pairs(run, run_dir / "checkpoints" / "latest.pt", pairs, freeze_path, evaluation_dir)
    cmd_report_inputs(argparse.Namespace(output_dir=str(cfg.project.output_dir / "report_inputs" / "synthetic"), metrics=[str(evaluation_dir / "metrics.json")]))
    _json({"status": "synthetic_smoke_complete", "work_dir": str(root), "warning": "Synthetic behavior only; no EuroSAT/GPU/scientific claim is validated."})


def _validated_training_preflight(
    cfg: Config, campaign: Campaign | None, path: str | Path, model: str,
    resolved_runs: dict[str, ResolvedRun], world_size: int, legacy_max_batch: int | None,
) -> dict[str, Any]:
    if campaign is None:
        artifact = validate_hardware_preflight(
            cfg, cfg.output_path(cfg.data.split_relpath), path,
            world_size=world_size, max_device_batch=legacy_max_batch,
        )
    else:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") == "sac-hardware-preflight-v1":
            model_runs = {key: run for key, run in resolved_runs.items() if run.model.name == model}
            artifact = validate_hardware_preflight(
                cfg, cfg.output_path(cfg.data.split_relpath), path,
                world_size=world_size, expected_runs=model_runs,
            )
        else:
            model_runs = {key: run for key, run in resolved_runs.items() if run.model.name == model}
            validation_runs = resolved_runs if set(raw.get("resolved_run_sha256", {})) == set(resolved_runs) else model_runs
            artifact = validate_hardware_preflight(
                cfg, cfg.output_path(cfg.data.split_relpath), path, world_size=world_size,
                expected_runs=validation_runs, campaign_sha256=campaign_digest(campaign),
                max_memory_fraction=campaign.max_memory_fraction,
                max_projected_hours_per_seed=campaign.max_projected_hours_per_seed,
            )
    if model not in artifact.get("weight_preflights", {}):
        raise ValueError(f"training preflight does not contain {model}")
    if campaign:
        for key, run in resolved_runs.items():
            if run.model.name == model and artifact.get("resolved_run_sha256", {}).get(key) != config_digest(run):
                raise ValueError(f"training preflight does not authenticate {key}")
    return artifact


def cmd_run_all(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ValueError("run-all is a sequential single-process orchestrator; use train with torchrun for DDP")
    cfg = _cfg(args)
    campaign = _campaign(args, cfg)
    stage = getattr(args, "stage", "all")
    models, seeds = _matrix(cfg, campaign)
    resolved_runs = campaign_runs(campaign, cfg, world_size) if campaign else {
        f"{model}-seed{seed}": materialize_run(cfg, model, seed, world_size, getattr(args, "max_device_batch", None))
        for model in models for seed in seeds
    }
    pair_map = _mapping(args.pair_manifest, "pair manifest")
    required_pair_roles = {"validation", "calibration"} if stage == "prepare" else {"validation", "calibration", "final_test"}
    if set(pair_map) != required_pair_roles:
        raise ValueError(f"run-all --stage {stage} requires exactly {sorted(required_pair_roles)} pair manifests")

    supplied_training = _mapping(getattr(args, "training_preflight", []) or [], "training preflight")
    legacy_preflight = getattr(args, "preflight", None)
    if campaign:
        if set(supplied_training) != set(models):
            raise ValueError("campaign run-all requires --training-preflight MODEL=PATH for every active model")
        if stage in {"finalize", "all"} and not getattr(args, "inference_preflight", None):
            raise ValueError("campaign finalization requires --inference-preflight for the common A100 session")
        training_paths = supplied_training
    else:
        if not legacy_preflight:
            raise ValueError("legacy run-all requires --preflight")
        training_paths = {model: legacy_preflight for model in models}

    staged: dict[str, dict[str, str]] = {}
    checkpoint_map: dict[str, str] = {}
    detector_map: dict[str, str] = {}
    ecdf_map: dict[str, str] = {}
    preflights: dict[str, dict[str, Any]] = {}
    if stage in {"prepare", "all"}:
        for model in models:
            preflights[model] = _validated_training_preflight(
                cfg, campaign, training_paths[model], model, resolved_runs, world_size,
                getattr(args, "max_device_batch", None),
            )

    for model in models:
        for seed in seeds:
            run_key = f"{model}-seed{seed}"
            run = resolved_runs[run_key]
            run_dir = cfg.project.output_dir / "runs" / run_key
            checkpoint = run_dir / "checkpoints" / "best.pt"
            validation_dir = cfg.project.output_dir / "evaluations" / run_key / "validation"
            calibration_dir = cfg.project.output_dir / "evaluations" / run_key / "calibration"
            ecdf = validation_dir / "control_ecdf.json"
            detector = cfg.project.output_dir / "detectors" / f"{run_key}.json"
            if stage in {"prepare", "all"}:
                weight = preflights[model]["weight_preflights"][model]
                if (run_dir / "summary.json").is_file() and checkpoint.is_file():
                    print(f"[run-all] Reusing completed training run without rewrite: {run_key}")
                else:
                    run_training(run, run_dir, pretrained_checkpoint_path=weight["checkpoint_path"], pretrained_checkpoint_sha256=weight["checkpoint_sha256"])
                if not (validation_dir / "pair_metrics.jsonl").is_file():
                    evaluate_pairs(run, checkpoint, pair_map["validation"], args.review_freeze, validation_dir)
                if not (calibration_dir / "pair_metrics.jsonl").is_file():
                    evaluate_pairs(run, checkpoint, pair_map["calibration"], args.review_freeze, calibration_dir, control_ecdf_path=ecdf)
                if not detector.is_file():
                    calibration_rows = read_jsonl(calibration_dir / "pair_metrics.jsonl")
                    calibration_provenance = json.loads((calibration_dir / "provenance.json").read_text(encoding="utf-8"))
                    detector_fit = fit_nested_detectors(calibration_rows, cfg.detector, calibration_provenance, detector, cfg.evaluation.rq3_min_positives_per_model_seed)
                    publish_detector_to_wandb(cfg, detector, detector_fit, stage="detector-fit")
            staged[run_key] = {"checkpoint": str(checkpoint), "detector": str(detector), "ecdf": str(ecdf)}
            checkpoint_map[run_key], detector_map[run_key], ecdf_map[run_key] = str(checkpoint), str(detector), str(ecdf)

    preparation_path = cfg.project.output_dir / "preparation_summary.json"
    if stage == "prepare":
        atomic_write_json(preparation_path, {
            "schema_version": "sac-preparation-v1", "config_sha256": config_digest(cfg),
            "campaign_sha256": campaign_digest(campaign) if campaign else None,
            "runs": staged, "final_test_accessed": False,
        })
        _json({"stage": "prepare", "runs": len(staged), "summary": str(preparation_path), "final_test_accessed": False})
        return

    protocol_path = cfg.project.output_dir / "freezes" / "experiment.json"
    if not protocol_path.is_file():
        if campaign:
            create_protocol_freeze(
                cfg, cfg.output_path(cfg.data.split_relpath), pair_map, args.review_freeze,
                checkpoint_map, detector_map, ecdf_map, legacy_preflight, protocol_path,
                resolved_runs=resolved_runs, campaign=campaign,
                training_preflight_paths=training_paths,
                inference_preflight_path=getattr(args, "inference_preflight", None),
            )
        else:
            create_protocol_freeze(
                cfg, cfg.output_path(cfg.data.split_relpath), pair_map, args.review_freeze,
                checkpoint_map, detector_map, ecdf_map, legacy_preflight, protocol_path,
                resolved_runs=resolved_runs,
            )
    else:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (protocol.get("config_sha256") != config_digest(cfg)
                or set(protocol.get("resolved_run_sha256", {})) != set(resolved_runs)
                or any(protocol.get("checkpoint_sha256", {}).get(key) != sha256_file(path) for key, path in checkpoint_map.items())
                or any(protocol.get("detector_sha256", {}).get(key) != sha256_file(path) for key, path in detector_map.items())
                or any(protocol.get("control_ecdf_sha256", {}).get(key) != sha256_file(path) for key, path in ecdf_map.items())
                or (campaign and protocol.get("campaign", {}).get("sha256") != campaign_digest(campaign))
                or (campaign and protocol.get("training_hardware_preflight_sha256") != {
                    model: sha256_file(training_paths[model]) for model in models
                })
                or (campaign and protocol.get("inference_hardware_preflight_sha256") != sha256_file(getattr(args, "inference_preflight", "")))):
            raise ValueError("existing protocol does not authenticate the requested campaign inputs")

    results: list[dict[str, str]] = []
    for model in models:
        for seed in seeds:
            run_key = f"{model}-seed{seed}"
            run, paths = resolved_runs[run_key], staged[run_key]
            test_dir = cfg.project.output_dir / "evaluations" / run_key / "final_test"
            if not (test_dir / "detector_metrics.json").is_file():
                evaluate_pairs(run, paths["checkpoint"], pair_map["final_test"], args.review_freeze, test_dir, control_ecdf_path=paths["ecdf"], protocol_path=protocol_path)
                test_rows = read_jsonl(test_dir / "pair_metrics.jsonl")
                test_provenance = json.loads((test_dir / "provenance.json").read_text(encoding="utf-8"))
                detector_metrics = evaluate_nested_detectors(test_rows, load_detector(paths["detector"]), test_provenance, cfg.evaluation.rq3_min_positives_per_model_seed, detector_path=paths["detector"], protocol_path=protocol_path)
                atomic_write_json(test_dir / "detector_metrics.json", detector_metrics)
                publish_detector_to_wandb(cfg, test_dir / "detector_metrics.json", detector_metrics, stage="detector-evaluation")
            results.append({"run": run_key, "test_evaluation": str(test_dir), "detector": paths["detector"]})

    evaluation_dirs = [item["test_evaluation"] for item in results]
    statistics_path = cfg.project.output_dir / "statistics" / "primary.json"
    generate_primary_statistics(cfg, evaluation_dirs, statistics_path, protocol_path=protocol_path)
    report_dir = cfg.project.output_dir / "report_artifacts" / "final"
    training_run_dirs = [cfg.project.output_dir / "runs" / key for key in resolved_runs]
    generate_report_artifacts(
        cfg, evaluation_dirs, report_dir, protocol_path=protocol_path,
        statistics_path=statistics_path, review_freeze_path=args.review_freeze,
        training_run_dirs=training_run_dirs,
    )
    summary_path = cfg.project.output_dir / "run_all_summary.json"
    atomic_write_json(summary_path, {
        "schema_version": "sac-run-all-v3", "stage": stage,
        "campaign_sha256": campaign_digest(campaign) if campaign else None,
        "protocol": str(protocol_path), "protocol_sha256": sha256_file(protocol_path),
        "statistics": str(statistics_path), "runs": results, "report_artifacts": str(report_dir),
    })
    _json({"stage": stage, "runs": len(results), "summary": str(summary_path), "report_artifacts": str(report_dir)})


def _base(sub: argparse._SubParsersAction, name: str, function: Any, help_text: str) -> argparse.ArgumentParser:
    parser = sub.add_parser(name, help=help_text)
    parser.add_argument("--config", default="configs/core.yaml")
    parser.add_argument("--campaign", help="external campaign selection; does not enter Config/ResolvedRun hashes")
    parser.add_argument("--set", action="append", default=[])
    parser.set_defaults(function=function)
    return parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sac-qutab", description="SAC-QUTAB reproducible pre-training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = _base(sub, "validate-config", cmd_validate, "validate and materialize the run matrix")
    p.add_argument("--world-size", type=int, default=1)
    p = _base(sub, "validate-weights", cmd_validate_weights, "verify a local pretrained checkpoint and strict adapter")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p = _base(sub, "hardware-preflight", cmd_hardware_preflight, "freeze measured CUDA memory and timing evidence")
    p.add_argument("--splits", required=True)
    p.add_argument("--summary", action="append", required=True)
    p.add_argument("--weight-preflight", action="append", required=True)
    p.add_argument("--output", required=True)
    p = _base(sub, "data-manifest", cmd_manifest, "build an integrity manifest without downloading data")
    p.add_argument("--output")
    p.add_argument("--allow-count-mismatch", action="store_true", help=argparse.SUPPRESS)
    p = _base(sub, "split-create", cmd_split, "create grouped deterministic splits")
    p.add_argument("--manifest")
    p.add_argument("--output")
    p = _base(sub, "data-verify", cmd_data_verify, "verify manifest, split hash, and group isolation")
    p.add_argument("--manifest")
    p.add_argument("--splits")
    p = _base(sub, "pairs-create", cmd_pairs, "create a deterministic paired intervention manifest")
    p.add_argument("--role", required=True, choices=["validation", "calibration", "final_test"])
    p.add_argument("--output", required=True)
    p = _base(sub, "cache-generate", cmd_cache, "materialize deterministic counterfactual cache")
    p.add_argument("--pairs", required=True)
    p.add_argument("--output")
    p = _base(sub, "cache-validate", cmd_cache_validate, "validate cache digests and provenance")
    p.add_argument("--pairs", required=True)
    p = _base(sub, "review-export", cmd_review_export, "export independent blinded pilot/final review sheets")
    p.add_argument("--pairs", required=True)
    p.add_argument("--output-dir", required=True)
    p = _base(sub, "review-freeze", cmd_review_freeze, "validate and freeze human review decisions")
    p.add_argument("--review-dir", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--attestation", required=True)
    p.add_argument("--output", required=True)
    p = _base(sub, "condition-policy", cmd_condition_policy, "freeze an explicit no-human-review condition policy")
    p.add_argument("--pairs", required=True)
    p.add_argument("--output", required=True)
    p = _base(sub, "train", cmd_train, "run one bounded timing or full training job")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--preflight")
    p.add_argument("--resume")
    p.add_argument("--max-device-batch", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--weight-preflight", help="validate-weights JSON required for bounded CUDA timing")
    p = _base(sub, "evaluate", cmd_evaluate, "evaluate paired predictions and representations")
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-device-batch", type=int)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--review-freeze", "--condition-policy", dest="review_freeze", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--control-ecdf")
    p.add_argument("--protocol")
    p.add_argument("--software-rerun-reason")
    p = _base(sub, "detector-fit", cmd_detector_fit, "fit nested detectors from calibration only")
    p.add_argument("--evaluation-dir", required=True)
    p.add_argument("--output", required=True)
    p = _base(sub, "detector-evaluate", cmd_detector_evaluate, "score frozen detectors on final test")
    p.add_argument("--evaluation-dir", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p = _base(sub, "statistics", cmd_statistics, "run one paired original-image-cluster contrast")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--output", required=True)
    p = _base(sub, "primary-statistics", cmd_primary_statistics, "generate the frozen configured-run primary analysis")
    p.add_argument("--evaluation-dir", action="append", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p = _base(sub, "report-inputs", cmd_report_inputs, "collect hash-linked non-scientific report inputs")
    p.add_argument("--metrics", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p = _base(sub, "report-artifacts", cmd_report_artifacts, "generate final frozen scientific tables and figures")
    p.add_argument("--evaluation-dir", action="append", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--statistics", required=True)
    p.add_argument("--review-freeze", "--condition-policy", dest="review_freeze", required=True)
    p.add_argument("--training-run-dir", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p = _base(sub, "freeze-protocol", cmd_protocol, "hash-lock all configured final-test inputs")
    p.add_argument("--splits", required=True)
    p.add_argument("--pair-manifest", action="append", required=True)
    p.add_argument("--review-freeze", "--condition-policy", dest="review_freeze", required=True)
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--detector", action="append", required=True)
    p.add_argument("--control-ecdf", action="append", required=True)
    p.add_argument("--preflight", help="legacy single hardware preflight")
    p.add_argument("--training-preflight", action="append", default=[], help="repeat MODEL=PATH for mixed-hardware training provenance")
    p.add_argument("--inference-preflight", help="common final-inference hardware preflight")
    p.add_argument("--max-device-batch", type=int)
    p.add_argument("--output", required=True)
    p = sub.add_parser("inspect-checkpoint", help="inspect checkpoint metadata without constructing a model")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(function=cmd_checkpoint)
    p = _base(sub, "synthetic-smoke", cmd_synthetic_smoke, "run the bounded synthetic validation pipeline")
    p.add_argument("--work-dir", required=True)
    p = _base(sub, "run-all", cmd_run_all, "one-command configured-matrix training and frozen headline evaluation")
    p.add_argument("--pair-manifest", action="append", required=True)
    p.add_argument("--review-freeze", "--condition-policy", dest="review_freeze", required=True)
    p.add_argument("--preflight", help="legacy single hardware preflight")
    p.add_argument("--training-preflight", action="append", default=[], help="repeat MODEL=PATH for mixed-hardware training provenance")
    p.add_argument("--inference-preflight", help="common final-inference hardware preflight")
    p.add_argument("--max-device-batch", type=int)
    p.add_argument("--stage", choices=["prepare", "finalize", "all"], default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
