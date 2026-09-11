from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sac_qutab.config import Config, config_digest, load_config, materialize_run
from sac_qutab.governance import create_hardware_preflight, validate_hardware_preflight
from sac_qutab.runtime import atomic_write_json, sha256_file, write_jsonl


def _fixture(
    tmp_path: Path, *, device_type: str = "cuda", device_name: str = "NVIDIA GeForce RTX 4090",
    world_size: int = 1, max_device_batch: int | None = None,
) -> tuple[Config, Path, dict[str, str], dict[str, str]]:
    output = tmp_path / "artifacts"
    cfg = load_config("configs/core.yaml", [f"project.output_dir={json.dumps(str(output))}"])
    checkpoints: dict[str, tuple[Path, str]] = {}
    presets = []
    for preset in cfg.experiment.model_presets:
        checkpoint = tmp_path / f"{preset.name}.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{preset.name}".encode())
        digest = sha256_file(checkpoint)
        checkpoints[preset.name] = checkpoint, digest
        presets.append(replace(preset, checkpoint_sha256=digest))
    cfg = replace(cfg, experiment=replace(cfg.experiment, model_presets=tuple(presets)))
    manifest = cfg.output_path(cfg.data.manifest_relpath)
    rows, roles = [], {}
    role_names = ("validation", "calibration", "final_test")
    for label, class_name in enumerate(cfg.data.class_names):
        for index in range(10):
            sample_id = f"sample-{label}-{index}"
            role = "train" if index < 7 else role_names[index - 7]
            roles[sample_id] = role
            rows.append({"sample_id": sample_id, "relative_path": f"{class_name}/{index}.jpg", "label": label,
                         "class_name": class_name, "width": 64, "height": 64, "mode": "RGB", "bytes": 1,
                         "sha256": f"{label:02x}{index:02x}".ljust(64, "0"), "dhash": f"{label * 10 + index:016x}",
                         "group_id": f"group-{label}-{index}"})
    write_jsonl(manifest, rows)
    split = tmp_path / "splits.json"
    atomic_write_json(split, {"schema_version": "sac-splits-v1", "manifest_sha256": sha256_file(manifest),
        "grouping_version": cfg.data.grouping_version, "seed": cfg.data.split_seed,
        "fractions": cfg.data.split_fractions, "sample_roles": roles,
        "counts": {"calibration": 10, "final_test": 10, "train": 70, "validation": 10},
        "class_counts": {"train": [7] * 10, "validation": [1] * 10, "calibration": [1] * 10, "final_test": [1] * 10},
        "group_counts": {"train": 70, "validation": 10, "calibration": 10, "final_test": 10}})
    summaries, weights = {}, {}
    for preset in cfg.experiment.model_presets:
        run = materialize_run(cfg, preset.name, 17, world_size, max_device_batch)
        projected_steps = 70 // run.effective_global_batch
        median_step = 4 * 3600 / (projected_steps * cfg.train.epochs)
        checkpoint, digest = checkpoints[preset.name]
        summary = tmp_path / f"{preset.name}-summary.json"
        rank_memory = [{"rank": rank, "reserved_bytes": 4 * 1024**3} for rank in range(world_size)]
        atomic_write_json(summary, {"status": "dry_run_complete", "model_name": preset.name, "training_seed": 17,
            "resolved_run_sha256": config_digest(run), "manifest_sha256": sha256_file(manifest),
            "split_sha256": sha256_file(split), "pretrained_checkpoint_sha256": digest, "world_size": world_size,
            "batch_size_per_device": run.batch_size_per_device, "grad_accum_steps": run.grad_accum_steps,
            "effective_global_batch": run.effective_global_batch, "precision": "fp16",
            "device": {"type": device_type, "name": device_name, "capability": [8, 9], "local_rank": 0,
                       "cuda_ordinal": 0, "visible_device_selectors": [str(i) for i in range(world_size)],
                       "total_memory_bytes": 24 * 1024**3, "multi_processor_count": 128, "cuda_runtime": "12.8"},
            "peak_gpu_reserved_bytes": 4 * 1024**3, "peak_gpu_memory_by_rank": rank_memory,
            "median_optimizer_step_seconds_rank0": median_step, "measured_sample_count": 70,
            "measured_optimizer_steps_rank0": 25, "timing_warmup_steps_rank0": 5, "timed_optimizer_steps_rank0": 20,
            "projected_full_train_sample_count": 70, "projected_optimizer_steps_per_epoch": projected_steps,
            "projected_epoch_cap_hours_from_median_rank0": 4.0, "measured_validation_sample_count": 10,
            "projected_full_validation_sample_count": 10, "measured_validation_seconds_rank0": 72.0,
            "projected_validation_seconds_per_epoch_rank0": 72.0, "projected_training_validation_hours": 5.0})
        weight = tmp_path / f"{preset.name}-weight.json"
        atomic_write_json(weight, {"model": preset.name, "checkpoint_path": str(checkpoint), "sha256": digest})
        summaries[preset.name], weights[preset.name] = str(summary), str(weight)
    return cfg, split, summaries, weights


def test_portable_cuda_preflight_accepts_generic_gpu_and_authenticates_sources(tmp_path: Path) -> None:
    cfg, split, summaries, weights = _fixture(tmp_path)
    output = tmp_path / "preflight.json"
    artifact = create_hardware_preflight(cfg, split, summaries, weights, output)
    assert artifact["schema_version"] == "sac-hardware-preflight-v1"
    assert artifact["projected_training_and_validation_hours"] == pytest.approx(75.0)
    assert validate_hardware_preflight(cfg, split, output, world_size=1)["status"] == "ready_for_training"
    Path(next(iter(summaries.values()))).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        validate_hardware_preflight(cfg, split, output, world_size=1)


def test_preflight_binds_batch_and_supports_generic_multi_gpu(tmp_path: Path) -> None:
    cfg, split, summaries, weights = _fixture(tmp_path, world_size=2, max_device_batch=16, device_name="NVIDIA L40S")
    output = tmp_path / "preflight.json"
    create_hardware_preflight(cfg, split, summaries, weights, output)
    assert validate_hardware_preflight(cfg, split, output, world_size=2, max_device_batch=16)["world_size"] == 2
    with pytest.raises(ValueError, match="resolved_run_sha256"):
        validate_hardware_preflight(cfg, split, output, world_size=2)


def test_preflight_rejects_cpu_cold_timing_and_bad_projection(tmp_path: Path) -> None:
    cfg, split, summaries, weights = _fixture(tmp_path, device_type="cpu")
    with pytest.raises(ValueError, match="CPU/local"):
        create_hardware_preflight(cfg, split, summaries, weights, tmp_path / "cpu.json")
    cfg, split, summaries, weights = _fixture(tmp_path / "cold")
    path = Path(summaries["resnet50"])
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update({"measured_optimizer_steps_rank0": 2, "timing_warmup_steps_rank0": 2, "timed_optimizer_steps_rank0": 0})
    atomic_write_json(path, summary)
    with pytest.raises(ValueError, match="five warm-up and twenty"):
        create_hardware_preflight(cfg, split, summaries, weights, tmp_path / "cold.json")
    summary.update({"measured_optimizer_steps_rank0": 25, "timing_warmup_steps_rank0": 5,
                    "timed_optimizer_steps_rank0": 20, "projected_optimizer_steps_per_epoch": 2})
    atomic_write_json(path, summary)
    with pytest.raises(ValueError, match="full frozen train split"):
        create_hardware_preflight(cfg, split, summaries, weights, tmp_path / "bad-projection.json")


def test_preflight_rejects_memory_without_headroom(tmp_path: Path) -> None:
    cfg, split, summaries, weights = _fixture(tmp_path)
    path = Path(summaries["resnet50"])
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["peak_gpu_reserved_bytes"] = 24 * 1024**3
    summary["peak_gpu_memory_by_rank"] = [{"rank": 0, "reserved_bytes": 24 * 1024**3}]
    atomic_write_json(path, summary)
    with pytest.raises(ValueError, match="memory headroom"):
        create_hardware_preflight(cfg, split, summaries, weights, tmp_path / "oom.json")
