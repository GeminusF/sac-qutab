from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from sac_qutab.config import ModelPreset, ResolvedRun, load_config, materialize_run
from sac_qutab.interpretability import attention_rollout
import sac_qutab.models as model_module
from sac_qutab.models import SyntheticTinyCNN, build_model
from sac_qutab.runtime import seed_everything
import sac_qutab.train as train_module
from sac_qutab.train import (
    DistributedContext,
    GlobalBatchShardSampler,
    Progress,
    _loader_kwargs,
    _move_batch_to_device,
    _projection_summary,
    _visible_device_selectors,
    load_checkpoint,
    run_training,
    save_checkpoint,
)


CONFIG = Path("configs/core.yaml")


def test_required_models_build_without_pretrained_network_and_expose_contract(monkeypatch) -> None:
    cfg = load_config(CONFIG)

    dimensions = {preset.timm_model_id: preset.expected_representation_dim for preset in cfg.experiment.model_presets}
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeBackbone(torch.nn.Module):
        def __init__(self, dimension: int, classes: int) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, dimension)
            self.head = torch.nn.Linear(dimension, classes)

        def forward_features(self, value: torch.Tensor) -> torch.Tensor:
            return value.mean(dim=(2, 3))

        def forward_head(self, value: torch.Tensor, *, pre_logits: bool) -> torch.Tensor:
            assert pre_logits
            return self.projection(value)

        def get_classifier(self) -> torch.nn.Module:
            return self.head

    def create_model(model_id: str, **kwargs):
        calls.append((model_id, kwargs))
        return FakeBackbone(dimensions[model_id], int(kwargs["num_classes"]))

    monkeypatch.setattr(model_module, "_import_timm", lambda: SimpleNamespace(create_model=create_model))
    for preset in cfg.experiment.model_presets:
        model = build_model(preset, 10)
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 224, 224))
        assert result.logits.shape == (1, 10)
        assert result.representation.shape == (1, preset.expected_representation_dim)
    assert len(calls) == 5
    assert all(kwargs["pretrained"] is False for _, kwargs in calls)
    assert dict(calls)["vit_small_patch14_dinov2.lvd142m"]["img_size"] == 224


def test_safetensors_loader_is_strict_and_replaces_only_declared_classifier(tmp_path: Path, monkeypatch) -> None:
    from safetensors.torch import save_file
    from sac_qutab.runtime import sha256_file

    cfg = load_config(CONFIG)
    preset = next(item for item in cfg.experiment.model_presets if item.name == "resnet50")

    class FakeResNet(torch.nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 2048)
            self.fc = torch.nn.Linear(2048, classes)

        def forward_features(self, value: torch.Tensor) -> torch.Tensor:
            return value.mean(dim=(2, 3))

        def forward_head(self, value: torch.Tensor, *, pre_logits: bool) -> torch.Tensor:
            assert pre_logits
            return self.projection(value)

        def get_classifier(self) -> torch.nn.Module:
            return self.fc

    monkeypatch.setattr(
        model_module, "_import_timm",
        lambda: SimpleNamespace(create_model=lambda _model_id, **kwargs: FakeResNet(int(kwargs["num_classes"]))),
    )
    target = FakeResNet(10).state_dict()
    source = {key: value.clone() for key, value in target.items() if not key.startswith("fc.")}
    source["fc.weight"] = torch.zeros(1000, 2048)
    source["fc.bias"] = torch.zeros(1000)
    checkpoint = tmp_path / "resnet50.safetensors"
    save_file(source, str(checkpoint))
    preset = replace(preset, checkpoint_sha256=sha256_file(checkpoint))
    model = build_model(
        preset, 10, pretrained_checkpoint=checkpoint, expected_checkpoint_sha256=preset.checkpoint_sha256,
    )
    result = model(torch.zeros(1, 3, 224, 224))
    assert result.logits.shape == (1, 10) and result.representation.shape == (1, 2048)

    source["unexpected.weight"] = torch.zeros(1)
    save_file(source, str(checkpoint))
    changed = replace(preset, checkpoint_sha256=sha256_file(checkpoint))
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_model(changed, 10, pretrained_checkpoint=checkpoint, expected_checkpoint_sha256=changed.checkpoint_sha256)


def test_dinov2_adapter_allows_only_documented_positional_resample(tmp_path: Path, monkeypatch) -> None:
    from safetensors.torch import save_file
    from sac_qutab.runtime import sha256_file

    cfg = load_config(CONFIG)
    preset = next(item for item in cfg.experiment.model_presets if item.name == "dinov2_vits14")

    class FakeDino(torch.nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 384))
            self.projection = torch.nn.Linear(3, 384)
            self.head = torch.nn.Linear(384, classes)

        def forward_features(self, value: torch.Tensor) -> torch.Tensor:
            return value.mean(dim=(2, 3))

        def forward_head(self, value: torch.Tensor, *, pre_logits: bool) -> torch.Tensor:
            assert pre_logits
            return self.projection(value)

        def get_classifier(self) -> torch.nn.Module:
            return self.head

    monkeypatch.setattr(
        model_module, "_import_timm",
        lambda: SimpleNamespace(create_model=lambda _model_id, **kwargs: FakeDino(int(kwargs["num_classes"]))),
    )
    raw = {
        "pos_embed": torch.zeros(1, 1370, 384),
        "projection.weight": torch.zeros(384, 3),
        "projection.bias": torch.zeros(384),
    }
    monkeypatch.setattr(
        model_module, "_filter_dinov2_state",
        lambda state, _model: {**state, "pos_embed": torch.zeros(1, 257, 384)},
    )
    checkpoint = tmp_path / "dinov2.safetensors"
    save_file(raw, str(checkpoint))
    preset = replace(preset, checkpoint_sha256=sha256_file(checkpoint))
    model = build_model(
        preset, 10, pretrained_checkpoint=checkpoint, expected_checkpoint_sha256=preset.checkpoint_sha256,
    )
    assert model.adapter_diagnostics["adapted_tensors"] == [{
        "name": "pos_embed", "source_shape": [1, 1370, 384], "target_shape": [1, 257, 384],
        "method": "timm_vision_transformer_checkpoint_filter_bicubic",
    }]


def test_attention_rollout_uses_vit_qkv_blocks_and_returns_normalized_map() -> None:
    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 2
            self.scale = 0.5
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.qkv = torch.nn.Linear(4, 12, bias=False)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attention()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.attn.qkv(value)
            return value

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_prefix_tokens = 1
            self.blocks = torch.nn.ModuleList([Block(), Block()])

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Backbone()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            tokens = torch.arange(20, dtype=images.dtype, device=images.device).reshape(1, 5, 4).expand(len(images), -1, -1)
            for block in self.backbone.blocks:
                tokens = block(tokens)
            return tokens

    result = attention_rollout(Wrapper(), torch.zeros(2, 3, 8, 8))
    assert result.shape == (2, 8, 8)
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1


def test_global_sampler_has_no_duplicates_and_world_invariant() -> None:
    one = list(GlobalBatchShardSampler(130, 64, 0, 1, 7))
    rank0 = list(GlobalBatchShardSampler(130, 64, 0, 2, 7))
    rank1 = list(GlobalBatchShardSampler(130, 64, 1, 2, 7))
    assert len(one) == 128 and len(set(one)) == 128
    assert set(rank0).isdisjoint(rank1)
    assert set(one) == set(rank0) | set(rank1)


def test_dry_run_projection_uses_full_frozen_train_count() -> None:
    cfg = load_config(CONFIG)
    run = materialize_run(cfg, "resnet50", 17)
    summary = _projection_summary(
        run,
        0.5,
        measured_sample_count=128,
        measured_optimizer_steps=2,
        full_train_sample_count=16_200,
        measured_validation_sample_count=1_600,
        full_validation_sample_count=4_050,
        measured_validation_seconds=40.0,
    )
    assert summary["measured_optimizer_steps_rank0"] == 2
    assert summary["projected_optimizer_steps_per_epoch"] == 16_200 // run.effective_global_batch
    expected_hours = 0.5 * (16_200 // run.effective_global_batch) * cfg.train.epochs / 3600
    assert np.isclose(summary["projected_epoch_cap_hours_from_median_rank0"], expected_hours)
    expected_validation_seconds = 40.0 * 4_050 / 1_600
    assert np.isclose(summary["projected_validation_seconds_per_epoch_rank0"], expected_validation_seconds)
    assert np.isclose(
        summary["projected_training_validation_hours"],
        expected_hours + expected_validation_seconds * cfg.train.epochs / 3600,
    )


def test_single_gpu_batch_fallbacks_preserve_global_batch() -> None:
    cfg = load_config(CONFIG)
    resolved = [materialize_run(cfg, "resnet50", 17, world_size=1, max_batch_size_per_device=value) for value in (32, 16, 8)]
    assert [(run.batch_size_per_device, run.grad_accum_steps, run.effective_global_batch) for run in resolved] == [
        (32, 2, 64), (16, 4, 64), (8, 8, 64),
    ]


def test_cuda_selectors_loader_policy_and_cpu_transfer(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-example-uuid")
    assert _visible_device_selectors() == ["GPU-example-uuid"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert _visible_device_selectors() == ["0"]

    assert _loader_kwargs(4, True, persistent_workers=False) == {
        "num_workers": 4, "pin_memory": True, "persistent_workers": False,
    }
    assert _loader_kwargs(4, True, persistent_workers=True)["persistent_workers"] is True
    assert _loader_kwargs(0, False, persistent_workers=True)["persistent_workers"] is False

    class TransferProbe:
        def __init__(self) -> None:
            self.non_blocking: bool | None = None

        def to(self, _device, *, non_blocking: bool):
            self.non_blocking = non_blocking
            return self

    images, targets = TransferProbe(), TransferProbe()
    _move_batch_to_device(images, targets, torch.device("cpu"), pin_memory=True)  # type: ignore[arg-type]
    assert images.non_blocking is False and targets.non_blocking is False
    cuda_images, cuda_targets = TransferProbe(), TransferProbe()
    _move_batch_to_device(cuda_images, cuda_targets, torch.device("cuda"), pin_memory=True)  # type: ignore[arg-type]
    assert cuda_images.non_blocking is True and cuda_targets.non_blocking is True


def test_explicit_cuda_fails_closed_and_auto_amp_is_capability_aware(monkeypatch) -> None:
    cfg = load_config(CONFIG)
    run = materialize_run(cfg, "resnet50", 17)
    context = DistributedContext(rank=0, world_size=1, local_rank=0, initialized_here=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        train_module._device(run, context)

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert train_module._resolved_amp_dtype("auto", torch.device("cuda")) == "bf16"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert train_module._resolved_amp_dtype("auto", torch.device("cuda")) == "fp16"
    with pytest.raises(RuntimeError, match="BF16 configured but unsupported"):
        train_module._resolved_amp_dtype("bf16", torch.device("cuda"))


def test_dry_run_uses_production_workers_and_reuses_validation_loader(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "artifacts"
    cfg = load_config(CONFIG, [f"project.output_dir={json.dumps(str(output))}", "data.num_workers=4", "train.device=cpu", "wandb.mode=offline", "wandb.log_interval_steps=1"])
    output.joinpath("data").mkdir(parents=True)
    output.joinpath("data/manifest.jsonl").write_text("{}\n", encoding="utf-8")
    output.joinpath("data/splits.json").write_text("{}\n", encoding="utf-8")
    preset = ModelPreset("synthetic_tiny_cnn", "synthetic_tiny_cnn", "synthetic", "none", "0" * 64, 0, 224, "synthetic", "synthetic-v1", .001, "pooled8", 8)
    run = ResolvedRun(cfg, preset, 17, 1, 4, 2, 8)

    class TinyDataset:
        def __init__(self, size: int) -> None:
            self.size = size
            self.epoch = 0

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, index: int):
            return torch.zeros(3, 8, 8), index % 10, f"sample-{index}"

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    observed: list[dict[str, object]] = []
    real_loader = torch.utils.data.DataLoader

    def recording_loader(*args, **kwargs):
        observed.append(dict(kwargs))
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(train_module, "DataLoader", recording_loader)
    class FakeTracker:
        mode = "offline"
        run_id = "offline-run-id"
        error = None

        def __init__(self) -> None:
            self.logs = []
            self.summary = {}
            self.exit_code = None

        def log(self, values) -> None:
            self.logs.append(values)

        def summarize(self, values) -> None:
            self.summary.update(values)

        def log_table(self, key, rows) -> None:
            self.logs.append({key: rows})

        def log_artifact_files(self, name, artifact_type, files, *, metadata, root=None) -> None:
            self.logs.append({"artifact": name, "type": artifact_type, "files": list(files), "metadata": metadata, "root": root})

        def finish(self, exit_code: int) -> None:
            self.exit_code = exit_code

    tracker = FakeTracker()
    monkeypatch.setattr(train_module, "init_wandb_tracker", lambda *_args, **_kwargs: tracker)
    summary = train_module.train_model(
        run,
        SyntheticTinyCNN(10),
        TinyDataset(200),  # type: ignore[arg-type]
        TinyDataset(8),  # type: ignore[arg-type]
        output / "dry-run",
        dry_run=True,
        full_train_sample_count=16_200,
        full_validation_sample_count=4_050,
    )
    assert len(observed) == 2
    assert observed[0]["num_workers"] == 4 and observed[0]["persistent_workers"] is False
    assert observed[1]["num_workers"] == 4 and observed[1]["persistent_workers"] is True
    assert summary["measured_validation_sample_count"] == 8
    assert summary["projected_full_validation_sample_count"] == 4_050
    assert summary["measured_validation_seconds_rank0"] > 0
    assert summary["wandb"]["run_id"] == "offline-run-id" and tracker.exit_code == 0
    assert any("train/loss" in row for row in tracker.logs)
    assert any("validation/loss" in row for row in tracker.logs)
    checkpoint = torch.load(output / "dry-run/checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["tracking"]["wandb_run_id"] == "offline-run-id"


def test_checkpoint_roundtrip_restores_rng_and_rejects_world_change(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    cfg = load_config(CONFIG, [f"project.output_dir={json.dumps(str(output))}"])
    output.joinpath("data").mkdir(parents=True)
    output.joinpath("data/manifest.jsonl").write_text("{}\n", encoding="utf-8")
    output.joinpath("data/splits.json").write_text("{}\n", encoding="utf-8")
    preset = ModelPreset("synthetic_tiny_cnn", "synthetic_tiny_cnn", "synthetic", "none", "0" * 64, 0, 224, "synthetic", "synthetic-v1", .001, "pooled8", 8)
    run = ResolvedRun(cfg, preset, 17, 1, 4, 2, 8)
    model = SyntheticTinyCNN(10)
    optimizer = AdamW(model.parameters(), lr=.001)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    context = DistributedContext(0, 1, 0, False)
    hashes = {"manifest": __import__("sac_qutab.runtime", fromlist=["sha256_file"]).sha256_file(output / "data/manifest.jsonl"), "splits": __import__("sac_qutab.runtime", fromlist=["sha256_file"]).sha256_file(output / "data/splits.json")}
    seed_everything(17)
    checkpoint = tmp_path / "checkpoint.pt"
    progress = Progress(epoch=2, next_batch=3, global_step=7, best_score=.4, best_epoch=1, bad_epochs=1)
    save_checkpoint(checkpoint, run, model, optimizer, scheduler, None, progress, context, hashes)
    expected = (torch.rand(4), np.random.rand(4))
    torch.rand(10)
    np.random.rand(10)
    restored = load_checkpoint(checkpoint, run, model, optimizer, scheduler, None, context, hashes)
    actual = (torch.rand(4), np.random.rand(4))
    assert restored == progress
    assert torch.equal(actual[0], expected[0]) and np.array_equal(actual[1], expected[1])
    changed = ResolvedRun(cfg, preset, 17, 2, 4, 1, 8)
    try:
        load_checkpoint(checkpoint, changed, model, optimizer, scheduler, None, DistributedContext(0, 2, 0, False), hashes)
    except ValueError as exc:
        assert "world size" in str(exc)
    else:
        raise AssertionError("world-size-changing resume was accepted")


def test_resumed_training_reauthenticates_frozen_pretrained_bytes(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config(CONFIG)
    weight = tmp_path / "resnet.safetensors"
    weight.write_bytes(b"authorized exact bytes")
    digest = __import__("sac_qutab.runtime", fromlist=["sha256_file"]).sha256_file(weight)
    presets = tuple(replace(preset, checkpoint_sha256=digest) if preset.name == "resnet50" else preset for preset in cfg.experiment.model_presets)
    cfg = replace(cfg, experiment=replace(cfg.experiment, model_presets=presets))
    run = materialize_run(cfg, "resnet50", 17)
    samples = [SimpleNamespace(sample_id="train"), SimpleNamespace(sample_id="validation")]
    monkeypatch.setattr(train_module, "validate_split_contract", lambda *_args: (samples, {"sample_roles": {"train": "train", "validation": "validation"}}))
    monkeypatch.setattr(train_module, "ManifestDataset", lambda *_args, **_kwargs: [0])
    weight.write_bytes(b"mutated bytes")
    with pytest.raises(ValueError, match="full SHA-256"):
        run_training(run, tmp_path / "run", resume_path=tmp_path / "resume.pt", pretrained_checkpoint_path=weight, pretrained_checkpoint_sha256=digest)
    weight.unlink()
    with pytest.raises(ValueError, match="full SHA-256"):
        run_training(run, tmp_path / "run", resume_path=tmp_path / "resume.pt", pretrained_checkpoint_path=weight, pretrained_checkpoint_sha256=digest)
