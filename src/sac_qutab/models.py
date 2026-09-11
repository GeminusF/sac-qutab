from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .config import ModelPreset


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    representation: torch.Tensor


_CLASSIFIER_KEYS: dict[str, tuple[str, ...]] = {
    "resnet50": ("fc.weight", "fc.bias"),
    "convnext_tiny": ("head.fc.weight", "head.fc.bias"),
    "vit_small_patch16_224": ("head.weight", "head.bias"),
    "swin_tiny_patch4_window7_224": ("head.fc.weight", "head.fc.bias"),
    "dinov2_vits14": ("head.weight", "head.bias"),
}


def _import_timm() -> Any:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("the pinned 'timm==1.0.29' dependency is required for core models") from exc
    return timm


class FeatureTimmModel(nn.Module):
    """Expose a uniform logits/penultimate-representation contract for timm models."""

    def __init__(self, backbone: nn.Module, preset: ModelPreset) -> None:
        super().__init__()
        self.backbone = backbone
        self.preset = preset
        self.adapter_diagnostics: dict[str, Any] = {
            "adapter_version": preset.adapter_version,
            "source_input_size": preset.source_input_size,
            "target_input_size": 224,
            "adapted_tensors": [],
        }

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 224, 224):
            raise ValueError(f"{self.preset.name} input must have shape [N,3,224,224]")
        features = self.backbone.forward_features(x)  # type: ignore[attr-defined]
        representation = self.backbone.forward_head(features, pre_logits=True)  # type: ignore[attr-defined]
        if representation.ndim != 2 or representation.shape[1] != self.preset.expected_representation_dim:
            raise RuntimeError(
                f"{self.preset.name} representation contract changed: "
                f"{tuple(representation.shape)} != [N,{self.preset.expected_representation_dim}]"
            )
        classifier = self.backbone.get_classifier()  # type: ignore[attr-defined]
        logits = classifier(representation)
        return ModelOutput(logits=logits, representation=representation)


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() != ".safetensors":
        raise ValueError("core pretrained checkpoints must use the .safetensors format")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("the pinned 'safetensors==0.8.0' dependency is required for model weights") from exc
    state = load_file(str(path), device="cpu")
    if not state or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("pretrained checkpoint is not a non-empty tensor state dict")
    return dict(state)


def _validate_source_classifier(
    state: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor], preset: ModelPreset,
) -> tuple[str, ...]:
    classifier_keys = _CLASSIFIER_KEYS[preset.name]
    if not set(classifier_keys) <= set(target):
        raise RuntimeError(f"timm classifier layout changed for {preset.name}: {classifier_keys}")
    if preset.source_num_classes == 0:
        unexpected = set(classifier_keys) & set(state)
        if unexpected:
            raise ValueError(f"classifier-free checkpoint unexpectedly contains {sorted(unexpected)}")
        return classifier_keys
    if not set(classifier_keys) <= set(state):
        raise ValueError(f"checkpoint omits the declared source classifier: {classifier_keys}")
    weight_key, bias_key = classifier_keys
    expected_weight = (preset.source_num_classes, preset.expected_representation_dim)
    expected_bias = (preset.source_num_classes,)
    if tuple(state[weight_key].shape) != expected_weight or tuple(state[bias_key].shape) != expected_bias:
        raise ValueError(
            f"source classifier shape mismatch for {preset.name}: "
            f"{tuple(state[weight_key].shape)}/{tuple(state[bias_key].shape)}"
        )
    return classifier_keys


def _filter_dinov2_state(state: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    try:
        from timm.models.vision_transformer import checkpoint_filter_fn
    except ImportError as exc:
        raise RuntimeError("timm Vision Transformer checkpoint adapter is unavailable") from exc
    return dict(checkpoint_filter_fn(state, model))


def _adapt_pretrained_state(
    state: dict[str, torch.Tensor], model: nn.Module, preset: ModelPreset,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    target = model.state_dict()
    classifier_keys = _validate_source_classifier(state, target, preset)
    adapted = dict(state)
    diagnostics: dict[str, Any] = {
        "adapter_version": preset.adapter_version,
        "source_input_size": preset.source_input_size,
        "target_input_size": 224,
        "adapted_tensors": [],
    }
    if preset.name == "dinov2_vits14":
        filtered = _filter_dinov2_state(adapted, model)
        if set(filtered) != set(adapted):
            raise ValueError("DINOv2 adapter changed checkpoint keys instead of only positional geometry")
        changed_shapes = sorted(
            key for key in adapted if tuple(adapted[key].shape) != tuple(filtered[key].shape)
        )
        if changed_shapes != ["pos_embed"]:
            raise ValueError(f"DINOv2 adaptation must change only pos_embed, changed={changed_shapes}")
        source_shape = list(adapted["pos_embed"].shape)
        target_shape = list(filtered["pos_embed"].shape)
        if source_shape != [1, 1370, 384] or target_shape != [1, 257, 384]:
            raise ValueError(f"unexpected DINOv2 positional geometry: {source_shape} -> {target_shape}")
        adapted = dict(filtered)
        diagnostics["adapted_tensors"].append({
            "name": "pos_embed", "source_shape": source_shape, "target_shape": target_shape,
            "method": "timm_vision_transformer_checkpoint_filter_bicubic",
        })
    if preset.name.startswith("swin"):
        ignored_swin_buffers = [
            k for k in adapted
            if k.endswith("relative_position_index") or k.endswith("attn_mask")
        ]
        for k in ignored_swin_buffers:
            adapted.pop(k)
        if ignored_swin_buffers:
            diagnostics["adapted_tensors"].append({
                "name": "swin_non_persistent_buffers",
                "ignored_keys": sorted(ignored_swin_buffers),
                "method": "filtered_non_persistent_swin_buffers",
            })

    source_keys = set(adapted) - set(classifier_keys)
    target_backbone_keys = set(target) - set(classifier_keys)
    if source_keys != target_backbone_keys:
        raise ValueError(
            f"{preset.name} checkpoint coverage mismatch: "
            f"missing={sorted(target_backbone_keys-source_keys)}, extra={sorted(source_keys-target_backbone_keys)}"
        )
    for key in sorted(source_keys):
        if tuple(adapted[key].shape) != tuple(target[key].shape):
            raise ValueError(
                f"{preset.name} tensor shape mismatch for {key}: "
                f"{tuple(adapted[key].shape)} != {tuple(target[key].shape)}"
            )
    return {key: value for key, value in adapted.items() if key not in classifier_keys}, diagnostics


def build_model(
    preset: ModelPreset,
    num_classes: int,
    *,
    pretrained_checkpoint: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> nn.Module:
    if num_classes <= 1:
        raise ValueError("classification requires at least two classes")
    if preset.name == "synthetic_tiny_cnn":
        if pretrained_checkpoint is not None or expected_checkpoint_sha256 is not None:
            raise ValueError("synthetic model cannot consume pretrained weights")
        return SyntheticTinyCNN(num_classes)
    if preset.name not in _CLASSIFIER_KEYS:
        raise ValueError(f"unknown model preset: {preset.name}")

    timm = _import_timm()
    create_args: dict[str, Any] = {"pretrained": False, "num_classes": num_classes}
    if preset.name == "dinov2_vits14":
        create_args["img_size"] = 224
    backbone = timm.create_model(preset.timm_model_id, **create_args)
    model = FeatureTimmModel(backbone, preset)
    checkpoint = Path(pretrained_checkpoint) if pretrained_checkpoint is not None else None
    if checkpoint is None:
        if expected_checkpoint_sha256 is not None:
            raise ValueError("pretrained checkpoint path is required when a digest is frozen")
        return model

    from .runtime import sha256_file

    digest = sha256_file(checkpoint)
    if expected_checkpoint_sha256 is None or digest != expected_checkpoint_sha256:
        raise ValueError("pretrained checkpoint full SHA-256 differs from the compute preflight")
    if digest != preset.checkpoint_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch for {preset.name}: {digest}")
    adapted, diagnostics = _adapt_pretrained_state(_load_safetensors(checkpoint), backbone, preset)
    incompatible = backbone.load_state_dict(adapted, strict=False)
    expected_missing = set(_CLASSIFIER_KEYS[preset.name])
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"{preset.name} adapted checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.adapter_diagnostics = diagnostics
    return model


class SyntheticTinyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        representation = self.features(x).flatten(1)
        return ModelOutput(self.head(representation), representation)


def model_metadata(
    model: nn.Module,
    preset: ModelPreset,
    num_classes: int,
    pretrained_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    return {
        "name": preset.name,
        "timm_model_id": preset.timm_model_id,
        "checkpoint_id": preset.checkpoint_id,
        "checkpoint_url": preset.checkpoint_url,
        "checkpoint_sha256": preset.checkpoint_sha256,
        "pretrained_checkpoint_sha256": pretrained_checkpoint_sha256,
        "source_num_classes": preset.source_num_classes,
        "source_input_size": preset.source_input_size,
        "pretraining": preset.pretraining,
        "adapter_version": preset.adapter_version,
        "adapter_diagnostics": getattr(model, "adapter_diagnostics", None),
        "representation": preset.representation,
        "expected_representation_dim": preset.expected_representation_dim,
        "num_classes": num_classes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def validate_local_checkpoint(preset: ModelPreset, path: str, num_classes: int) -> dict[str, Any]:
    from .runtime import sha256_file

    digest = sha256_file(path)
    model = build_model(
        preset, num_classes, pretrained_checkpoint=path, expected_checkpoint_sha256=digest,
    )
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    return {
        "model": preset.name, "timm_model_id": preset.timm_model_id,
        "checkpoint_path": str(Path(path).resolve()), "sha256": digest,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "logits_shape": list(output.logits.shape), "representation_shape": list(output.representation.shape),
        "adapter_version": preset.adapter_version,
        "adapter_diagnostics": getattr(model, "adapter_diagnostics", None),
    }
