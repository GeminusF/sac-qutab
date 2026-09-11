from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .runtime import logical_path, runtime_path


REPORT_ARTIFACT_PLAN = (
    "clean_metrics",
    "severity_condition_class_cells",
    "primary_clustered_contrasts",
    "rq3_scores_and_deltas",
    "learning_curves",
    "master_summary",
    "pairwise_forest_plot",
    "cross_model_cka_matrix",
    "throughput_vs_robustness",
    "qualitative_failure_gallery",
    "attention_rollout",
    "macro_cluster_confusion",
    "severity_response",
    "representation_scatter",
    "class_heatmap",
    "condition_confusions",
    "reliability_diagrams",
    "review_qualitative_panel",
)


class ConfigError(ValueError):
    """Raised when a configuration is incomplete, unsafe, or inconsistent."""


def _keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        raise ConfigError(f"{where}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return value


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    track: str
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    root: Path
    manifest_relpath: Path
    split_relpath: Path
    group_map_path: Path | None
    grouping_version: str
    near_duplicate_hamming_threshold: int
    dataset_version: str
    source_url: str
    license: str
    expected_samples: int
    native_size: int
    input_size: int
    class_names: tuple[str, ...]
    split_seed: int
    split_fractions: dict[str, float]
    num_workers: int
    pin_memory: bool
    horizontal_flip_probability: float
    vertical_flip_probability: float


@dataclass(frozen=True)
class ModelPreset:
    name: str
    timm_model_id: str
    checkpoint_id: str
    checkpoint_url: str
    checkpoint_sha256: str
    source_num_classes: int
    source_input_size: int
    pretraining: str
    adapter_version: str
    learning_rate: float
    representation: str
    expected_representation_dim: int


@dataclass(frozen=True)
class ExperimentConfig:
    model_presets: tuple[ModelPreset, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class TrainConfig:
    device: str
    deterministic: bool
    epochs: int
    target_global_batch: int
    max_batch_size_per_device: int
    optimizer: str
    weight_decay: float
    label_smoothing: float
    freeze_backbone: bool
    warmup_epochs: int
    min_lr_ratio: float
    early_stopping_patience: int
    min_delta: float
    amp_dtype: str
    checkpoint_every_steps: int
    dry_run_max_samples: int
    dry_run_max_steps: int


@dataclass(frozen=True)
class WandbConfig:
    mode: str
    project: str
    entity: str | None
    run_name: str | None
    group: str | None
    log_interval_steps: int


@dataclass(frozen=True)
class DistributedConfig:
    backend: str
    timeout_seconds: int
    exact_resume_requires_same_world_size: bool


@dataclass(frozen=True)
class TransformConfig:
    version: str
    seed: int
    cache_relpath: Path
    spectral_sampling: str
    spectral_gain_radii: dict[str, float]
    texture_lambdas: dict[str, float]
    resolution_sizes: dict[str, int]
    reuse_texture_donor_across_severities: bool
    include_identity_sham: bool


@dataclass(frozen=True)
class AuditConfig:
    sampling_seed: int
    samples_per_class_condition: int
    reuse_sources_across_conditions: bool
    reviewer_count: int
    pilot_rows: int
    kappa_scope: str
    disagreement_policy: str
    exclusion_granularity: str
    min_kappa: float
    min_condition_acceptance_rate: float
    min_class_acceptance_rate: float
    confidence_level: float


@dataclass(frozen=True)
class EvaluationConfig:
    ece_bins: int
    rq3_min_positives_per_model_seed: int
    pr_summary: str
    cka_enabled: bool
    representation_normalization: str


@dataclass(frozen=True)
class DetectorConfig:
    enabled: bool
    solver: str
    penalty: str
    regularization_c: float
    class_weight: str | None
    tolerance: float
    max_iter: int


@dataclass(frozen=True)
class ModelPairContrast:
    left: str
    right: str


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_samples: int
    confidence_level: float
    seed: int
    fdr_q: float
    primary_familywise_alpha: float
    model_pair_contrasts: tuple[ModelPairContrast, ...]
    primary_contrasts: tuple[str, ...]
    wilcoxon_metrics: tuple[str, ...]


@dataclass(frozen=True)
class ScopeConfig:
    rq4_spatial: bool
    mae: bool
    multispectral: bool
    gamma: bool


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    experiment: ExperimentConfig
    train: TrainConfig
    wandb: WandbConfig
    distributed: DistributedConfig
    transforms: TransformConfig
    audit: AuditConfig
    evaluation: EvaluationConfig
    detector: DetectorConfig
    statistics: StatisticsConfig
    scope: ScopeConfig

    def output_path(self, relative: Path) -> Path:
        if relative.is_absolute():
            raise ConfigError(f"generated path must be relative: {relative}")
        return self.project.output_dir / relative


@dataclass(frozen=True)
class ResolvedRun:
    config: Config
    model: ModelPreset
    seed: int
    world_size: int
    batch_size_per_device: int
    grad_accum_steps: int
    effective_global_batch: int

    def canonical(self) -> dict[str, Any]:
        return {
            "config": as_dict(self.config),
            "model": as_dict(self.model),
            "seed": self.seed,
            "world_size": self.world_size,
            "batch_size_per_device": self.batch_size_per_device,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_global_batch": self.effective_global_batch,
        }


def _dc(cls: type[Any], raw: dict[str, Any], where: str, *, path_fields: set[str] | None = None) -> Any:
    expected = {f.name for f in dataclasses.fields(cls)}
    raw = _keys(raw, expected, where)
    values = dict(raw)
    for name in path_fields or set():
        values[name] = None if values[name] is None else Path(values[name])
    return cls(**values)


def _apply_override(raw: dict[str, Any], text: str) -> None:
    if "=" not in text:
        raise ConfigError(f"override must be key=value: {text}")
    dotted, encoded = text.split("=", 1)
    parts = dotted.split(".")
    node: Any = raw
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"unknown override path: {dotted}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"unknown override path: {dotted}")
    node[parts[-1]] = yaml.safe_load(encoded)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    top = _keys(raw, {"project", "data", "experiment", "train", "wandb", "distributed", "transforms", "audit", "evaluation", "detector", "statistics", "scope"}, "root")
    top = copy.deepcopy(top)
    for override in overrides or []:
        _apply_override(top, override)

    project = _dc(ProjectConfig, top["project"], "project", path_fields={"output_dir"})
    data = _dc(DataConfig, top["data"], "data", path_fields={"root", "manifest_relpath", "split_relpath", "group_map_path"})
    exp_raw = _keys(top["experiment"], {"model_presets", "seeds"}, "experiment")
    presets: list[ModelPreset] = []
    preset_keys = {f.name for f in dataclasses.fields(ModelPreset)}
    required_preset = preset_keys
    for i, item in enumerate(exp_raw["model_presets"]):
        if not isinstance(item, dict):
            raise ConfigError(f"experiment.model_presets[{i}] must be a mapping")
        unknown = set(item) - preset_keys
        missing = required_preset - set(item)
        if unknown or missing:
            raise ConfigError(f"experiment.model_presets[{i}]: unknown={sorted(unknown)}, missing={sorted(missing)}")
        presets.append(ModelPreset(**item))
    experiment = ExperimentConfig(tuple(presets), tuple(exp_raw["seeds"]))
    train = _dc(TrainConfig, top["train"], "train")
    wandb = _dc(WandbConfig, top["wandb"], "wandb")
    distributed = _dc(DistributedConfig, top["distributed"], "distributed")
    transforms = _dc(TransformConfig, top["transforms"], "transforms", path_fields={"cache_relpath"})
    audit = _dc(AuditConfig, top["audit"], "audit")
    evaluation = _dc(EvaluationConfig, top["evaluation"], "evaluation")
    detector = _dc(DetectorConfig, top["detector"], "detector")
    statistics_raw = _keys(top["statistics"], {f.name for f in dataclasses.fields(StatisticsConfig)}, "statistics")
    pair_contrasts = tuple(
        _dc(ModelPairContrast, item, f"statistics.model_pair_contrasts[{i}]")
        for i, item in enumerate(statistics_raw["model_pair_contrasts"])
    )
    statistics = StatisticsConfig(**{
        **statistics_raw,
        "model_pair_contrasts": pair_contrasts,
        "primary_contrasts": tuple(statistics_raw["primary_contrasts"]),
        "wilcoxon_metrics": tuple(statistics_raw["wilcoxon_metrics"]),
    })
    scope = _dc(ScopeConfig, top["scope"], "scope")
    cfg = Config(project, data, experiment, train, wandb, distributed, transforms, audit, evaluation, detector, statistics, scope)
    validate_config(cfg)
    return dataclasses.replace(
        cfg,
        project=dataclasses.replace(cfg.project, output_dir=runtime_path(cfg.project.output_dir)),
        data=dataclasses.replace(
            cfg.data,
            root=runtime_path(cfg.data.root),
            group_map_path=runtime_path(cfg.data.group_map_path) if cfg.data.group_map_path is not None else None,
        ),
    )


def validate_config(cfg: Config) -> None:
    if cfg.project.track != "track1_pure_research":
        raise ConfigError("core configuration must declare Track 1")
    if any(dataclasses.astuple(cfg.scope)):
        raise ConfigError("optional RQ4/MAE/multispectral/gamma scope is deferred in the core configuration")
    if len(cfg.data.class_names) != 10 or len(set(cfg.data.class_names)) != 10:
        raise ConfigError("EuroSAT core requires ten unique classes")
    if cfg.data.expected_samples != 27000 or cfg.data.native_size != 64 or cfg.data.input_size != 224:
        raise ConfigError("EuroSAT/core geometry contract changed")
    if cfg.data.grouping_version != "content-dhash-bktree-v1" or not 0 <= cfg.data.near_duplicate_hamming_threshold <= 8:
        raise ConfigError("unsupported grouping contract")
    roles = {"train", "validation", "calibration", "final_test"}
    if set(cfg.data.split_fractions) != roles or abs(sum(cfg.data.split_fractions.values()) - 1.0) > 1e-9 or any(v <= 0 for v in cfg.data.split_fractions.values()):
        raise ConfigError("split fractions must be positive train/validation/calibration/final_test values summing to one")
    expected_presets = (
        ModelPreset("resnet50", "resnet50.tv_in1k", "timm/resnet50.tv_in1k/model.safetensors", "https://huggingface.co/timm/resnet50.tv_in1k/resolve/main/model.safetensors", "5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888", 1000, 224, "supervised_imagenet1k", "timm-safetensors-v1", 0.0003, "pre_logits_global_pool", 2048),
        ModelPreset("convnext_tiny", "convnext_tiny.fb_in1k", "timm/convnext_tiny.fb_in1k/model.safetensors", "https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/main/model.safetensors", "08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276", 1000, 224, "supervised_imagenet1k", "timm-safetensors-v1", 0.0005, "pre_logits_global_pool", 768),
        ModelPreset("vit_small_patch16_224", "vit_small_patch16_224.augreg_in1k", "timm/vit_small_patch16_224.augreg_in1k/model.safetensors", "https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k/resolve/main/model.safetensors", "c20a91d93f4757b0f581519f21ad91a875cf041764873a7a5bccc8e36f360dbf", 1000, 224, "supervised_imagenet1k", "timm-safetensors-v1", 0.0005, "final_normalized_cls", 384),
        ModelPreset("swin_tiny_patch4_window7_224", "swin_tiny_patch4_window7_224.ms_in1k", "timm/swin_tiny_patch4_window7_224.ms_in1k/model.safetensors", "https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k/resolve/main/model.safetensors", "fb01861f793143135fa0d6cd97b1631e4b33eaa3ee162bbea9e62de1c76ebac1", 1000, 224, "supervised_imagenet1k", "timm-safetensors-v1", 0.0005, "pre_logits_global_pool", 768),
        ModelPreset("dinov2_vits14", "vit_small_patch14_dinov2.lvd142m", "timm/vit_small_patch14_dinov2.lvd142m/model.safetensors", "https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/resolve/main/model.safetensors", "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081", 0, 518, "self_supervised_lvd142m", "timm-dinov2-pos-resample-v1", 0.0001, "final_normalized_cls", 384),
    )
    if cfg.experiment.model_presets != expected_presets or tuple(cfg.experiment.seeds) != (17, 29, 43):
        raise ConfigError("core model checkpoints, adapters, representations, learning rates, and seeds must match the frozen contract")
    model_names = tuple(preset.name for preset in cfg.experiment.model_presets)
    if len(set(model_names)) != len(model_names):
        raise ConfigError("core model names must be unique")
    if len({preset.checkpoint_id for preset in cfg.experiment.model_presets}) != len(model_names):
        raise ConfigError("core model checkpoints must be unique")
    for preset in cfg.experiment.model_presets:
        if (len(preset.checkpoint_sha256) != 64
                or any(character not in "0123456789abcdef" for character in preset.checkpoint_sha256)):
            raise ConfigError(f"{preset.name} checkpoint_sha256 must be a lowercase full SHA-256")
        _positive(preset.learning_rate, f"{preset.name}.learning_rate")
        _positive(preset.source_input_size, f"{preset.name}.source_input_size")
        _positive(preset.expected_representation_dim, f"{preset.name}.expected_representation_dim")
        if preset.source_num_classes < 0:
            raise ConfigError(f"{preset.name}.source_num_classes cannot be negative")
    if not cfg.train.deterministic or cfg.train.optimizer != "adamw" or cfg.train.freeze_backbone:
        raise ConfigError("core deterministic AdamW full-fine-tuning contract changed")
    for name in ("epochs", "target_global_batch", "max_batch_size_per_device", "early_stopping_patience", "checkpoint_every_steps", "dry_run_max_samples", "dry_run_max_steps"):
        _positive(getattr(cfg.train, name), f"train.{name}")
    if cfg.train.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("train.device must be auto, cpu, or cuda")
    if cfg.train.warmup_epochs >= cfg.train.epochs or cfg.train.amp_dtype not in {"auto", "none", "bf16", "fp16"}:
        raise ConfigError("invalid warmup/AMP configuration")
    if cfg.train.dry_run_max_samples < cfg.train.target_global_batch * cfg.train.dry_run_max_steps:
        raise ConfigError("dry-run sample cap cannot provide the declared optimizer steps")
    if cfg.wandb.mode not in {"online", "offline", "disabled"}:
        raise ConfigError("wandb.mode must be online, offline, or disabled")
    if not cfg.wandb.project.strip() or cfg.wandb.log_interval_steps <= 0:
        raise ConfigError("wandb.project must be nonempty and wandb.log_interval_steps must be positive")
    if not cfg.distributed.exact_resume_requires_same_world_size or cfg.distributed.backend != "nccl":
        raise ConfigError("distributed and exact-resume contract changed")
    levels = {"mild", "moderate", "severe"}
    for mapping in (cfg.transforms.spectral_gain_radii, cfg.transforms.texture_lambdas, cfg.transforms.resolution_sizes):
        if set(mapping) != levels:
            raise ConfigError("every transform requires mild/moderate/severe")
    if (list(cfg.transforms.spectral_gain_radii.values()) != sorted(cfg.transforms.spectral_gain_radii.values())
            or not cfg.transforms.reuse_texture_donor_across_severities or not cfg.transforms.include_identity_sham
            or cfg.transforms.spectral_sampling != "nested_direction_uniform" or cfg.transforms.version != "sac-cf-v1"):
        raise ConfigError("transform generator and severity-coupling contract changed")
    if (cfg.audit.reviewer_count != 2 or cfg.audit.exclusion_granularity != "family_severity"
            or not cfg.audit.reuse_sources_across_conditions or cfg.audit.kappa_scope != "all_final_rows_pre_adjudication"
            or cfg.audit.disagreement_policy != "adjudicate"):
        raise ConfigError("semantic audit independence/support contract changed")
    for value in (cfg.audit.min_kappa, cfg.audit.min_condition_acceptance_rate, cfg.audit.min_class_acceptance_rate, cfg.audit.confidence_level, cfg.statistics.confidence_level, cfg.statistics.fdr_q, cfg.statistics.primary_familywise_alpha):
        if not 0 < value < 1:
            raise ConfigError("probability/threshold values must be in (0,1)")
    if cfg.evaluation.pr_summary != "average_precision" or cfg.evaluation.representation_normalization != "validation_negative_control_ecdf":
        raise ConfigError("metric contract changed")
    if not cfg.detector.enabled or cfg.detector.solver != "lbfgs" or cfg.detector.penalty != "l2" or cfg.detector.class_weight is not None:
        raise ConfigError("nested detector contract changed")
    expected_contrasts = {
        "rq1_label_inconsistency_severe_minus_mild", "rq1_js_severe_minus_mild",
        "rq2_normalized_instability_severe_minus_mild", "rq2_cross_model_difference_in_differences",
        "rq3_nested_detector_delta_average_precision", "rq3_nested_detector_delta_auroc",
    }
    if set(cfg.statistics.primary_contrasts) != expected_contrasts or len(cfg.statistics.primary_contrasts) != len(expected_contrasts):
        raise ConfigError("primary contrast registry changed")
    expected_pairs = tuple(
        ModelPairContrast(model_names[left], model_names[right])
        for left in range(len(model_names))
        for right in range(left + 1, len(model_names))
    )
    if cfg.statistics.model_pair_contrasts != expected_pairs:
        raise ConfigError("model_pair_contrasts must contain every unique model pair once in frozen model order")


def materialize_run(cfg: Config, model_name: str, seed: int, world_size: int = 1, max_batch_size_per_device: int | None = None) -> ResolvedRun:
    presets = {p.name: p for p in cfg.experiment.model_presets}
    if model_name not in presets:
        raise ConfigError(f"model must be one of {sorted(presets)}")
    if seed not in cfg.experiment.seeds:
        raise ConfigError(f"seed must be one of {cfg.experiment.seeds}")
    _positive(world_size, "world_size")
    target = cfg.train.target_global_batch
    max_batch = max_batch_size_per_device or cfg.train.max_batch_size_per_device
    choices = [b for b in range(1, max_batch + 1) if target % (b * world_size) == 0]
    if not choices:
        raise ConfigError(f"cannot preserve global batch {target} with world_size={world_size}, max_device_batch={max_batch}")
    batch = max(choices)
    accum = target // (batch * world_size)
    return ResolvedRun(cfg, presets[model_name], seed, world_size, batch, accum, batch * accum * world_size)


def as_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: as_dict(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return logical_path(value).as_posix()
    if isinstance(value, tuple):
        return [as_dict(v) for v in value]
    if isinstance(value, dict):
        return {str(k): as_dict(v) for k, v in value.items()}
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(as_dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def config_digest(value: Config | ResolvedRun) -> str:
    return hashlib.sha256(canonical_bytes(value.canonical() if isinstance(value, ResolvedRun) else value)).hexdigest()
