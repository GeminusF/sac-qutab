from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import Config, ModelPairContrast, ResolvedRun, config_digest, materialize_run


@dataclass(frozen=True)
class Campaign:
    source_sha256: str
    schema_version: str
    name: str
    selected_at: str
    active_models: tuple[str, ...]
    seeds: tuple[int, ...]
    model_pair_contrasts: tuple[ModelPairContrast, ...]
    max_device_batch: dict[str, int]
    base_config_sha256: str
    base_overrides: dict[str, Any]
    max_memory_fraction: float
    max_projected_hours_per_seed: float
    supersedes: str
    legacy_provenance: str
    rationale: str


_FIELDS = {
    "schema_version", "name", "selected_at", "active_models", "seeds",
    "model_pair_contrasts", "max_device_batch", "base_config_sha256", "base_overrides",
    "max_memory_fraction", "max_projected_hours_per_seed", "supersedes",
    "legacy_provenance", "rationale",
}


def read_campaign_base_overrides(path: str | Path) -> list[str]:
    """Read only the base overrides, before the frozen Config is materialized."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("base_overrides"), dict):
        raise ValueError("campaign must contain a base_overrides mapping")
    return [f"{key}={json.dumps(value)}" for key, value in raw["base_overrides"].items()]


def load_campaign(path: str | Path, cfg: Config) -> Campaign:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        unknown = sorted(set(raw or {}) - _FIELDS)
        missing = sorted(_FIELDS - set(raw or {}))
        raise ValueError(f"campaign fields mismatch: unknown={unknown}, missing={missing}")
    pairs_raw = raw["model_pair_contrasts"]
    if not isinstance(pairs_raw, list) or any(not isinstance(item, dict) or set(item) != {"left", "right"} for item in pairs_raw):
        raise ValueError("campaign model_pair_contrasts must be left/right mappings")
    campaign = Campaign(
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        schema_version=str(raw["schema_version"]),
        name=str(raw["name"]),
        selected_at=str(raw["selected_at"]),
        active_models=tuple(raw["active_models"]),
        seeds=tuple(int(value) for value in raw["seeds"]),
        model_pair_contrasts=tuple(ModelPairContrast(str(item["left"]), str(item["right"])) for item in pairs_raw),
        max_device_batch={str(key): int(value) for key, value in raw["max_device_batch"].items()},
        base_config_sha256=str(raw["base_config_sha256"]),
        base_overrides=dict(raw["base_overrides"]),
        max_memory_fraction=float(raw["max_memory_fraction"]),
        max_projected_hours_per_seed=float(raw["max_projected_hours_per_seed"]),
        supersedes=str(raw["supersedes"]),
        legacy_provenance=str(raw["legacy_provenance"]),
        rationale=str(raw["rationale"]),
    )
    _validate_campaign(campaign, cfg)
    return campaign


def _validate_campaign(campaign: Campaign, cfg: Config) -> None:
    if campaign.schema_version != "sac-campaign-v1" or not campaign.name.strip():
        raise ValueError("unsupported or unnamed campaign")
    if config_digest(cfg) != campaign.base_config_sha256:
        raise ValueError("campaign base_config_sha256 does not match the effective frozen Config")
    core_models = tuple(preset.name for preset in cfg.experiment.model_presets)
    if not campaign.active_models or len(set(campaign.active_models)) != len(campaign.active_models):
        raise ValueError("campaign active_models must be nonempty and unique")
    if any(model not in core_models for model in campaign.active_models):
        raise ValueError("campaign contains a model outside the frozen base Config")
    if not campaign.seeds or len(set(campaign.seeds)) != len(campaign.seeds) or any(seed not in cfg.experiment.seeds for seed in campaign.seeds):
        raise ValueError("campaign seeds must be a unique subset of the frozen base seeds")
    expected_pairs = tuple(
        ModelPairContrast(campaign.active_models[left], campaign.active_models[right])
        for left in range(len(campaign.active_models))
        for right in range(left + 1, len(campaign.active_models))
    )
    if campaign.model_pair_contrasts != expected_pairs:
        raise ValueError("campaign pairs must contain each active-model pair once in frozen campaign order")
    if set(campaign.max_device_batch) != set(campaign.active_models):
        raise ValueError("campaign requires one max_device_batch for every active model")
    target = cfg.train.target_global_batch
    if any(value <= 0 or target % value for value in campaign.max_device_batch.values()):
        raise ValueError("each campaign max_device_batch must be positive and divide the effective global batch")
    if not 0 < campaign.max_memory_fraction < 1 or campaign.max_projected_hours_per_seed <= 0:
        raise ValueError("campaign resource gates are invalid")


def campaign_digest(campaign: Campaign) -> str:
    return campaign.source_sha256


def campaign_runs(campaign: Campaign, cfg: Config, world_size: int = 1) -> dict[str, ResolvedRun]:
    return {
        f"{model}-seed{seed}": materialize_run(cfg, model, seed, world_size, campaign.max_device_batch[model])
        for model in campaign.active_models for seed in campaign.seeds
    }


def protocol_matrix(protocol: Mapping[str, Any], cfg: Config) -> tuple[tuple[str, ...], tuple[int, ...], tuple[ModelPairContrast, ...]]:
    """Resolve the matrix from v5, while retaining v4 compatibility."""
    if protocol.get("schema_version") == "sac-experiment-protocol-v5":
        campaign = protocol.get("campaign")
        if not isinstance(campaign, dict):
            raise ValueError("protocol v5 omits campaign metadata")
        models = tuple(str(value) for value in campaign.get("active_models", []))
        seeds = tuple(int(value) for value in campaign.get("seeds", []))
        pairs = tuple(ModelPairContrast(str(item["left"]), str(item["right"])) for item in campaign.get("model_pair_contrasts", []))
        return models, seeds, pairs
    if protocol.get("schema_version") == "sac-experiment-protocol-v4":
        return (
            tuple(preset.name for preset in cfg.experiment.model_presets),
            tuple(cfg.experiment.seeds),
            tuple(cfg.statistics.model_pair_contrasts),
        )
    raise ValueError("unsupported experiment protocol schema")
