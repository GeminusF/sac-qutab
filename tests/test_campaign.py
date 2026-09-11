from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sac_qutab.campaign import (
    campaign_digest,
    campaign_runs,
    load_campaign,
    protocol_matrix,
    read_campaign_base_overrides,
)
from sac_qutab.cli import build_parser
from sac_qutab.config import config_digest, load_config, materialize_run
from sac_qutab.governance import _validate_campaign_device
from sac_qutab.runtime import logical_path, runtime_path


CORE = Path("configs/core.yaml")
CAMPAIGN = Path("configs/sac-campaign-v1.yaml")


def _loaded():
    cfg = load_config(CORE, read_campaign_base_overrides(CAMPAIGN))
    return cfg, load_campaign(CAMPAIGN, cfg)


def test_legacy_core_file_and_effective_digest_are_frozen() -> None:
    assert hashlib.sha256(CORE.read_bytes()).hexdigest() == "139c544a59e98004e725f920ec023b2204046aebabd8d9b4873cece48e75e8ec"
    cfg, campaign = _loaded()
    assert config_digest(cfg) == campaign.base_config_sha256 == "cc73206d4ff490bebaa9440af5689decd4a93d08038df486dadd8dfe0deb449c"


def test_campaign_materializes_only_two_models_three_seeds_and_one_pair() -> None:
    cfg, campaign = _loaded()
    runs = campaign_runs(campaign, cfg)
    assert list(runs) == [
        "resnet50-seed17", "resnet50-seed29", "resnet50-seed43",
        "vit_small_patch16_224-seed17", "vit_small_patch16_224-seed29", "vit_small_patch16_224-seed43",
    ]
    assert [(pair.left, pair.right) for pair in campaign.model_pair_contrasts] == [
        ("resnet50", "vit_small_patch16_224")
    ]
    assert len(campaign_digest(campaign)) == 64


def test_campaign_batch_cap_does_not_change_matching_legacy_resnet_hash() -> None:
    cfg, campaign = _loaded()
    selected = campaign_runs(campaign, cfg)["resnet50-seed17"]
    legacy = materialize_run(cfg, "resnet50", 17, 1, 32)
    assert config_digest(selected) == config_digest(legacy)


def test_protocol_v5_matrix_comes_from_campaign_not_five_model_config() -> None:
    cfg, campaign = _loaded()
    protocol = {
        "schema_version": "sac-experiment-protocol-v5",
        "campaign": {
            "active_models": list(campaign.active_models),
            "seeds": list(campaign.seeds),
            "model_pair_contrasts": [
                {"left": pair.left, "right": pair.right} for pair in campaign.model_pair_contrasts
            ],
        },
    }
    models, seeds, pairs = protocol_matrix(protocol, cfg)
    assert models == campaign.active_models
    assert seeds == campaign.seeds
    assert pairs == campaign.model_pair_contrasts


def test_campaign_rejects_any_post_base_override() -> None:
    cfg = load_config(CORE, [*read_campaign_base_overrides(CAMPAIGN), "data.num_workers=7"])
    with pytest.raises(ValueError, match="base_config_sha256"):
        load_campaign(CAMPAIGN, cfg)


def test_campaign_cli_surface_includes_stages_and_mixed_preflights() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "run-all", "--campaign", str(CAMPAIGN), "--stage", "finalize",
        "--pair-manifest", "validation=v", "--pair-manifest", "calibration=c", "--pair-manifest", "final_test=t",
        "--review-freeze", "review.json",
        "--training-preflight", "resnet50=h100.json",
        "--training-preflight", "vit_small_patch16_224=a100.json",
        "--inference-preflight", "a100.json",
    ])
    assert args.stage == "finalize"
    assert len(args.training_preflight) == 2


def test_campaign_device_gate_requires_a100_20gb_and_bf16() -> None:
    valid = {"device": {"name": "NVIDIA A100-SXM4-40GB MIG 3g.20gb", "total_memory_bytes": 20 * 1024**3}, "precision": "bf16"}
    _validate_campaign_device(valid, "vit_small_patch16_224")
    with pytest.raises(ValueError, match="A100 20 GB MIG"):
        _validate_campaign_device({**valid, "device": {**valid["device"], "name": "NVIDIA H100"}}, "vit_small_patch16_224")
    with pytest.raises(ValueError, match="BF16"):
        _validate_campaign_device({**valid, "precision": "fp16"}, "vit_small_patch16_224")


def test_physical_workspace_relocation_preserves_frozen_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    physical_root = Path.cwd() / "test-relocation-root"
    monkeypatch.setenv("SAC_QUTAB_LOGICAL_ROOT", "/workspace")
    monkeypatch.setenv("SAC_QUTAB_PHYSICAL_ROOT", str(physical_root))
    cfg, campaign = _loaded()
    assert cfg.project.output_dir == physical_root / "artifacts"
    assert cfg.data.root == physical_root / "data" / "eurosat"
    assert runtime_path("/workspace/weights/model.safetensors") == physical_root / "weights" / "model.safetensors"
    assert logical_path(physical_root / "artifacts") == Path("/workspace/artifacts")
    assert config_digest(cfg) == campaign.base_config_sha256 == "cc73206d4ff490bebaa9440af5689decd4a93d08038df486dadd8dfe0deb449c"
    assert config_digest(campaign_runs(campaign, cfg)["resnet50-seed17"]) == "3980d9a86e0cba90b454dc632b305cc16190314301cddeee9a16392c0d200368"
