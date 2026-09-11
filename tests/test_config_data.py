from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sac_qutab.config import ConfigError, load_config, materialize_run
from sac_qutab.data import build_manifest, create_splits, load_manifest, validate_split_contract
from sac_qutab.runtime import atomic_write_json


CONFIG = Path("configs/core.yaml")


def test_strict_config_and_global_batch() -> None:
    cfg = load_config(CONFIG)
    one = materialize_run(cfg, "resnet50", 17, 1)
    two = materialize_run(cfg, "resnet50", 17, 2)
    reduced = materialize_run(cfg, "resnet50", 17, 2, 16)
    assert len(cfg.experiment.model_presets) == 5
    assert len(cfg.statistics.model_pair_contrasts) == 10
    assert len({preset.checkpoint_sha256 for preset in cfg.experiment.model_presets}) == 5
    assert (one.batch_size_per_device, one.grad_accum_steps) == (32, 2)
    assert (two.batch_size_per_device, two.grad_accum_steps) == (32, 1)
    assert (reduced.batch_size_per_device, reduced.grad_accum_steps) == (16, 2)
    assert one.effective_global_batch == two.effective_global_batch == reduced.effective_global_batch == 64
    with pytest.raises(ConfigError, match="unknown override"):
        load_config(CONFIG, ["train.unused=1"])
    with pytest.raises(ConfigError, match="optional"):
        load_config(CONFIG, ["scope.mae=true"])
    with pytest.raises(ConfigError, match="every unique model pair"):
        load_config(CONFIG, ["statistics.model_pair_contrasts=[{left: resnet50, right: convnext_tiny}]"])


def _images(root: Path, class_names: tuple[str, ...], count: int = 8) -> None:
    rng = np.random.default_rng(4)
    for class_index, name in enumerate(class_names):
        folder = root / name
        folder.mkdir(parents=True)
        for index in range(count):
            array = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            array[index % 64, :, class_index % 3] = 255
            Image.fromarray(array).save(folder / f"{index}.png")


def test_manifest_split_determinism_and_mutation_guard(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "artifacts"
    cfg = load_config(CONFIG, [f"data.root={json.dumps(str(data_root))}", f"project.output_dir={json.dumps(str(output))}", "data.near_duplicate_hamming_threshold=0"])
    _images(data_root, cfg.data.class_names)
    build_manifest(cfg, allow_count_mismatch=True)
    create_splits(cfg)
    first_bytes = cfg.output_path(cfg.data.split_relpath).read_bytes()
    create_splits(cfg)
    assert first_bytes == cfg.output_path(cfg.data.split_relpath).read_bytes()
    samples, split = validate_split_contract(
        cfg, cfg.output_path(cfg.data.manifest_relpath), cfg.output_path(cfg.data.split_relpath),
    )
    assert len(samples) == 80 and sum(split["counts"].values()) == 80
    manifest_path = cfg.output_path(cfg.data.manifest_relpath)
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_split_contract(cfg, manifest_path, cfg.output_path(cfg.data.split_relpath))


def test_exact_duplicate_groups_cannot_cross_split(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "artifacts"
    cfg = load_config(CONFIG, [f"data.root={json.dumps(str(data_root))}", f"project.output_dir={json.dumps(str(output))}", "data.near_duplicate_hamming_threshold=0"])
    _images(data_root, cfg.data.class_names)
    source = data_root / cfg.data.class_names[0] / "0.png"
    duplicate = data_root / cfg.data.class_names[1] / "duplicate.png"
    duplicate.write_bytes(source.read_bytes())
    build_manifest(cfg, allow_count_mismatch=True)
    samples = load_manifest(cfg.output_path(cfg.data.manifest_relpath))
    duplicate_group = [s for s in samples if s.sha256 == samples[[x.relative_path for x in samples].index(f"{cfg.data.class_names[0]}/0.png")].sha256]
    assert len({s.group_id for s in duplicate_group}) == 1
    split = create_splits(cfg)
    assert len({split["sample_roles"][s.sample_id] for s in duplicate_group}) == 1


def test_split_seed_changes_equal_size_group_tie_order(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    base_overrides = [f"data.root={json.dumps(str(data_root))}", "data.near_duplicate_hamming_threshold=0"]
    first_cfg = load_config(CONFIG, [*base_overrides, f"project.output_dir={json.dumps(str(tmp_path / 'first'))}", "data.split_seed=101"])
    second_cfg = load_config(CONFIG, [*base_overrides, f"project.output_dir={json.dumps(str(tmp_path / 'second'))}", "data.split_seed=202"])
    _images(data_root, first_cfg.data.class_names)
    build_manifest(first_cfg, allow_count_mismatch=True)
    build_manifest(second_cfg, allow_count_mismatch=True)
    first = create_splits(first_cfg)
    second = create_splits(second_cfg)
    assert first["sample_roles"] != second["sample_roles"]


def test_split_contract_rejects_seed_fraction_role_and_count_mutations(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "artifacts"
    cfg = load_config(CONFIG, [f"data.root={json.dumps(str(data_root))}", f"project.output_dir={json.dumps(str(output))}", "data.near_duplicate_hamming_threshold=0"])
    _images(data_root, cfg.data.class_names)
    build_manifest(cfg, allow_count_mismatch=True)
    path = cfg.output_path(cfg.data.split_relpath)
    create_splits(cfg)
    original = json.loads(path.read_text(encoding="utf-8"))
    mutations = []
    changed = json.loads(json.dumps(original)); changed["seed"] += 1; mutations.append(changed)
    changed = json.loads(json.dumps(original)); changed["fractions"]["train"] -= 0.01; changed["fractions"]["validation"] += 0.01; mutations.append(changed)
    changed = json.loads(json.dumps(original)); sample_id = next(iter(changed["sample_roles"])); changed["sample_roles"][sample_id] = "train" if changed["sample_roles"][sample_id] != "train" else "validation"; mutations.append(changed)
    changed = json.loads(json.dumps(original)); changed["counts"]["train"] += 1; mutations.append(changed)
    for mutation in mutations:
        atomic_write_json(path, mutation)
        with pytest.raises(ValueError):
            validate_split_contract(cfg, cfg.output_path(cfg.data.manifest_relpath), path)
    atomic_write_json(path, original)
    validate_split_contract(cfg, cfg.output_path(cfg.data.manifest_relpath), path)
