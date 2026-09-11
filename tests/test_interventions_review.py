from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sac_qutab.cli import _fill_synthetic_reviews
from sac_qutab.config import Config, load_config
from sac_qutab.data import build_manifest, create_splits, load_manifest
from sac_qutab.interventions import apply_intervention, build_pair_manifest, materialize_pair_cache, validate_severity_support
from sac_qutab.review import apply_review_freeze, create_no_human_review_policy, export_review, freeze_review, validate_review_freeze_contract
from sac_qutab.runtime import read_jsonl, write_jsonl

CONFIG = Path("configs/core.yaml")


def test_transform_formulas_and_nested_determinism() -> None:
    source = torch.linspace(0, 1, 3 * 64 * 64).reshape(3, 64, 64)
    donor = source.flip(-1)
    spectral = apply_intervention(source, "spectral_appearance", {"gains": [1.05, 0.95, 1.0]})
    assert torch.allclose(spectral[2], source[2])
    texture_sham = apply_intervention(source, "texture", {"lambda": 0.0}, donor)
    assert torch.allclose(texture_sham, source, atol=2e-6)
    mild = apply_intervention(source, "resolution", {"internal_size": 48, "restore_size": 64})
    severe = apply_intervention(source, "resolution", {"internal_size": 16, "restore_size": 64})
    assert mild.shape == severe.shape == source.shape
    assert torch.mean(torch.abs(severe - source)) > torch.mean(torch.abs(mild - source))


def _severity_rows() -> list[dict]:
    rows = []
    for family in ("spectral_appearance", "texture", "resolution"):
        for severity, radius in zip(("sham", "mild", "moderate", "severe"), (0.0, 0.05, 0.1, 0.2), strict=True):
            rows.append({
                "source_id": "source", "family": family, "severity": severity, "latent_seed": 4,
                "donor_id": "donor" if family == "texture" else None,
                "parameters": {"gains": [1 + radius, 1 - radius, 1]} if family == "spectral_appearance" else {},
            })
    return rows


def test_severity_manifest_invariants() -> None:
    rows = _severity_rows()
    validate_severity_support(rows)
    changed = [dict(row) for row in rows]
    changed[6] = {**changed[6], "donor_id": "other"}
    with pytest.raises(ValueError, match="donor"):
        validate_severity_support(changed)


def _make_images(cfg: Config) -> None:
    rng = np.random.default_rng(17)
    for class_index, class_name in enumerate(cfg.data.class_names):
        folder = cfg.data.root / class_name
        folder.mkdir(parents=True)
        for index in range(10):
            image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            image[:, :, class_index % 3] = np.clip(image[:, :, class_index % 3].astype(int) + class_index * 5, 0, 255)
            Image.fromarray(image).save(folder / f"{class_index:02d}_{index:03d}.png")


@pytest.fixture(scope="module")
def review_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, Path, Path, Path]:
    root = tmp_path_factory.mktemp("semantic-review")
    cfg = load_config(CONFIG, [
        f"project.output_dir={json.dumps(str(root / 'artifacts'))}", f"data.root={json.dumps(str(root / 'data'))}",
        "data.split_fractions.train=0.20", "data.split_fractions.validation=0.60",
        "data.split_fractions.calibration=0.10", "data.split_fractions.final_test=0.10",
        "audit.samples_per_class_condition=5", "audit.pilot_rows=30",
    ])
    _make_images(cfg)
    build_manifest(cfg, allow_count_mismatch=True)
    split = create_splits(cfg)
    pair_path = cfg.project.output_dir / "counterfactuals" / "validation_pairs.jsonl"
    build_pair_manifest(cfg, load_manifest(cfg.output_path(cfg.data.manifest_relpath)), split, "validation", pair_path)
    materialize_pair_cache(cfg, pair_path)
    review_dir = cfg.project.output_dir / "counterfactuals" / "review"
    export_review(cfg, pair_path, review_dir)
    attestation = _fill_synthetic_reviews(review_dir)
    return cfg, review_dir, pair_path, attestation


def test_review_freeze_whole_condition_rules(review_fixture: tuple[Config, Path, Path, Path], tmp_path: Path) -> None:
    cfg, review_dir, pair_path, attestation = review_fixture
    output = tmp_path / "freeze.json"
    artifact = freeze_review(cfg, review_dir, pair_path, output, attestation)
    validate_review_freeze_contract(cfg, artifact, validation_pair_manifest=pair_path)
    assert artifact["cohen_kappa"] == 1.0
    assert len(artifact["included_conditions"]) == 9
    assert Path(artifact["qualitative_panel_path"]).is_file()
    pair_rows = [
        {"severity": "mild", "audit_condition": "resolution:mild", "condition_signature": artifact["condition_signature"]},
        {"severity": "mild", "audit_condition": "unknown:mild", "condition_signature": artifact["condition_signature"]},
        {"severity": "sham", "audit_condition": "unknown:sham", "condition_signature": artifact["condition_signature"]},
    ]
    assert len(apply_review_freeze(pair_rows, artifact)) == 2
    changed = json.loads(json.dumps(artifact))
    changed["rules"]["min_kappa"] = 0.0
    with pytest.raises(ValueError, match="rules"):
        validate_review_freeze_contract(cfg, changed)
    changed = json.loads(json.dumps(artifact))
    first_condition = next(iter(changed["conditions"]))
    changed["conditions"][first_condition]["eligible"] = False
    with pytest.raises(ValueError, match="eligibility"):
        validate_review_freeze_contract(cfg, changed)


def test_no_human_review_policy_keeps_all_predeclared_conditions(
    review_fixture: tuple[Config, Path, Path, Path], tmp_path: Path,
) -> None:
    cfg, _, pair_path, _ = review_fixture
    output = tmp_path / "condition-policy.json"
    artifact = create_no_human_review_policy(cfg, pair_path, output)
    validate_review_freeze_contract(cfg, artifact, validation_pair_manifest=pair_path)
    assert artifact["human_semantic_review_performed"] is False
    assert artifact["synthetic_pipeline_qa_is_scientific_evidence"] is False
    assert len(artifact["included_conditions"]) == 9
    rows = [
        {"severity": "severe", "audit_condition": "texture:severe", "condition_signature": artifact["condition_signature"]},
        {"severity": "sham", "audit_condition": "texture:sham", "condition_signature": artifact["condition_signature"]},
    ]
    assert apply_review_freeze(rows, artifact) == rows


def test_review_disagreement_requires_adjudication(review_fixture: tuple[Config, Path, Path, Path], tmp_path: Path) -> None:
    cfg, review_dir, pair_path, attestation = review_fixture
    changed = tmp_path / "changed-review"
    shutil.copytree(review_dir, changed)
    reviewer_path = changed / "final_reviewer_2.csv"
    with reviewer_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["decision"] = "reject" if rows[0]["decision"] == "accept" else "accept"
    with reviewer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    adjudication_path = changed / "adjudication.csv"
    with adjudication_path.open(newline="", encoding="utf-8") as handle:
        adjudication = list(csv.DictReader(handle))
    target = rows[0]["review_id"]
    for row in adjudication:
        if row["review_id"] == target:
            row["decision"] = ""
    with adjudication_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=adjudication[0].keys())
        writer.writeheader()
        writer.writerows(adjudication)
    with pytest.raises(ValueError, match="adjudication"):
        freeze_review(cfg, changed, pair_path, tmp_path / "freeze.json", attestation)


def test_review_key_mapping_tamper_is_rejected(review_fixture: tuple[Config, Path, Path, Path], tmp_path: Path) -> None:
    cfg, review_dir, pair_path, attestation = review_fixture
    changed = tmp_path / "tampered-review"
    shutil.copytree(review_dir, changed)
    key_path = changed / "review_key.jsonl"
    rows = read_jsonl(key_path)
    rows[0]["family"] = "resolution" if rows[0]["family"] != "resolution" else "texture"
    write_jsonl(key_path, rows)
    with pytest.raises(ValueError, match="contract|mapping"):
        freeze_review(cfg, changed, pair_path, tmp_path / "freeze.json", attestation)


def test_reviewer_display_fields_are_immutable(review_fixture: tuple[Config, Path, Path, Path], tmp_path: Path) -> None:
    cfg, review_dir, pair_path, attestation = review_fixture
    changed = tmp_path / "tampered-display"
    shutil.copytree(review_dir, changed)
    sheet = changed / "final_reviewer_1.csv"
    with sheet.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["intended_class"] = "tampered-class"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="display"):
        freeze_review(cfg, changed, pair_path, tmp_path / "freeze.json", attestation)
