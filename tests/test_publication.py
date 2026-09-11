from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sac_qutab.config import Config, load_config
from sac_qutab.publication import (
    _failure_candidates,
    _macro_cluster_confusion,
    _qualitative_gallery,
    _throughput_pareto,
)
from sac_qutab.runtime import atomic_write_bytes, sha256_bytes, sha256_file, write_jsonl


def _config(tmp_path: Path) -> Config:
    return load_config("configs/core.yaml", [
        f"project.output_dir={json.dumps(str(tmp_path / 'artifacts'))}",
        f"data.root={json.dumps(str(tmp_path / 'data'))}",
    ])


def _failure_row(cfg: Config, *, model: str = "resnet50", source_id: str = "source-1", confidence: float = .995) -> dict:
    clean = [0.0] * len(cfg.data.class_names)
    transformed = [0.0] * len(cfg.data.class_names)
    clean[cfg.data.class_names.index("Forest")] = confidence
    clean[cfg.data.class_names.index("AnnualCrop")] = 1 - confidence
    transformed[cfg.data.class_names.index("Highway")] = .9
    transformed[cfg.data.class_names.index("Forest")] = .1
    return {
        "run": f"{model}-seed17",
        "model": model,
        "seed": 17,
        "pair_id": f"{source_id}:spectral_appearance:mild",
        "source_id": source_id,
        "source_group_id": f"group:{source_id}",
        "family": "spectral_appearance",
        "severity": "mild",
        "class_name": "Forest",
        "label": cfg.data.class_names.index("Forest"),
        "audit_qualified": True,
        "original_probabilities": clean,
        "transformed_probabilities": transformed,
        "label_consistency": 0.0,
        "cosine_similarity": .7,
    }


def test_failure_selection_is_frozen_and_model_specific(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    rows = [
        _failure_row(cfg, source_id=f"resnet-{index}", confidence=.999 - index * .001)
        for index in range(4)
    ]
    rows.append(_failure_row(cfg, model="vit_small_patch16_224", source_id="vit", confidence=.991))
    assert len(_failure_candidates(rows)) == 4
    selected = _failure_candidates(rows, limit=1, model="vit_small_patch16_224")
    assert [row["source_id"] for row in selected] == ["vit"]


def test_gallery_and_macro_confusion_use_authenticated_real_images(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    output = tmp_path / "publication"
    output.mkdir()

    image_path = cfg.data.root / "Forest" / "sample.png"
    image_path.parent.mkdir(parents=True)
    image = np.zeros((cfg.data.native_size, cfg.data.native_size, 3), dtype=np.uint8)
    image[..., 1] = 180
    Image.fromarray(image).save(image_path)
    write_jsonl(cfg.output_path(cfg.data.manifest_relpath), [{
        "sample_id": "source-1",
        "relative_path": "Forest/sample.png",
        "label": cfg.data.class_names.index("Forest"),
        "class_name": "Forest",
        "width": cfg.data.native_size,
        "height": cfg.data.native_size,
        "mode": "RGB",
        "bytes": image_path.stat().st_size,
        "sha256": sha256_file(image_path),
        "dhash": "0" * 16,
        "group_id": "group:source-1",
    }])

    transformed = np.zeros((3, cfg.data.native_size, cfg.data.native_size), dtype=np.float32)
    transformed[0] = .8
    buffer = io.BytesIO()
    np.save(buffer, transformed, allow_pickle=False)
    payload = buffer.getvalue()
    cache_relpath = "counterfactuals/cache/pair.npy"
    atomic_write_bytes(cfg.project.output_dir / cache_relpath, payload)
    row = {
        **_failure_row(cfg),
        "cache_relpath": cache_relpath,
        "cache_sha256": sha256_bytes(payload),
    }

    selected = _qualitative_gallery(cfg, [row], output)
    _macro_cluster_confusion(cfg, [row], output)

    assert len(selected) == 1
    gallery = json.loads((output / "qualitative_failure_gallery.json").read_text(encoding="utf-8"))
    assert gallery["selected_count"] == 1
    assert (output / "qualitative_failure_gallery.png").is_file()
    with (output / "macro_cluster_confusion.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["true_macro_cluster"] == "natural"
    assert rows[0]["predicted_macro_cluster"] == "anthropogenic"
    assert rows[0]["error_scope"] == "cross_macro"
    mapping = json.loads((output / "macro_cluster_mapping.json").read_text(encoding="utf-8"))["mapping"]
    assert mapping["PermanentCrop"] == "anthropogenic"


def test_throughput_pareto_requires_complete_cuda_measurements(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    output = tmp_path / "publication"
    output.mkdir()
    provenance = []
    pairs = []
    for model_index, preset in enumerate(cfg.experiment.model_presets):
        for seed in cfg.experiment.seeds:
            provenance.append({
                "model_name": preset.name,
                "training_seed": seed,
                "inference": {"device_type": "cuda", "forward_images_per_second": 100.0 + model_index},
            })
        pairs.append({
            "model": preset.name,
            "severity": "mild",
            "audit_qualified": True,
            "label_consistency": .5 + model_index * .1,
        })

    _throughput_pareto(cfg, provenance, pairs, output)
    with (output / "throughput_vs_robustness.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frontier = [row["model"] for row in rows if row["pareto_frontier"] == "True"]
    assert frontier == [cfg.experiment.model_presets[-1].name]

    incomplete = provenance[:-1]
    _throughput_pareto(cfg, incomplete, pairs, output)
    with (output / "throughput_vs_robustness.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["defined"] == "False"
