from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .campaign import protocol_matrix
from .config import Config, canonical_bytes
from .data import load_manifest, load_native_image, preprocess_image
from .interpretability import attention_rollout
from .interventions import load_cached_tensor
from .metrics import linear_cka
from .models import build_model
from .runtime import atomic_write_bytes, atomic_write_json, read_jsonl, sha256_file


_DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-Small/16",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
    "dinov2_vits14": "DINOv2-Small",
}
_MACRO_CLUSTER = {
    "Forest": "natural",
    "River": "natural",
    "SeaLake": "natural",
    "HerbaceousVegetation": "natural",
    "Pasture": "natural",
    "Highway": "anthropogenic",
    "Residential": "anthropogenic",
    "Industrial": "anthropogenic",
    "AnnualCrop": "anthropogenic",
    "PermanentCrop": "anthropogenic",
}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty publication table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _training_contracts(
    cfg: Config, training_run_dirs: Sequence[str | Path], protocol: Mapping[str, Any],
) -> tuple[dict[str, Path], list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {f"{preset.name}-seed{seed}" for preset in cfg.experiment.model_presets for seed in cfg.experiment.seeds}
    paths = {Path(value).name: Path(value) for value in training_run_dirs}
    if len(paths) != len(training_run_dirs) or set(paths) != expected:
        raise ValueError(f"publication requires the exact {len(expected)} configured training-run directories")
    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for run_name, directory in sorted(paths.items()):
        event_path = directory / "events.jsonl"
        model_path = directory / "model.json"
        resolved_path = directory / "resolved_run.json"
        if any(not path.is_file() for path in (event_path, model_path, resolved_path)):
            raise FileNotFoundError(f"training publication input is incomplete: {directory}")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(canonical_bytes(resolved)).hexdigest()
        if digest != protocol["resolved_run_sha256"][run_name]:
            raise ValueError(f"training resolved-run contract differs from protocol: {run_name}")
        model_name, seed_text = run_name.rsplit("-seed", 1)
        metadata = json.loads(model_path.read_text(encoding="utf-8"))
        preset = next(item for item in cfg.experiment.model_presets if item.name == model_name)
        if (metadata.get("name") != model_name
                or metadata.get("checkpoint_sha256") != preset.checkpoint_sha256
                or metadata.get("adapter_version") != preset.adapter_version
                or metadata.get("expected_representation_dim") != preset.expected_representation_dim
                or not isinstance(metadata.get("parameter_count"), int)):
            raise ValueError(f"training model metadata differs from configured contract: {run_name}")
        rows = read_jsonl(event_path)
        if not rows or not any(row.get("event") == "validation" and not row.get("bounded") for row in rows):
            raise ValueError(f"training events omit completed validation epochs: {run_name}")
        for row in rows:
            events.append({**row, "run": run_name, "model": model_name, "seed": int(seed_text)})
        sources.append({
            "run": run_name, "directory": str(directory),
            "events_sha256": sha256_file(event_path),
            "model_sha256": sha256_file(model_path),
            "resolved_run_file_sha256": sha256_file(resolved_path),
            "resolved_run_contract_sha256": digest,
            "parameter_count": metadata["parameter_count"],
        })
    return paths, events, sources


def _learning_curves(
    cfg: Config, events: Sequence[dict[str, Any]], output: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_run[str(event["run"])].append(event)
    for run_name, run_events in sorted(by_run.items()):
        model, seed_text = run_name.rsplit("-seed", 1)
        train_by_epoch: dict[int, list[float]] = defaultdict(list)
        validations: dict[int, dict[str, Any]] = {}
        for event in run_events:
            epoch = int(event["epoch"])
            if event["event"] == "optimizer_step":
                train_by_epoch[epoch].append(float(event["loss"]))
            elif event["event"] == "validation" and not event.get("bounded"):
                if epoch in validations:
                    raise ValueError(f"duplicate validation epoch in {run_name}: {epoch}")
                validations[epoch] = event
        if not validations or any(epoch not in train_by_epoch for epoch in validations):
            raise ValueError(f"training/validation epoch alignment is incomplete: {run_name}")
        for epoch, validation in sorted(validations.items()):
            rows.append({
                "run": run_name, "model": model, "seed": int(seed_text), "epoch": epoch + 1,
                "train_loss": float(np.mean(train_by_epoch[epoch])),
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_macro_f1": float(validation["macro_f1"]),
                "best": bool(validation.get("best", False)),
            })
    _write_csv(output / "learning_curves.csv", rows)
    figure, axes = plt.subplots(len(cfg.experiment.model_presets), 2, figsize=(12, 3.2 * len(cfg.experiment.model_presets)), constrained_layout=True)
    for row_index, preset in enumerate(cfg.experiment.model_presets):
        for seed in cfg.experiment.seeds:
            selected = sorted(
                (row for row in rows if row["model"] == preset.name and row["seed"] == seed),
                key=lambda row: row["epoch"],
            )
            epochs = [row["epoch"] for row in selected]
            axes[row_index, 0].plot(epochs, [row["train_loss"] for row in selected], label=f"train s{seed}")
            axes[row_index, 0].plot(epochs, [row["validation_loss"] for row in selected], linestyle="--", label=f"val s{seed}")
            axes[row_index, 1].plot(epochs, [row["validation_accuracy"] for row in selected], label=f"acc s{seed}")
            axes[row_index, 1].plot(epochs, [row["validation_macro_f1"] for row in selected], linestyle="--", label=f"F1 s{seed}")
        axes[row_index, 0].set(title=f"{_DISPLAY_NAMES[preset.name]} loss", xlabel="Epoch", ylabel="Loss")
        axes[row_index, 1].set(title=f"{_DISPLAY_NAMES[preset.name]} validation", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
        axes[row_index, 0].legend(fontsize=7, ncol=2)
        axes[row_index, 1].legend(fontsize=7, ncol=2)
    figure.savefig(output / "learning_curves.png", dpi=180)
    plt.close(figure)
    return rows


def _master_summary(
    cfg: Config,
    clean_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    statistics: Mapping[str, Any],
    training_sources: Sequence[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    parameters: dict[str, set[int]] = defaultdict(set)
    for source in training_sources:
        model = str(source["run"]).rsplit("-seed", 1)[0]
        parameters[model].add(int(source["parameter_count"]))
    detector = {
        row["model"]: row
        for row in statistics["primary_seed_summaries"]
        if row["contrast"] == "rq3_nested_detector_delta_auroc" and row["family"] == "all_accepted_non_sham"
    }
    result: list[dict[str, Any]] = []
    for preset in cfg.experiment.model_presets:
        model_clean = [row for row in clean_rows if row["model"] == preset.name]
        model_pairs = [
            row for row in pair_rows
            if row["model"] == preset.name and row["severity"] != "sham" and row.get("audit_qualified") is True
        ]
        if len(model_clean) != len(cfg.experiment.seeds) or not model_pairs or len(parameters[preset.name]) != 1:
            raise ValueError(f"master summary inputs are incomplete for {preset.name}")
        by_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in model_pairs:
            by_condition[(row["family"], row["severity"])].append(float(row["label_consistency"]))
        worst_condition, minimum_stability = min(
            ((f"{key[0]}:{key[1]}", float(np.mean(values))) for key, values in by_condition.items()),
            key=lambda item: item[1],
        )
        delta = detector.get(preset.name)
        result.append({
            "model": _DISPLAY_NAMES[preset.name],
            "model_id": preset.name,
            "params_m": next(iter(parameters[preset.name])) / 1_000_000,
            "clean_accuracy_percent": 100 * float(np.mean([row["accuracy"] for row in model_clean])),
            "clean_ece": float(np.mean([row["ece"] for row in model_clean])),
            "min_s_label": minimum_stability,
            "min_s_label_condition": worst_condition,
            "mean_cosine_similarity": float(np.mean([row["cosine_similarity"] for row in model_pairs])),
            "detector_delta_auroc": delta.get("mean") if delta and delta.get("defined") else None,
            "seed_count": len(cfg.experiment.seeds),
        })
    _write_csv(output / "master_summary.csv", result)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Model & Params (M) & Clean Acc. (\%) & Clean ECE & Min $S_{label}$ & Mean Cosine & $\Delta$AUROC \\",
        r"\hline",
    ]
    for row in result:
        delta = "--" if row["detector_delta_auroc"] is None else f"{row['detector_delta_auroc']:.3f}"
        lines.append(
            f"{row['model']} & {row['params_m']:.1f} & {row['clean_accuracy_percent']:.2f} & "
            f"{row['clean_ece']:.3f} & {row['min_s_label']:.3f} & "
            f"{row['mean_cosine_similarity']:.3f} & {delta} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{SAC-QUTAB campaign summary. Values aggregate the frozen training seeds.}",
        r"\label{tab:master-summary}",
        r"\end{table*}",
    ])
    atomic_write_bytes(output / "master_summary.tex", ("\n".join(lines) + "\n").encode("utf-8"))
    return result


def _pairwise_forest(cfg: Config, statistics: Mapping[str, Any], output: Path) -> None:
    rows = list(statistics.get("pairwise_model_summaries", []))
    expected = [(pair.left, pair.right) for pair in cfg.statistics.model_pair_contrasts]
    if [(row.get("left_model"), row.get("right_model")) for row in rows] != expected:
        raise ValueError("forest plot requires the exact campaign pairwise summaries")
    table: list[dict[str, Any]] = []
    for row in rows:
        lower, upper = map(float, row["ci"])
        significant = bool(row.get("holm_rejected")) and (lower > 0 or upper < 0)
        table.append({
            **row, "ci_lower": lower, "ci_upper": upper,
            "significance_marker": "*" if significant else "",
        })
    _write_csv(output / "pairwise_forest_plot.csv", table)
    figure, axis = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    y = np.arange(len(table))
    effects = np.asarray([row["effect"] for row in table], dtype=float)
    lower = np.asarray([row["ci_lower"] for row in table], dtype=float)
    upper = np.asarray([row["ci_upper"] for row in table], dtype=float)
    axis.errorbar(effects, y, xerr=np.vstack((effects - lower, upper - effects)), fmt="o", capsize=3)
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    labels = [
        f"{_DISPLAY_NAMES[row['right_model']]} − {_DISPLAY_NAMES[row['left_model']]} {row['significance_marker']}"
        for row in table
    ]
    axis.set(yticks=y, yticklabels=labels, xlabel="Difference-in-differences (right − left)", title="Pairwise representation effects (95% clustered bootstrap CI)")
    axis.invert_yaxis()
    figure.savefig(output / "pairwise_forest_plot.png", dpi=180)
    plt.close(figure)


def _cross_model_cka(
    cfg: Config, evaluation_dirs: Sequence[Path], provenance: Sequence[dict[str, Any]], output: Path,
) -> None:
    by_run: dict[str, tuple[list[str], np.ndarray]] = {}
    for directory, record in zip(evaluation_dirs, provenance, strict=True):
        run = f"{record['model_name']}-seed{record['training_seed']}"
        pair_rows = read_jsonl(directory / "pair_metrics.jsonl")
        payload = np.load(directory / "representations.npz", allow_pickle=False)
        original = np.asarray(payload["original"], dtype=float)
        if len(pair_rows) != len(original):
            raise ValueError(f"representation rows do not match pair rows: {run}")
        first: dict[str, int] = {}
        for index, row in enumerate(pair_rows):
            first.setdefault(str(row["source_id"]), index)
        source_ids = sorted(first)
        by_run[run] = (source_ids, original[[first[source_id] for source_id in source_ids]])
    names = [preset.name for preset in cfg.experiment.model_presets]
    per_pair: list[dict[str, Any]] = []
    matrix = np.eye(len(names), dtype=float)
    for left_index, left in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right = names[right_index]
            values: list[float] = []
            reasons: list[str] = []
            for seed in cfg.experiment.seeds:
                left_ids, left_values = by_run[f"{left}-seed{seed}"]
                right_ids, right_values = by_run[f"{right}-seed{seed}"]
                if left_ids != right_ids:
                    raise ValueError("cross-model CKA requires identical clean source support")
                try:
                    values.append(float(linear_cka(left_values, right_values)))
                except ValueError as exc:
                    reasons.append(str(exc))
            defined = len(values) == len(cfg.experiment.seeds)
            mean = float(np.mean(values)) if defined else None
            if defined:
                matrix[left_index, right_index] = matrix[right_index, left_index] = mean
            else:
                matrix[left_index, right_index] = matrix[right_index, left_index] = np.nan
            per_pair.append({
                "left_model": left, "right_model": right, "defined": defined,
                "mean_linear_cka": mean, "per_seed": json.dumps(values),
                "reason": None if defined else "; ".join(reasons),
                "sample_basis": "one_clean_representation_per_source_id_and_matching_training_seed",
            })
    _write_csv(output / "cross_model_cka_matrix.csv", per_pair)
    figure, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    labels = [_DISPLAY_NAMES[name] for name in names]
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=labels, yticklabels=labels, title="Cross-model clean representation linear CKA")
    axis.tick_params(axis="x", rotation=35)
    for row in range(len(names)):
        for column in range(len(names)):
            text = "NA" if np.isnan(matrix[row, column]) else f"{matrix[row, column]:.2f}"
            axis.text(column, row, text, ha="center", va="center", color="white" if not np.isnan(matrix[row, column]) and matrix[row, column] < .55 else "black")
    figure.colorbar(image, ax=axis)
    figure.savefig(output / "cross_model_cka_matrix.png", dpi=180)
    plt.close(figure)


def _throughput_pareto(
    cfg: Config,
    provenance: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    records: list[dict[str, Any]] = []
    for preset in cfg.experiment.model_presets:
        timings = [
            float(row["inference"]["forward_images_per_second"])
            for row in provenance
            if row["model_name"] == preset.name
            and row.get("inference", {}).get("device_type") == "cuda"
            and isinstance(row.get("inference", {}).get("forward_images_per_second"), (int, float))
        ]
        stability_values = [
            float(row["label_consistency"]) for row in pair_rows
            if row["model"] == preset.name and row["severity"] != "sham" and row.get("audit_qualified") is True
        ]
        defined = len(timings) == len(cfg.experiment.seeds) and bool(stability_values)
        records.append({
            "model": preset.name, "display_name": _DISPLAY_NAMES[preset.name], "defined": defined,
            "forward_images_per_second": float(np.mean(timings)) if defined else None,
            "mean_s_label": float(np.mean(stability_values)) if stability_values else None,
            "timing_scope": "model_forward_only_after_device_transfer",
            "claim_boundary": "common final-inference hardware only; training throughput is not compared across architectures",
            "timing_seed_count": len(timings),
        })
    for row in records:
        if not row["defined"]:
            row["pareto_frontier"] = False
            continue
        row["pareto_frontier"] = not any(
            other["defined"]
            and other["forward_images_per_second"] >= row["forward_images_per_second"]
            and other["mean_s_label"] >= row["mean_s_label"]
            and (
                other["forward_images_per_second"] > row["forward_images_per_second"]
                or other["mean_s_label"] > row["mean_s_label"]
            )
            for other in records
        )
    _write_csv(output / "throughput_vs_robustness.csv", records)
    figure, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    defined_rows = [row for row in records if row["defined"]]
    if defined_rows:
        for row in defined_rows:
            axis.scatter(
                row["forward_images_per_second"], row["mean_s_label"],
                marker="*" if row["pareto_frontier"] else "o", s=110 if row["pareto_frontier"] else 55,
            )
            axis.annotate(row["display_name"], (row["forward_images_per_second"], row["mean_s_label"]), xytext=(5, 5), textcoords="offset points")
        axis.set(xlabel="Common-session A100 model-forward images/s", ylabel="Mean accepted counterfactual $S_{label}$", title="Descriptive final-inference throughput–robustness view")
    else:
        axis.text(.5, .5, "CUDA inference throughput unavailable", ha="center", va="center")
        axis.set_axis_off()
    figure.savefig(output / "throughput_vs_robustness.png", dpi=180)
    plt.close(figure)


def _is_failure(row: Mapping[str, Any]) -> bool:
    label = int(row["label"])
    clean = int(np.argmax(row["original_probabilities"]))
    transformed = int(np.argmax(row["transformed_probabilities"]))
    return clean == label and transformed != label


def _failure_candidates(
    rows: Sequence[dict[str, Any]], *, limit: int | None = 4, model: str | None = None,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if row.get("audit_qualified") is True
        and row.get("severity") == "mild"
        and row.get("family") in {"spectral_appearance", "texture"}
        and max(row["original_probabilities"]) >= .99
        and (model is None or row.get("model") == model)
        and _is_failure(row)
    ]
    eligible.sort(key=lambda row: (-max(row["original_probabilities"]), row["model"], row["source_id"], row["pair_id"]))
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in eligible:
        if row["source_id"] not in seen_sources:
            selected.append(row)
            seen_sources.add(row["source_id"])
        if limit is not None and len(selected) == limit:
            break
    return selected


def _qualitative_gallery(cfg: Config, pair_rows: Sequence[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    selected = _failure_candidates(pair_rows)
    metadata: list[dict[str, Any]] = []
    figure, axes = plt.subplots(max(1, len(selected)), 2, figsize=(8, 3.4 * max(1, len(selected))), squeeze=False, constrained_layout=True)
    if not selected:
        axes[0, 0].text(.5, .5, "No failure met the frozen ≥0.99 / mild spectral-or-texture criterion", ha="center", va="center", wrap=True)
        axes[0, 0].set_axis_off()
        axes[0, 1].set_axis_off()
    else:
        samples = {sample.sample_id: sample for sample in load_manifest(cfg.output_path(cfg.data.manifest_relpath))}
        for index, row in enumerate(selected):
            original = load_native_image(cfg.data.root, samples[row["source_id"]])
            transformed = load_cached_tensor(cfg, row)
            clean_prediction = int(np.argmax(row["original_probabilities"]))
            transformed_prediction = int(np.argmax(row["transformed_probabilities"]))
            for axis, image, title in (
                (axes[index, 0], original, f"Clean: {row['class_name']} → {cfg.data.class_names[clean_prediction]} ({max(row['original_probabilities']):.3f})"),
                (axes[index, 1], transformed, f"{row['family']} mild → {cfg.data.class_names[transformed_prediction]} ({max(row['transformed_probabilities']):.3f})"),
            ):
                axis.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
                axis.set_title(title, fontsize=9)
                axis.set_axis_off()
            metadata.append({
                "run": row["run"], "model": row["model"], "seed": row["seed"],
                "pair_id": row["pair_id"], "source_id": row["source_id"],
                "family": row["family"], "severity": row["severity"],
                "true_class": row["class_name"],
                "clean_prediction": cfg.data.class_names[clean_prediction],
                "transformed_prediction": cfg.data.class_names[transformed_prediction],
                "clean_confidence": max(row["original_probabilities"]),
                "transformed_confidence": max(row["transformed_probabilities"]),
            })
    figure.savefig(output / "qualitative_failure_gallery.png", dpi=180)
    plt.close(figure)
    atomic_write_json(output / "qualitative_failure_gallery.json", {
        "schema_version": "sac-qualitative-failure-gallery-v1",
        "selection": "audit-qualified mild spectral/texture counterfactual failure with clean confidence >= 0.99; deterministic top four unique sources",
        "selected_count": len(metadata), "rows": metadata,
        "empty_is_valid": True,
    })
    return selected


def _load_trained_model(
    cfg: Config, run_name: str, run_dir: Path, protocol: Mapping[str, Any], device: torch.device,
) -> torch.nn.Module:
    model_name = run_name.rsplit("-seed", 1)[0]
    preset = next(item for item in cfg.experiment.model_presets if item.name == model_name)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("schema_version") != "sac-checkpoint-v2"
            or checkpoint.get("resolved_run_sha256") != protocol["resolved_run_sha256"][run_name]
            or sha256_file(checkpoint_path) != protocol["checkpoint_sha256"][run_name]):
        raise ValueError(f"attention checkpoint differs from frozen protocol: {run_name}")
    model = build_model(preset, len(cfg.data.class_names))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def _attention_figure(
    cfg: Config,
    pair_rows: Sequence[dict[str, Any]],
    training_paths: Mapping[str, Path],
    protocol: Mapping[str, Any],
    output: Path,
) -> None:
    attention_models = tuple(
        preset.name for preset in cfg.experiment.model_presets
        if preset.name in {"vit_small_patch16_224", "dinov2_vits14"}
    )
    if not attention_models:
        atomic_write_json(output / "attention_rollout.json", {
            "schema_version": "sac-attention-rollout-v1", "rows": [],
            "claim_boundary": "No attention-rollout-compatible campaign model is active.",
        })
        figure, axis = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        axis.text(.5, .5, "No attention-rollout-compatible campaign model", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(output / "attention_rollout.png", dpi=180)
        plt.close(figure)
        return
    chosen: list[dict[str, Any]] = []
    for model_name in attention_models:
        chosen.extend(_failure_candidates(pair_rows, limit=1, model=model_name))
    metadata: list[dict[str, Any]] = []
    figure, axes = plt.subplots(len(attention_models), 4, figsize=(14, 3.5 * len(attention_models)), squeeze=False, constrained_layout=True)
    samples = None
    for row_index, model_name in enumerate(attention_models):
        row = next((candidate for candidate in chosen if candidate["model"] == model_name), None)
        if row is None:
            axes[row_index, 0].text(.5, .5, f"{_DISPLAY_NAMES[model_name]}: no eligible failure", ha="center", va="center")
            for axis in axes[row_index]:
                axis.set_axis_off()
            metadata.append({"model": model_name, "defined": False, "reason": "no gallery-eligible failure"})
            continue
        if samples is None:
            samples = {sample.sample_id: sample for sample in load_manifest(cfg.output_path(cfg.data.manifest_relpath))}
        original = load_native_image(cfg.data.root, samples[row["source_id"]])
        transformed = load_cached_tensor(cfg, row)
        device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device != "cpu" else "cpu")
        model = _load_trained_model(cfg, row["run"], training_paths[row["run"]], protocol, device)
        batch = torch.stack([
            preprocess_image(original, cfg.data.input_size),
            preprocess_image(transformed, cfg.data.input_size),
        ]).to(device)
        heat = attention_rollout(model, batch)
        raw_images = [original.permute(1, 2, 0).clamp(0, 1).numpy(), transformed.permute(1, 2, 0).clamp(0, 1).numpy()]
        titles = ("Clean", f"{row['family']} mild")
        for item_index, (raw, title) in enumerate(zip(raw_images, titles, strict=True)):
            plain_axis = axes[row_index, item_index * 2]
            overlay_axis = axes[row_index, item_index * 2 + 1]
            plain_axis.imshow(raw)
            plain_axis.set_title(f"{_DISPLAY_NAMES[model_name]} {title}")
            overlay_axis.imshow(raw)
            overlay_axis.imshow(heat[item_index], cmap="jet", alpha=.45, vmin=0, vmax=1)
            overlay_axis.set_title(f"{title} attention rollout")
            plain_axis.set_axis_off()
            overlay_axis.set_axis_off()
        metadata.append({
            "model": model_name, "defined": True, "run": row["run"], "pair_id": row["pair_id"],
            "method": "CLS-to-patch rollout from recomputed timm qkv attention; head-mean with residual normalization",
        })
    figure.savefig(output / "attention_rollout.png", dpi=180)
    plt.close(figure)
    atomic_write_json(output / "attention_rollout.json", {
        "schema_version": "sac-attention-rollout-v1", "rows": metadata,
        "claim_boundary": "Attention rollout is descriptive and is not causal feature attribution.",
    })


def _macro_cluster_confusion(
    cfg: Config, pair_rows: Sequence[dict[str, Any]], output: Path,
) -> None:
    if set(_MACRO_CLUSTER) != set(cfg.data.class_names):
        raise ValueError("macro-cluster mapping must cover every EuroSAT class exactly once")
    detail: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in pair_rows:
        if row.get("audit_qualified") is not True or row["severity"] == "sham" or not _is_failure(row):
            continue
        predicted_class = cfg.data.class_names[int(np.argmax(row["transformed_probabilities"]))]
        true_macro = _MACRO_CLUSTER[row["class_name"]]
        predicted_macro = _MACRO_CLUSTER[predicted_class]
        grouped[(row["model"], row["family"], row["severity"], true_macro, predicted_macro)] += 1
    for (model, family, severity, true_macro, predicted_macro), count in sorted(grouped.items()):
        detail.append({
            "model": model, "family": family, "severity": severity,
            "true_macro_cluster": true_macro, "predicted_macro_cluster": predicted_macro,
            "count": count, "error_scope": "within_macro" if true_macro == predicted_macro else "cross_macro",
        })
    if not detail:
        detail = [{
            "model": "none", "family": "none", "severity": "none",
            "true_macro_cluster": "none", "predicted_macro_cluster": "none",
            "count": 0, "error_scope": "no_counterfactual_failures",
        }]
    _write_csv(output / "macro_cluster_confusion.csv", detail)
    labels = ("natural", "anthropogenic")
    figure, axes = plt.subplots(1, len(cfg.experiment.model_presets), figsize=(4 * len(cfg.experiment.model_presets), 4), constrained_layout=True)
    for axis, preset in zip(axes, cfg.experiment.model_presets, strict=True):
        matrix = np.zeros((2, 2), dtype=int)
        for row in detail:
            if row["model"] == preset.name and row["true_macro_cluster"] in labels:
                matrix[labels.index(row["true_macro_cluster"]), labels.index(row["predicted_macro_cluster"])] += int(row["count"])
        image = axis.imshow(matrix, cmap="Reds")
        axis.set(
            xticks=range(2), yticks=range(2), xticklabels=labels, yticklabels=labels,
            xlabel="Predicted macro-cluster", ylabel="True macro-cluster",
            title=f"{_DISPLAY_NAMES[preset.name]}\nclean-correct CF failures",
        )
        axis.tick_params(axis="x", rotation=30)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=.046)
    figure.savefig(output / "macro_cluster_confusion.png", dpi=180)
    plt.close(figure)
    atomic_write_json(output / "macro_cluster_mapping.json", {
        "schema_version": "sac-eurosat-macro-cluster-v1",
        "mapping": _MACRO_CLUSTER,
        "note": "PermanentCrop is included with the other human-managed land-cover classes; no EuroSAT class is omitted.",
    })


def generate_extended_publication_artifacts(
    cfg: Config,
    evaluation_dirs: Sequence[Path],
    provenance: Sequence[dict[str, Any]],
    clean_rows: Sequence[dict[str, Any]],
    condition_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    statistics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    training_run_dirs: Sequence[str | Path],
    output: Path,
) -> list[dict[str, Any]]:
    model_names, seeds, pairs = protocol_matrix(protocol, cfg)
    scoped_cfg = dataclasses.replace(
        cfg,
        experiment=dataclasses.replace(
            cfg.experiment,
            model_presets=tuple(preset for preset in cfg.experiment.model_presets if preset.name in model_names),
            seeds=tuple(seeds),
        ),
        statistics=dataclasses.replace(cfg.statistics, model_pair_contrasts=tuple(pairs)),
    )
    training_paths, events, training_sources = _training_contracts(scoped_cfg, training_run_dirs, protocol)
    _learning_curves(scoped_cfg, events, output)
    _master_summary(scoped_cfg, clean_rows, pair_rows, statistics, training_sources, output)
    _pairwise_forest(scoped_cfg, statistics, output)
    _cross_model_cka(scoped_cfg, evaluation_dirs, provenance, output)
    _throughput_pareto(scoped_cfg, provenance, pair_rows, output)
    _macro_cluster_confusion(scoped_cfg, pair_rows, output)
    _qualitative_gallery(scoped_cfg, pair_rows, output)
    _attention_figure(scoped_cfg, pair_rows, training_paths, protocol, output)
    return training_sources
