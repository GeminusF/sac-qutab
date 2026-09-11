from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .campaign import protocol_matrix
from .config import REPORT_ARTIFACT_PLAN, Config, config_digest
from .interventions import pair_condition_signature
from .metrics import expected_calibration_error
from .publication import generate_extended_publication_artifacts
from .review import validate_review_freeze_contract
from .runtime import atomic_write_json, read_jsonl, runtime_path, sha256_file
from .tracking import publish_report_to_wandb

_SEVERITY_ORDER = {"sham": 0, "mild": 1, "moderate": 2, "severe": 3}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty report table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    materialized = list(rows)
    return {key: float(np.mean([float(row[key]) for row in materialized])) for key in keys}


def _reliability(axis: Any, probabilities: np.ndarray, targets: np.ndarray, bins: int, label: str) -> None:
    result = expected_calibration_error(probabilities, targets, bins)
    populated = [row for row in result["bins"] if row["count"]]
    axis.plot([row["mean_confidence"] for row in populated], [row["accuracy"] for row in populated], marker="o", label=f"{label} (ECE={result['ece']:.3f})")


def generate_report_artifacts(
    cfg: Config,
    evaluation_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    statistics_path: str | Path | None = None,
    review_freeze_path: str | Path | None = None,
    training_run_dirs: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    if not evaluation_dirs:
        raise ValueError("at least one evaluation directory is required")
    protocol_path = Path(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selected_models, selected_seeds, selected_pairs = protocol_matrix(protocol, cfg)
    expected_names = {f"{model}-seed{seed}" for model in selected_models for seed in selected_seeds}
    metric_contract = {
        "ece_bins": cfg.evaluation.ece_bins,
        "rq3_population": "selected non-sham pairs whose clean prediction is correct, per model seed",
        "rq3_min_positives_calibration_and_test": cfg.evaluation.rq3_min_positives_per_model_seed,
        "pr_summary": cfg.evaluation.pr_summary,
        "normalization": cfg.evaluation.representation_normalization,
    }
    statistics_contract = {
        "primary_contrasts": list(cfg.statistics.primary_contrasts),
        "model_pair_contrasts": [
            {"left": pair.left, "right": pair.right} for pair in selected_pairs
        ],
        "cluster_unit": "source_group_id",
        "bootstrap_samples": cfg.statistics.bootstrap_samples, "confidence_level": cfg.statistics.confidence_level,
        "primary_familywise_alpha": cfg.statistics.primary_familywise_alpha,
        "seed_summary": "mean_and_sample_sd", "exploratory_fdr_q": cfg.statistics.fdr_q,
    }
    if (protocol.get("config_sha256") != config_digest(cfg)
            or protocol.get("condition_signature") != pair_condition_signature(cfg)
            or set(protocol.get("resolved_run_sha256", {})) != expected_names
            or set(protocol.get("checkpoint_sha256", {})) != expected_names
            or set(protocol.get("detector_sha256", {})) != expected_names
            or set(protocol.get("control_ecdf_sha256", {})) != expected_names
            or set(protocol.get("pair_manifest_sha256", {})) != {"validation", "calibration", "final_test"}
            or protocol.get("metric_contract") != metric_contract
            or protocol.get("statistics_contract") != statistics_contract
            or protocol.get("report_plan") != list(REPORT_ARTIFACT_PLAN)):
        raise ValueError("experiment protocol does not match the configured run-matrix scientific contract")
    directories = [Path(value) for value in evaluation_dirs]
    provenance_records: list[dict[str, Any]] = []
    run_names: set[str] = set()
    for directory in directories:
        provenance_path = directory / "provenance.json"
        pair_path = directory / "pair_metrics.jsonl"
        metrics_path = directory / "metrics.json"
        representations_path = directory / "representations.npz"
        detector_path = directory / "detector_metrics.json"
        index_path = directory / "evaluation_index.json"
        if any(not path.is_file() for path in (provenance_path, pair_path, metrics_path, representations_path, detector_path, index_path)):
            raise FileNotFoundError(f"evaluation publication input is incomplete: {directory}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("schema_version") != "sac-evaluation-v2" or provenance.get("split_role") != "final_test":
            raise ValueError("scientific report artifacts require frozen final-test evaluations")
        run_name = f"{provenance.get('model_name')}-seed{provenance.get('training_seed')}"
        if run_name in run_names or run_name not in expected_names:
            raise ValueError(f"duplicate or unexpected final-test evaluation: {run_name}")
        if (provenance.get("protocol_sha256") != protocol_sha256
                or provenance.get("config_sha256") != protocol.get("config_sha256")
                or provenance.get("resolved_run_sha256") != protocol["resolved_run_sha256"][run_name]
                or provenance.get("checkpoint_sha256") != protocol["checkpoint_sha256"][run_name]
                or provenance.get("pair_manifest_sha256") != protocol["pair_manifest_sha256"]["final_test"]
                or provenance.get("review_freeze_sha256") != protocol["review_freeze_sha256"]
                or provenance.get("condition_signature") != protocol["condition_signature"]):
            raise ValueError(f"report evaluation differs from the real protocol: {run_name}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (index.get("schema_version") != "sac-evaluation-index-v1" or index.get("protocol_sha256") != protocol_sha256
                or index.get("provenance_sha256") != sha256_file(provenance_path)
                or index.get("pair_metrics_sha256") != sha256_file(pair_path)
                or index.get("metrics_sha256") != sha256_file(metrics_path)
                or index.get("representations_sha256") != sha256_file(representations_path)):
            raise ValueError(f"evaluation index does not authenticate report inputs: {run_name}")
        run_names.add(run_name)
        provenance_records.append(provenance)
    if run_names != expected_names:
        raise ValueError("scientific report artifacts require the exact configured protocol matrix")
    if statistics_path is None:
        raise ValueError("scientific report artifacts require frozen primary statistics")
    statistics_path = Path(statistics_path)
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if (statistics.get("schema_version") != "sac-primary-statistics-v3"
            or statistics.get("protocol_sha256") != protocol_sha256
            or statistics.get("protocol_source", {}).get("sha256") != protocol_sha256):
        raise ValueError("primary statistics do not match the real final-test experiment protocol")
    statistical_sources = {row["run"]: row for row in statistics.get("sources", [])}
    if len(statistical_sources) != len(expected_names) or set(statistical_sources) != expected_names:
        raise ValueError("report evaluation runs differ from the frozen statistics sources")
    for directory, provenance in zip(directories, provenance_records, strict=True):
        run_name = f"{provenance['model_name']}-seed{provenance['training_seed']}"
        source = statistical_sources[run_name]
        paths = {
            "provenance_sha256": directory / "provenance.json", "pairs_sha256": directory / "pair_metrics.jsonl",
            "metrics_sha256": directory / "metrics.json", "representations_sha256": directory / "representations.npz",
            "evaluation_index_sha256": directory / "evaluation_index.json", "detector_sha256": directory / "detector_metrics.json",
        }
        if any(source.get(field) != sha256_file(path) for field, path in paths.items()):
            raise ValueError(f"report source digests differ from frozen statistics: {run_name}")
    if review_freeze_path is None:
        raise ValueError("scientific report artifacts require a frozen condition-selection policy")
    review = json.loads(Path(review_freeze_path).read_text(encoding="utf-8"))
    validate_review_freeze_contract(cfg, review)
    review_hashes = {row.get("review_freeze_sha256") for row in provenance_records}
    condition_signatures = {row.get("condition_signature") for row in provenance_records}
    if (sha256_file(review_freeze_path) != protocol.get("review_freeze_sha256")
            or review_hashes != {protocol.get("review_freeze_sha256")}
            or condition_signatures != {protocol.get("condition_signature")}
            or review.get("condition_signature") != protocol.get("condition_signature")):
        raise ValueError("condition-selection policy does not match the final-test protocol")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    clean_table: list[dict[str, Any]] = []
    condition_table: list[dict[str, Any]] = []
    class_table: list[dict[str, Any]] = []
    sham_table: list[dict[str, Any]] = []
    detector_table: list[dict[str, Any]] = []
    prevalence_table: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for directory, provenance in zip(directories, provenance_records, strict=True):
        provenance_path = directory / "provenance.json"
        metrics_path = directory / "metrics.json"
        pair_path = directory / "pair_metrics.jsonl"
        if not metrics_path.is_file() or not pair_path.is_file():
            raise FileNotFoundError(f"incomplete evaluation directory: {directory}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_name = f"{provenance['model_name']}-seed{provenance['training_seed']}"
        clean = metrics["clean"]
        calibration = metrics["clean_calibration"]
        clean_table.append({
            "run": run_name, "model": provenance["model_name"], "seed": provenance["training_seed"],
            "count": clean["count"], "accuracy": clean["accuracy"], "macro_f1": clean["macro_f1"],
            "ece": calibration["ece"]["ece"], "brier": calibration["brier"], "nll": calibration["nll"],
        })
        rows = read_jsonl(pair_path)
        for row in rows:
            all_rows.append({**row, "run": run_name, "model": provenance["model_name"], "seed": provenance["training_seed"]})
        for condition, values in sorted(metrics["conditions"].items()):
            family, severity = condition.split(":", 1)
            confidence = values["confidence_change"]
            calibration_values = values["calibration"]
            record = {
                "run": run_name, "model": provenance["model_name"], "seed": provenance["training_seed"],
                "family": family, "severity": severity, "count": values["count"], "unique_source_count": values["unique_source_count"],
                "transformed_accuracy": values["transformed_accuracy"], "label_consistency": values["label_consistency"],
                "js_mean": values["js_mean"], "entropy_mean": values["entropy_mean"],
                "cosine_mean_within_model": values["cosine_mean_within_model"], "control_margin_mean_within_model": values["control_margin_mean_within_model"],
                "normalized_instability_mean": values["normalized_instability_mean"], "failure_count": values["failure_count"],
                "clean_correct_count": values["clean_correct_count"], "failure_prevalence_among_clean_correct": values["failure_prevalence_among_clean_correct"],
                "confidence_mean_signed": confidence["mean_signed"], "confidence_mean_absolute": confidence["mean_absolute"],
                "confidence_median": confidence["median"], "confidence_q25": confidence["q25"], "confidence_q75": confidence["q75"],
                "increased_confidence_failure_fraction": confidence["increased_confidence_failure_fraction"],
                "ece": calibration_values["ece"]["ece"], "ece_bin_counts": json.dumps([row["count"] for row in calibration_values["ece"]["bins"]]),
                "brier": calibration_values["brier"], "nll": calibration_values["nll"], "linear_cka": values.get("linear_cka"),
            }
            (sham_table if severity == "sham" else condition_table).append(record)
            for class_name, cell in values["class_cells"].items():
                if severity != "sham":
                    class_table.append({"run": run_name, "model": provenance["model_name"], "seed": provenance["training_seed"], "family": family, "severity": severity, "class_name": class_name, **cell})
        detector_path = directory / "detector_metrics.json"
        if detector_path.is_file():
            detector = json.loads(detector_path.read_text(encoding="utf-8"))
            for condition, values in detector.get("condition_prevalence", {}).items():
                family, severity = condition.split(":", 1)
                prevalence_table.append({"run": run_name, "model": provenance["model_name"], "seed": provenance["training_seed"], "family": family, "severity": severity, **values})
        sources.append({
            "directory": str(directory), "provenance_sha256": sha256_file(provenance_path),
            "metrics_sha256": sha256_file(metrics_path), "pairs_sha256": sha256_file(pair_path),
            "representations_sha256": sha256_file(representations_path),
            "evaluation_index_sha256": sha256_file(directory / "evaluation_index.json"),
            "detector_sha256": sha256_file(detector_path),
        })
    detector_table = [
        {"run": f"{row['model']}-seed{row['seed']}", "detector": row["score"], **row}
        for row in statistics.get("rq3_absolute_clustered", [])
    ]
    _write_csv(output / "clean_metrics.csv", clean_table)
    _write_csv(output / "condition_metrics.csv", condition_table)
    _write_csv(output / "condition_class_metrics.csv", class_table)
    _write_csv(output / "sham_controls.csv", sham_table)
    _write_csv(output / "detector_metrics.csv", detector_table)
    _write_csv(output / "rq3_condition_prevalence.csv", prevalence_table)
    _write_csv(output / "primary_contrasts_per_seed.csv", statistics["primary_per_seed"])
    _write_csv(output / "primary_contrasts_seed_summary.csv", statistics["primary_seed_summaries"])
    _write_csv(output / "exploratory_bh.csv", statistics["exploratory_bh"])

    families = sorted({row["family"] for row in condition_table})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model in sorted({row["model"] for row in condition_table}):
        for family in families:
            selected = [row for row in condition_table if row["model"] == model and row["family"] == family]
            values: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                values[row["severity"]].append(row)
            severities = sorted(values, key=lambda severity: _SEVERITY_ORDER[severity])
            x = [_SEVERITY_ORDER[item] for item in severities]
            label_means = [np.mean([row["label_consistency"] for row in values[item]]) for item in severities]
            label_sd = [np.std([row["label_consistency"] for row in values[item]], ddof=1) if len(values[item]) > 1 else 0 for item in severities]
            repr_means = [np.mean([row["normalized_instability_mean"] for row in values[item]]) for item in severities]
            repr_sd = [np.std([row["normalized_instability_mean"] for row in values[item]], ddof=1) if len(values[item]) > 1 else 0 for item in severities]
            axes[0].errorbar(x, label_means, yerr=label_sd, marker="o", capsize=2, label=f"{model}/{family}")
            axes[1].errorbar(x, repr_means, yerr=repr_sd, marker="o", capsize=2, label=f"{model}/{family}")
    axes[0].set(title="Prediction stability (seed mean±SD)", xlabel="Severity", ylabel="Label consistency", xticks=[1, 2, 3], xticklabels=["mild", "moderate", "severe"])
    axes[1].set(title="Normalized instability (seed mean±SD)", xlabel="Severity", ylabel="Validation-control ECDF percentile", xticks=[1, 2, 3], xticklabels=["mild", "moderate", "severe"])
    axes[1].legend(fontsize=7, loc="best")
    figure.savefig(output / "severity_response.png", dpi=180)
    plt.close(figure)

    non_sham_rows = [row for row in all_rows if row["severity"] != "sham"]
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for model in sorted({row["model"] for row in non_sham_rows}):
        selected = [row for row in non_sham_rows if row["model"] == model]
        axis.scatter([row["confidence_change"] for row in selected], [row["normalized_representation_instability"] for row in selected], s=8, alpha=0.25, label=model)
    axis.set(xlabel="Signed confidence change", ylabel="Normalized representation instability", title="Selected non-sham pairs")
    axis.legend()
    figure.savefig(output / "confidence_representation_scatter.png", dpi=180)
    plt.close(figure)

    class_names = sorted({row["class_name"] for row in class_table})
    columns = [(model, family) for model in sorted({row["model"] for row in class_table}) for family in families]
    heat = np.full((len(class_names), len(columns)), np.nan)
    for i, class_name in enumerate(class_names):
        for j, (model, family) in enumerate(columns):
            selected = [row["label_consistency"] for row in class_table if row["class_name"] == class_name and row["model"] == model and row["family"] == family and row["severity"] == "severe"]
            if selected:
                heat[i, j] = np.mean(selected)
    figure, axis = plt.subplots(figsize=(max(7, len(columns) * 1.1), max(4, len(class_names) * 0.45)), constrained_layout=True)
    image = axis.imshow(heat, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axis.set(yticks=range(len(class_names)), yticklabels=class_names, xticks=range(len(columns)), xticklabels=[f"{model}\n{family}" for model, family in columns], title="Class label consistency at selected severe conditions")
    figure.colorbar(image, ax=axis)
    figure.savefig(output / "class_stability_heatmap.png", dpi=180)
    plt.close(figure)

    labels = list(range(len(class_names)))
    for run_name in sorted({row["run"] for row in all_rows}):
        run_rows = [row for row in all_rows if row["run"] == run_name]
        unique: dict[str, dict[str, Any]] = {}
        for row in run_rows:
            unique.setdefault(row["source_id"], row)
        clean_true = [row["label"] for row in unique.values()]
        clean_pred = [int(np.argmax(row["original_probabilities"])) for row in unique.values()]
        for family in families:
            family_rows = [row for row in run_rows if row["family"] == family and row["severity"] != "sham"]
            strongest = max({row["severity"] for row in family_rows}, key=lambda severity: _SEVERITY_ORDER[severity])
            transformed = [row for row in family_rows if row["severity"] == strongest]
            matrices = (
                confusion_matrix(clean_true, clean_pred, labels=labels),
                confusion_matrix([row["label"] for row in transformed], [int(np.argmax(row["transformed_probabilities"])) for row in transformed], labels=labels),
            )
            figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            for axis, matrix, title in zip(axes, matrices, ("Clean", f"{family}:{strongest}")):
                image = axis.imshow(matrix, cmap="Blues")
                axis.set(title=title, xlabel="Predicted", ylabel="True", xticks=labels, yticks=labels)
                figure.colorbar(image, ax=axis)
            figure.savefig(output / f"confusion_{run_name}_{family}_{strongest}.png", dpi=180)
            plt.close(figure)

            clean_prob = np.asarray([row["original_probabilities"] for row in unique.values()], dtype=float)
            clean_targets = np.asarray(clean_true, dtype=int)
            transformed_prob = np.asarray([row["transformed_probabilities"] for row in transformed], dtype=float)
            transformed_targets = np.asarray([row["label"] for row in transformed], dtype=int)
            figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
            axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
            _reliability(axis, clean_prob, clean_targets, cfg.evaluation.ece_bins, "clean")
            _reliability(axis, transformed_prob, transformed_targets, cfg.evaluation.ece_bins, f"{family}:{strongest}")
            axis.set(xlabel="Mean confidence", ylabel="Empirical accuracy", title=f"Reliability: {run_name}", xlim=(0, 1), ylim=(0, 1))
            axis.legend(fontsize=8)
            figure.savefig(output / f"reliability_{run_name}_{family}_{strongest}.png", dpi=180)
            plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for axis, metric, title in zip(axes, ("average_precision", "auroc"), ("Average precision", "AUROC"), strict=True):
        rows = [row for row in detector_table if row["metric"] == metric and row.get("defined") is True]
        if not rows:
            axis.text(0.5, 0.5, "RQ3 underpowered or undefined", ha="center", va="center")
            axis.set_axis_off()
            continue
        labels_text = [f"{row['run']}\n{row['detector']}" for row in rows]
        estimates = np.asarray([row["estimate"] for row in rows], dtype=float)
        lower = np.asarray([row["lower"] for row in rows], dtype=float)
        upper = np.asarray([row["upper"] for row in rows], dtype=float)
        axis.errorbar(range(len(rows)), estimates, yerr=np.vstack((estimates - lower, upper - estimates)), fmt="o", capsize=2)
        axis.set(xticks=range(len(labels_text)), xticklabels=labels_text, ylabel=title, title=f"RQ3 {title} with original-image clustered 95% intervals", ylim=(0, 1))
        axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    figure.savefig(output / "rq3_failure_detection.png", dpi=180)
    plt.close(figure)

    if training_run_dirs is None:
        raise ValueError("scientific report artifacts require the configured training-run directories")
    training_sources = generate_extended_publication_artifacts(
        cfg, directories, provenance_records, clean_table, condition_table, all_rows,
        statistics, protocol, training_run_dirs, output,
    )

    if review["schema_version"] == "sac-review-freeze-v2":
        panel = runtime_path(review["qualitative_panel_path"])
        if sha256_file(panel) != review["qualitative_panel_sha256"]:
            raise ValueError("semantic review qualitative panel digest mismatch")
        shutil.copyfile(panel, output / "qualitative_audit_panel.png")
        review_source = {"policy": "human_semantic_review", "freeze_sha256": sha256_file(review_freeze_path), "panel_sha256": sha256_file(panel)}
    else:
        shutil.copyfile(review_freeze_path, output / "condition_selection_policy.json")
        (output / "semantic_review_limitation.txt").write_text(review["limitation"] + "\n", encoding="utf-8")
        review_source = {
            "policy": review["policy"],
            "freeze_sha256": sha256_file(review_freeze_path),
            "human_semantic_review_performed": False,
            "limitation": review["limitation"],
        }
    generated = sorted(path.name for path in output.iterdir())
    required = {
        "clean_metrics.csv", "condition_metrics.csv", "condition_class_metrics.csv", "sham_controls.csv",
        "detector_metrics.csv", "rq3_condition_prevalence.csv", "primary_contrasts_per_seed.csv",
        "primary_contrasts_seed_summary.csv", "exploratory_bh.csv", "severity_response.png",
        "confidence_representation_scatter.png", "class_stability_heatmap.png", "rq3_failure_detection.png",
        "learning_curves.csv", "learning_curves.png", "master_summary.csv", "master_summary.tex",
        "pairwise_forest_plot.csv", "pairwise_forest_plot.png",
        "cross_model_cka_matrix.csv", "cross_model_cka_matrix.png",
        "throughput_vs_robustness.csv", "throughput_vs_robustness.png",
        "qualitative_failure_gallery.json", "qualitative_failure_gallery.png",
        "attention_rollout.json", "attention_rollout.png",
        "macro_cluster_confusion.csv", "macro_cluster_confusion.png", "macro_cluster_mapping.json",
    }
    if review["schema_version"] == "sac-review-freeze-v2":
        required.add("qualitative_audit_panel.png")
    else:
        required.update({"condition_selection_policy.json", "semantic_review_limitation.txt"})
    for run_name in expected_names:
        for family in families:
            severities = {row["severity"] for row in condition_table if row["run"] == run_name and row["family"] == family}
            if not severities:
                raise RuntimeError(f"missing report condition cells for {run_name}/{family}")
            strongest = max(severities, key=lambda severity: _SEVERITY_ORDER[severity])
            required.update({f"confusion_{run_name}_{family}_{strongest}.png", f"reliability_{run_name}_{family}_{strongest}.png"})
    if not required <= set(generated):
        raise RuntimeError(f"report artifact set is incomplete: {sorted(required-set(generated))}")
    index = {
        "schema_version": "sac-report-artifacts-v4", "protocol_sha256": protocol_sha256,
        "protocol_source": {"path": str(protocol_path), "sha256": protocol_sha256}, "sources": sources,
        "statistics_source": {"path": str(statistics_path), "sha256": sha256_file(statistics_path)},
        "training_sources": training_sources,
        "review_source": review_source, "generated": generated,
        "campaign_sha256": protocol.get("campaign", {}).get("sha256"),
        "hardware_limitation": protocol.get("hardware_limitation"),
        "claim_boundary": "Artifacts reflect supplied frozen final-test evaluations; local checks are not scientific results. Training throughput is not an architecture comparison when training hardware differs by model.",
    }
    atomic_write_json(output / "index.json", index)
    publish_report_to_wandb(cfg, output, index)
    return {"evaluation_count": len(evaluation_dirs), "pair_count": len(all_rows), "output": str(output), "generated": len(generated)}
