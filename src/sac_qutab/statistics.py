from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from .campaign import protocol_matrix
from .config import Config, ModelPairContrast, config_digest
from .metrics import MetricUndefinedError
from .runtime import atomic_write_json, read_jsonl, sha256_file
from .tracking import publish_statistics_to_wandb


def _validate_paired(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lmap = {row["pair_id"]: row for row in left}
    rmap = {row["pair_id"]: row for row in right}
    if len(lmap) != len(left) or len(rmap) != len(right):
        raise ValueError("duplicate pair IDs")
    if set(lmap) != set(rmap):
        raise ValueError("paired analyses require identical pair IDs; missing rows cannot be dropped")
    pair_ids = sorted(lmap)
    groups = np.asarray([lmap[pair_id]["source_group_id"] for pair_id in pair_ids], dtype=object)
    if any(rmap[pair_id]["source_group_id"] != lmap[pair_id]["source_group_id"] for pair_id in pair_ids):
        raise ValueError("paired group IDs differ")
    a = np.asarray([lmap[pair_id][value_key] for pair_id in pair_ids], dtype=float)
    b = np.asarray([rmap[pair_id][value_key] for pair_id in pair_ids], dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired values contain NaN or infinity")
    return groups, a, b


def cluster_bootstrap_difference(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], value_key: str, samples: int, confidence: float, seed: int, *, statistic: Callable[[np.ndarray, np.ndarray], float] | None = None) -> dict[str, Any]:
    groups, a, b = _validate_paired(left, right, value_key)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise MetricUndefinedError("cluster bootstrap requires at least two original-image groups")
    statistic = statistic or (lambda x, y: float(np.mean(x - y)))
    point = statistic(a, b)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([indices[group] for group in selected])
        estimates.append(statistic(a[sampled], b[sampled]))
    alpha = 1 - confidence
    return {"effect": float(point), "ci": [float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))], "clusters": len(unique), "replicates": samples, "confidence": confidence, "uncertainty_source": "original_image_sampling_within_seed"}


def clustered_detector_delta(targets: Sequence[int], scores_plus: Sequence[float], scores_output: Sequence[float], groups: Sequence[str], metric: str, samples: int, confidence: float, seed: int) -> dict[str, Any]:
    y = np.asarray(targets, dtype=int)
    plus = np.asarray(scores_plus, dtype=float)
    output = np.asarray(scores_output, dtype=float)
    g = np.asarray(groups, dtype=object)
    if not (len(y) == len(plus) == len(output) == len(g)) or len(np.unique(y)) < 2:
        raise MetricUndefinedError("paired detector delta requires equal lengths and both target classes")
    fn = average_precision_score if metric == "average_precision" else roc_auc_score if metric == "auroc" else None
    if fn is None:
        raise ValueError("detector metric must be average_precision or auroc")
    unique = np.unique(g)
    indices = {group: np.flatnonzero(g == group) for group in unique}
    point = float(fn(y, plus) - fn(y, output))
    rng = np.random.default_rng(seed)
    values = []
    invalid = 0
    for _ in range(samples):
        sampled = np.concatenate([indices[group] for group in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(y[sampled])) < 2:
            invalid += 1
            continue
        values.append(float(fn(y[sampled], plus[sampled]) - fn(y[sampled], output[sampled])))
    if len(values) < max(100, samples // 2):
        return {"defined": False, "reason": "too few valid clustered detector replicates", "effect": point, "valid_replicates": len(values), "invalid_replicates": invalid}
    alpha = 1 - confidence
    return {"defined": True, "effect": point, "ci": [float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))], "valid_replicates": len(values), "invalid_replicates": invalid, "uncertainty_source": "original_image_sampling_within_seed"}


def clustered_detector_metric(targets: Sequence[int], scores: Sequence[float], groups: Sequence[str], metric: str, samples: int, confidence: float, seed: int) -> dict[str, Any]:
    y = np.asarray(targets, dtype=int)
    values = np.asarray(scores, dtype=float)
    group_array = np.asarray(groups, dtype=object)
    if not (len(y) == len(values) == len(group_array)) or len(np.unique(y)) < 2:
        raise MetricUndefinedError("clustered detector metric requires equal lengths and both target classes")
    fn = average_precision_score if metric == "average_precision" else roc_auc_score if metric == "auroc" else None
    if fn is None:
        raise ValueError("detector metric must be average_precision or auroc")
    unique = np.unique(group_array)
    if len(unique) < 2:
        raise MetricUndefinedError("clustered detector metric requires at least two original-image groups")
    indices = {group: np.flatnonzero(group_array == group) for group in unique}
    point = float(fn(y, values))
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    invalid = 0
    for _ in range(samples):
        sampled = np.concatenate([indices[group] for group in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(y[sampled])) < 2:
            invalid += 1
            continue
        estimates.append(float(fn(y[sampled], values[sampled])))
    if len(estimates) < max(100, samples // 2):
        return {"defined": False, "reason": "too few valid clustered detector replicates", "estimate": point, "clusters": len(unique), "valid_replicates": len(estimates), "invalid_replicates": invalid, "confidence": confidence}
    alpha = 1 - confidence
    lower = float(np.quantile(estimates, alpha / 2))
    upper = float(np.quantile(estimates, 1 - alpha / 2))
    return {"defined": True, "estimate": point, "lower": lower, "upper": upper, "ci": [lower, upper], "clusters": len(unique), "valid_replicates": len(estimates), "invalid_replicates": invalid, "confidence": confidence, "uncertainty_source": "original_image_cluster_bootstrap_within_seed"}


def wilcoxon_sensitivity(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Wilcoxon requires matching finite non-empty vectors")
    differences = a - b
    nonzero = differences[differences != 0]
    if not len(nonzero):
        raise MetricUndefinedError("Wilcoxon is undefined when every paired difference is zero")
    test = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
    ranks = rankdata(np.abs(nonzero))
    rank_biserial = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    return {"statistic": float(test.statistic), "p_value": float(test.pvalue), "rank_biserial": rank_biserial, "nonzero_pairs": len(nonzero)}


def benjamini_hochberg(p_values: Sequence[float], q: float) -> dict[str, Any]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1)) or not np.all(np.isfinite(values)):
        raise ValueError("BH requires finite p-values in [0,1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1)
    return {"adjusted_q_values": adjusted.tolist(), "rejected": (adjusted <= q).tolist(), "fdr_q": q}


def holm_bonferroni(p_values: Sequence[float], alpha: float) -> dict[str, Any]:
    values = np.asarray(p_values, dtype=float)
    if (values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1))
            or not np.all(np.isfinite(values)) or not 0 < alpha < 1):
        raise ValueError("Holm correction requires finite p-values in [0,1] and alpha in (0,1)")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return {
        "adjusted_p_values": adjusted.tolist(),
        "rejected": (adjusted <= alpha).tolist(),
        "familywise_alpha": alpha,
        "test_count": len(values),
    }


def summarize_seed_estimates(estimates: dict[int, float]) -> dict[str, Any]:
    if not estimates:
        raise MetricUndefinedError("seed summary is empty")
    values = np.asarray(list(estimates.values()), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("seed estimates contain NaN or infinity")
    return {"per_seed": {str(k): float(v) for k, v in sorted(estimates.items())}, "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None, "seed_count": len(values), "uncertainty_source": "training_seed_estimates"}


def two_level_bootstrap(seed_rows: dict[int, Sequence[dict[str, Any]]], value_key: str, samples: int, confidence: float, seed: int) -> dict[str, Any]:
    if len(seed_rows) < 2:
        raise MetricUndefinedError("two-level bootstrap requires at least two training seeds")
    rng = np.random.default_rng(seed)
    seed_ids = np.asarray(sorted(seed_rows))
    estimates = []
    for _ in range(samples):
        selected_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_means = []
        for selected_seed in selected_seeds:
            rows = seed_rows[int(selected_seed)]
            by_group: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                by_group[row["source_group_id"]].append(float(row[value_key]))
            groups = list(by_group)
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            seed_means.append(float(np.mean([value for group in sampled_groups for value in by_group[group]])))
        estimates.append(float(np.mean(seed_means)))
    alpha = 1 - confidence
    return {"ci": [float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))], "replicates": samples, "seed_count": len(seed_ids), "uncertainty_source": "explicit_two_level_seed_and_image_bootstrap", "warning": "three seeds do not support a precise training-population claim"}


def _clustered_effect(values: Sequence[float], groups: Sequence[str], samples: int, confidence: float, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    group_array = np.asarray(groups, dtype=object)
    if len(array) != len(group_array) or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("clustered effect requires matching finite nonempty values/groups")
    unique = np.unique(group_array)
    if len(unique) < 2:
        raise MetricUndefinedError("clustered effect requires at least two original-image groups")
    indices = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([indices[group] for group in selected])
        estimates.append(float(array[sampled].mean()))
    alpha = 1 - confidence
    return {
        "defined": True, "effect": float(array.mean()),
        "ci": [float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))],
        "clusters": len(unique), "observations": len(array), "valid_replicates": samples, "invalid_replicates": 0,
        "uncertainty_source": "original_image_cluster_bootstrap_within_seed",
    }


def _severity_differences(rows: Sequence[dict[str, Any]], family: str, metric: str, *, label_inconsistency: bool = False) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("audit_qualified") is True and row["family"] == family and row["severity"] in {"mild", "severe"}]
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        if row["severity"] in by_source[row["source_id"]]:
            raise ValueError("duplicate severity row for one source/model seed")
        by_source[row["source_id"]][row["severity"]] = row
    if not by_source or any(set(values) != {"mild", "severe"} for values in by_source.values()):
        raise MetricUndefinedError(f"{family} lacks complete accepted mild/severe support")
    differences = []
    for source_id, values in sorted(by_source.items()):
        mild, severe = values["mild"], values["severe"]
        if mild["source_group_id"] != severe["source_group_id"] or mild.get("latent_seed") != severe.get("latent_seed"):
            raise ValueError("severity pairing changed original group or latent intervention")
        mild_value, severe_value = float(mild[metric]), float(severe[metric])
        effect = (1 - severe_value) - (1 - mild_value) if label_inconsistency else severe_value - mild_value
        differences.append({"source_id": source_id, "source_group_id": mild["source_group_id"], "effect": effect})
    return differences


def _pairwise_model_summaries(
    cfg: Config,
    difference_cache: dict[tuple[str, str], list[dict[str, Any]]],
    families: Sequence[str],
    *, seeds: Sequence[int] | None = None,
    model_pairs: Sequence[ModelPairContrast] | None = None,
) -> list[dict[str, Any]]:
    selected_seeds = tuple(seeds or cfg.experiment.seeds)
    selected_pairs = tuple(model_pairs or cfg.statistics.model_pair_contrasts)
    summaries: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    valid_p_values: list[float] = []
    for pair_index, pair in enumerate(selected_pairs):
        by_seed_group: dict[int, dict[str, float]] = {}
        for seed in selected_seeds:
            grouped: dict[str, list[float]] = defaultdict(list)
            for family in families:
                left = {row["source_id"]: row for row in difference_cache[(f"{pair.left}-seed{seed}", family)]}
                right = {row["source_id"]: row for row in difference_cache[(f"{pair.right}-seed{seed}", family)]}
                if set(left) != set(right):
                    raise ValueError("pairwise summary requires identical source support")
                for source_id in sorted(left):
                    if left[source_id]["source_group_id"] != right[source_id]["source_group_id"]:
                        raise ValueError("pairwise summary original group mismatch")
                    grouped[str(left[source_id]["source_group_id"])].append(
                        float(right[source_id]["effect"]) - float(left[source_id]["effect"])
                    )
            by_seed_group[seed] = {group: float(np.mean(values)) for group, values in grouped.items()}
        group_sets = {frozenset(values) for values in by_seed_group.values()}
        if len(group_sets) != 1 or not next(iter(group_sets), frozenset()):
            raise ValueError("pairwise summary requires identical original-image groups across seeds")
        point = float(np.mean([np.mean(list(values.values())) for values in by_seed_group.values()]))
        rng = np.random.default_rng(cfg.statistics.seed + 10_000 + pair_index)
        seed_ids = np.asarray(selected_seeds, dtype=int)
        estimates: list[float] = []
        for _ in range(cfg.statistics.bootstrap_samples):
            sampled_seed_means: list[float] = []
            for sampled_seed in rng.choice(seed_ids, size=len(seed_ids), replace=True):
                group_values = by_seed_group[int(sampled_seed)]
                groups = np.asarray(sorted(group_values), dtype=object)
                sampled_groups = rng.choice(groups, size=len(groups), replace=True)
                sampled_seed_means.append(float(np.mean([group_values[str(group)] for group in sampled_groups])))
            estimates.append(float(np.mean(sampled_seed_means)))
        alpha = 1 - cfg.statistics.confidence_level
        shared_groups = sorted(next(iter(group_sets)))
        group_effects = [
            float(np.mean([by_seed_group[seed][group] for seed in selected_seeds]))
            for group in shared_groups
        ]
        row: dict[str, Any] = {
            "left_model": pair.left, "right_model": pair.right,
            "model": f"{pair.right}_minus_{pair.left}", "effect_direction": "right_minus_left",
            "effect": point,
            "ci": [
                float(np.quantile(estimates, alpha / 2)),
                float(np.quantile(estimates, 1 - alpha / 2)),
            ],
            "seed_count": len(selected_seeds), "family_count": len(families),
            "clusters": len(shared_groups), "replicates": cfg.statistics.bootstrap_samples,
            "uncertainty_source": "training_seed_and_original_image_cluster_bootstrap",
            "inference_unit": "source_group_id_averaged_across_families_and_frozen_seeds",
        }
        try:
            sensitivity = wilcoxon_sensitivity(group_effects, [0.0] * len(group_effects))
        except MetricUndefinedError as exc:
            row.update({
                "primary_test_defined": False, "primary_test_reason": str(exc),
                "p_value": None, "holm_adjusted_p": None, "holm_rejected": False,
            })
        else:
            row.update({"primary_test_defined": True, **sensitivity})
            valid_rows.append(row)
            valid_p_values.append(float(sensitivity["p_value"]))
        summaries.append(row)
    if valid_p_values:
        correction = holm_bonferroni(valid_p_values, cfg.statistics.primary_familywise_alpha)
        for row, adjusted, rejected in zip(
            valid_rows, correction["adjusted_p_values"], correction["rejected"], strict=True,
        ):
            row["holm_adjusted_p"] = adjusted
            row["holm_rejected"] = rejected
            row["holm_family_test_count"] = correction["test_count"]
            row["holm_familywise_alpha"] = correction["familywise_alpha"]
    return summaries


def generate_primary_statistics(
    cfg: Config,
    evaluation_dirs: Sequence[str | Path],
    output: str | Path,
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selected_models, selected_seeds, selected_pairs = protocol_matrix(protocol, cfg)
    expected_runs = {f"{model}-seed{seed}" for model in selected_models for seed in selected_seeds}
    if (protocol.get("config_sha256") != config_digest(cfg)
            or set(protocol.get("resolved_run_sha256", {})) != expected_runs):
        raise ValueError("primary statistics require the real matching experiment protocol")
    run_rows: dict[str, list[dict[str, Any]]] = {}
    detector_artifacts: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for value in evaluation_dirs:
        directory = Path(value)
        provenance_path = directory / "provenance.json"
        pair_path = directory / "pair_metrics.jsonl"
        metrics_path = directory / "metrics.json"
        representations_path = directory / "representations.npz"
        index_path = directory / "evaluation_index.json"
        detector_path = directory / "detector_metrics.json"
        if any(not path.is_file() for path in (provenance_path, pair_path, metrics_path, representations_path, index_path, detector_path)):
            raise FileNotFoundError(f"final statistical input is incomplete: {directory}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (index.get("schema_version") != "sac-evaluation-index-v1" or index.get("protocol_sha256") != protocol_sha256
                or index.get("provenance_sha256") != sha256_file(provenance_path)
                or index.get("pair_metrics_sha256") != sha256_file(pair_path)
                or index.get("metrics_sha256") != sha256_file(metrics_path)
                or index.get("representations_sha256") != sha256_file(representations_path)):
            raise ValueError(f"evaluation index does not authenticate all final outputs: {directory}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("schema_version") != "sac-evaluation-v2" or provenance.get("split_role") != "final_test":
            raise ValueError("primary statistics require final-test evaluation-v2 artifacts")
        run_key = f"{provenance['model_name']}-seed{provenance['training_seed']}"
        if run_key in run_rows or run_key not in expected_runs:
            raise ValueError(f"duplicate or unexpected final evaluation: {run_key}")
        if (provenance.get("protocol_sha256") != protocol_sha256
                or provenance.get("config_sha256") != protocol["config_sha256"]
                or provenance.get("resolved_run_sha256") != protocol["resolved_run_sha256"][run_key]
                or provenance.get("checkpoint_sha256") != protocol["checkpoint_sha256"][run_key]
                or provenance.get("pair_manifest_sha256") != protocol["pair_manifest_sha256"]["final_test"]
                or provenance.get("review_freeze_sha256") != protocol["review_freeze_sha256"]
                or provenance.get("condition_signature") != protocol["condition_signature"]):
            raise ValueError(f"final evaluation differs from the real protocol: {run_key}")
        rows = read_jsonl(pair_path)
        run_rows[run_key] = rows
        detector = json.loads(detector_path.read_text(encoding="utf-8"))
        if (detector.get("schema_version") != "sac-detector-evaluation-v2"
                or detector.get("model_name") != provenance["model_name"]
                or detector.get("training_seed") != provenance["training_seed"]
                or detector.get("protocol_sha256") != protocol_sha256
                or detector.get("population", {}).get("sham_count") != 0):
            raise ValueError(f"detector evaluation provenance mismatch: {run_key}")
        detector_artifacts[run_key] = detector
        sources.append({
            "run": run_key, "provenance_sha256": sha256_file(provenance_path),
            "pairs_sha256": sha256_file(pair_path), "metrics_sha256": sha256_file(metrics_path),
            "representations_sha256": sha256_file(representations_path), "evaluation_index_sha256": sha256_file(index_path),
            "detector_sha256": sha256_file(detector_path),
        })
    if set(run_rows) != expected_runs:
        raise ValueError("primary statistics require the complete configured run matrix under the real experiment freeze")
    families = [family for family in ("spectral_appearance", "texture", "resolution") if all(any(row.get("audit_qualified") is True and row["family"] == family and row["severity"] == severity for row in rows) for rows in run_rows.values() for severity in ("mild", "severe"))]
    if not families:
        raise MetricUndefinedError("no family has complete accepted mild/severe support across all model seeds")
    primary: list[dict[str, Any]] = []
    difference_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    contrast_specs = (
        ("rq1_label_inconsistency_severe_minus_mild", "label_consistency", True),
        ("rq1_js_severe_minus_mild", "js_divergence", False),
        ("rq2_normalized_instability_severe_minus_mild", "normalized_representation_instability", False),
    )
    for run_key, rows in sorted(run_rows.items()):
        model, seed_text = run_key.rsplit("-seed", 1)
        seed = int(seed_text)
        for family in families:
            for contrast, metric, label_inconsistency in contrast_specs:
                differences = _severity_differences(rows, family, metric, label_inconsistency=label_inconsistency)
                if contrast == "rq2_normalized_instability_severe_minus_mild":
                    difference_cache[(run_key, family)] = differences
                result = _clustered_effect([row["effect"] for row in differences], [row["source_group_id"] for row in differences], cfg.statistics.bootstrap_samples, cfg.statistics.confidence_level, cfg.statistics.seed + seed)
                primary.append({"contrast": contrast, "model": model, "seed": seed, "family": family, **result})
    for seed in selected_seeds:
        for family_index, family in enumerate(families):
            family_rows: list[dict[str, Any]] = []
            valid_p_values: list[float] = []
            valid_p_rows: list[dict[str, Any]] = []
            for pair_index, pair in enumerate(selected_pairs):
                left = {
                    row["source_id"]: row
                    for row in difference_cache[(f"{pair.left}-seed{seed}", family)]
                }
                right = {
                    row["source_id"]: row
                    for row in difference_cache[(f"{pair.right}-seed{seed}", family)]
                }
                if set(left) != set(right):
                    raise ValueError("cross-model difference-in-differences requires identical source support")
                values: list[float] = []
                groups: list[str] = []
                for source_id in sorted(left):
                    if left[source_id]["source_group_id"] != right[source_id]["source_group_id"]:
                        raise ValueError("cross-model original group mismatch")
                    values.append(float(right[source_id]["effect"]) - float(left[source_id]["effect"]))
                    groups.append(str(left[source_id]["source_group_id"]))
                result = _clustered_effect(
                    values, groups, cfg.statistics.bootstrap_samples, cfg.statistics.confidence_level,
                    cfg.statistics.seed + seed + 1000 + family_index * 100 + pair_index,
                )
                row = {
                    "contrast": "rq2_cross_model_difference_in_differences",
                    "model": f"{pair.right}_minus_{pair.left}",
                    "left_model": pair.left, "right_model": pair.right,
                    "effect_direction": "right_minus_left", "seed": seed, "family": family,
                    **result,
                }
                effects_by_group: dict[str, list[float]] = defaultdict(list)
                for value, group in zip(values, groups, strict=True):
                    effects_by_group[group].append(value)
                group_effects = [float(np.mean(effects_by_group[group])) for group in sorted(effects_by_group)]
                try:
                    sensitivity = wilcoxon_sensitivity(group_effects, [0.0] * len(group_effects))
                except MetricUndefinedError as exc:
                    row.update({
                        "primary_test_defined": False, "primary_test_reason": str(exc),
                        "p_value": None, "holm_adjusted_p": None, "holm_rejected": False,
                        "cluster_pairs": len(group_effects),
                    })
                else:
                    row.update({"primary_test_defined": True, "cluster_pairs": len(group_effects), **sensitivity})
                    valid_p_values.append(float(sensitivity["p_value"]))
                    valid_p_rows.append(row)
                family_rows.append(row)
            if valid_p_values:
                correction = holm_bonferroni(valid_p_values, cfg.statistics.primary_familywise_alpha)
                for row, adjusted_p, rejected in zip(
                    valid_p_rows, correction["adjusted_p_values"], correction["rejected"], strict=True,
                ):
                    row["holm_adjusted_p"] = adjusted_p
                    row["holm_rejected"] = rejected
                    row["holm_family_test_count"] = correction["test_count"]
                    row["holm_familywise_alpha"] = correction["familywise_alpha"]
            primary.extend(family_rows)
    rq3_absolute: list[dict[str, Any]] = []
    for run_key, detector in sorted(detector_artifacts.items()):
        model, seed_text = run_key.rsplit("-seed", 1)
        seed = int(seed_text)
        population_rows = detector.get("population_rows", [])
        detector_metrics = detector.get("models", {})
        population = detector.get("population", {})
        learned_names = ("output_only", "output_plus_representation")
        population_adequate = (
            population.get("positives", 0) >= cfg.evaluation.rq3_min_positives_per_model_seed
            and population.get("positives", 0) < population.get("count", 0)
            and bool(population_rows)
        )
        learned_adequate = population_adequate and all(
            detector_metrics.get(name, {}).get("defined") is True for name in learned_names
        ) and all(all(name in row.get("scores", {}) for name in learned_names) for row in population_rows)
        score_names = sorted({name for row in population_rows for name in row.get("scores", {})})
        for score_index, score_name in enumerate(score_names):
            for metric_index, metric in enumerate(("average_precision", "auroc")):
                bootstrap_seed = cfg.statistics.seed + seed + 3000 + score_index * 10 + metric_index
                score_adequate = population_adequate and all(score_name in row.get("scores", {}) for row in population_rows)
                base = {
                    "model": model, "seed": seed, "score": score_name, "metric": metric,
                    "count": population.get("count"), "positives": population.get("positives"),
                    "prevalence": population.get("prevalence"), "underpowered": not score_adequate,
                    "bootstrap_seed": bootstrap_seed,
                }
                if not score_adequate:
                    rq3_absolute.append({**base, "defined": False, "bootstrap_state": "underpowered", "reason": detector.get("omission_reason") or "final-test RQ3 population is underpowered, single-class, or lacks the score"})
                    continue
                try:
                    result = clustered_detector_metric(
                        [row["target"] for row in population_rows],
                        [row["scores"][score_name] for row in population_rows],
                        [row["source_group_id"] for row in population_rows], metric,
                        cfg.statistics.bootstrap_samples, cfg.statistics.confidence_level, bootstrap_seed,
                    )
                except MetricUndefinedError as exc:
                    result = {"defined": False, "reason": str(exc)}
                rq3_absolute.append({**base, "bootstrap_state": "defined" if result.get("defined") else "undefined", **result})
        for metric, contrast in (("average_precision", "rq3_nested_detector_delta_average_precision"), ("auroc", "rq3_nested_detector_delta_auroc")):
            if not learned_adequate:
                primary.append({"contrast": contrast, "model": model, "seed": seed, "family": "all_accepted_non_sham", "defined": False, "reason": detector.get("omission_reason") or "final-test RQ3 population is underpowered or single-class"})
                continue
            try:
                result = clustered_detector_delta(
                    [row["target"] for row in population_rows],
                    [row["scores"]["output_plus_representation"] for row in population_rows],
                    [row["scores"]["output_only"] for row in population_rows],
                    [row["source_group_id"] for row in population_rows], metric,
                    cfg.statistics.bootstrap_samples, cfg.statistics.confidence_level, cfg.statistics.seed + seed + 2000,
                )
            except MetricUndefinedError as exc:
                primary.append({"contrast": contrast, "model": model, "seed": seed, "family": "all_accepted_non_sham", "defined": False, "reason": str(exc)})
                continue
            primary.append({"contrast": contrast, "model": model, "seed": seed, "family": "all_accepted_non_sham", **result})
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        grouped_rows[(row["contrast"], row["model"], row["family"])].append(row)
    seed_summaries = []
    required_seeds = set(selected_seeds)
    for (contrast, model, family), rows in sorted(grouped_rows.items()):
        estimates = {int(row["seed"]): float(row["effect"]) for row in rows if row.get("defined", True) and "effect" in row}
        if set(estimates) != required_seeds:
            seed_summaries.append({"contrast": contrast, "model": model, "family": family, "defined": False, "available_seed_count": len(estimates), "reason": "all frozen seeds require defined per-seed effects"})
            continue
        seed_summaries.append({"contrast": contrast, "model": model, "family": family, "defined": True, **summarize_seed_estimates(estimates)})
    exploratory: list[dict[str, Any]] = []
    p_values: list[float] = []
    for run_key, rows in sorted(run_rows.items()):
        model, seed_text = run_key.rsplit("-seed", 1)
        for family in families:
            for metric in cfg.statistics.wilcoxon_metrics:
                differences = _severity_differences(rows, family, metric)
                effects_by_group: dict[str, list[float]] = defaultdict(list)
                for difference in differences:
                    effects_by_group[difference["source_group_id"]].append(float(difference["effect"]))
                group_effects = [float(np.mean(effects_by_group[group])) for group in sorted(effects_by_group)]
                try:
                    result = wilcoxon_sensitivity(group_effects, [0.0] * len(group_effects))
                    result["cluster_pairs"] = len(group_effects)
                except MetricUndefinedError as exc:
                    exploratory.append({"model": model, "seed": int(seed_text), "family": family, "metric": metric, "defined": False, "reason": str(exc)})
                    continue
                p_values.append(result["p_value"])
                exploratory.append({"model": model, "seed": int(seed_text), "family": family, "metric": metric, "defined": True, **result})
    if p_values:
        adjusted = benjamini_hochberg(p_values, cfg.statistics.fdr_q)
        index = 0
        for row in exploratory:
            if row["defined"]:
                row["bh_q_value"] = adjusted["adjusted_q_values"][index]
                row["bh_rejected"] = adjusted["rejected"][index]
                index += 1
    pairwise_summaries = _pairwise_model_summaries(
        cfg, difference_cache, families, seeds=selected_seeds, model_pairs=selected_pairs,
    )
    artifact = {
        "schema_version": "sac-primary-statistics-v3",
        "protocol_source": {"path": str(protocol_path), "sha256": protocol_sha256},
        "protocol_sha256": protocol_sha256,
        "cluster_unit": "source_group_id", "families": families, "sources": sources,
        "primary_per_seed": primary, "primary_seed_summaries": seed_summaries,
        "pairwise_model_summaries": pairwise_summaries,
        "rq3_absolute_clustered": rq3_absolute, "exploratory_bh": exploratory,
    }
    atomic_write_json(output, artifact)
    artifact["artifact_sha256"] = sha256_file(output)
    publish_statistics_to_wandb(cfg, output, artifact)
    return artifact
