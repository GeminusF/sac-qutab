from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import DetectorConfig
from .metrics import detection_metrics
from .runtime import atomic_write_json, sha256_file

OUTPUT_FEATURES = ("transformed_uncertainty", "transformed_entropy")
PLUS_FEATURES = OUTPUT_FEATURES + ("normalized_representation_instability",)


def _population(rows: Sequence[dict[str, Any]], required_role: str) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row.get("role") == required_role
        and row.get("audit_qualified") is True
        and row.get("clean_correct") is True
        and row.get("severity") != "sham"
    ]
    if not selected:
        raise ValueError(f"no accepted non-sham clean-correct rows for {required_role}")
    if any(not row.get("audit_condition") for row in selected):
        raise ValueError("RQ3 population requires accepted condition provenance")
    ids = [row["pair_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("detector population contains duplicate pair IDs")
    return sorted(selected, key=lambda row: row["pair_id"])


def _fit_one(rows: Sequence[dict[str, Any]], features: tuple[str, ...], cfg: DetectorConfig) -> dict[str, Any]:
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=float)
    y = np.asarray([row["counterfactual_failure"] for row in rows], dtype=int)
    if not np.all(np.isfinite(x)) or len(np.unique(y)) < 2:
        raise ValueError("detector calibration requires finite features and both target classes")
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    l1_ratio = {"l2": 0.0, "l1": 1.0}.get(cfg.penalty)
    if l1_ratio is None:
        raise ValueError(f"unsupported detector penalty: {cfg.penalty}")
    model = LogisticRegression(
        solver=cfg.solver,
        l1_ratio=l1_ratio,
        C=cfg.regularization_c,
        class_weight=cfg.class_weight,
        tol=cfg.tolerance,
        max_iter=cfg.max_iter,
        random_state=0,
    )
    model.fit(standardized, y)
    if int(model.n_iter_[0]) >= cfg.max_iter:
        raise RuntimeError("detector did not converge before max_iter")
    return {
        "features": list(features),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
    }


def fit_nested_detectors(
    rows: Sequence[dict[str, Any]],
    cfg: DetectorConfig,
    provenance: dict[str, Any],
    output: str | Path,
    min_positives: int,
) -> dict[str, Any]:
    if provenance.get("split_role") != "calibration":
        raise ValueError("combined detectors may be fit only from calibration artifacts")
    population = _population(rows, "calibration")
    targets = np.asarray([row["counterfactual_failure"] for row in population], dtype=int)
    positives = int(targets.sum())
    condition_counts = Counter(row["audit_condition"] for row in population)
    adequate = positives >= min_positives and len(np.unique(targets)) == 2
    artifact: dict[str, Any] = {
        "schema_version": "sac-detector-v3",
        "fit_split": "calibration",
        "model_name": provenance["model_name"],
        "training_seed": provenance["training_seed"],
        "config_sha256": provenance["config_sha256"],
        "condition_signature": provenance["condition_signature"],
        "calibration_pair_manifest_sha256": provenance["pair_manifest_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "resolved_run_sha256": provenance["resolved_run_sha256"],
        "review_freeze_sha256": provenance["review_freeze_sha256"],
        "control_ecdf_sha256": provenance["control_ecdf_sha256"],
        "training_wandb_run_id": provenance.get("training_wandb_run_id"),
        "feature_contract": {"output_only": list(OUTPUT_FEATURES), "output_plus_representation": list(PLUS_FEATURES)},
        "fit_configuration": {
            "solver": cfg.solver, "penalty": cfg.penalty, "regularization_c": cfg.regularization_c,
            "class_weight": cfg.class_weight, "tolerance": cfg.tolerance, "max_iter": cfg.max_iter,
            "min_positives": min_positives,
        },
        "pair_ids": [row["pair_id"] for row in population],
        "targets": targets.tolist(),
        "fit_population": {
            "count": len(population), "positives": positives, "prevalence": positives / len(population),
            "condition_counts": dict(sorted(condition_counts.items())), "sham_count": 0,
        },
        "learned_detectors_defined": adequate,
        "omission_reason": None if adequate else f"calibration requires both classes and at least {min_positives} positives",
        "models": {},
    }
    if adequate:
        artifact["models"] = {
            "output_only": _fit_one(population, OUTPUT_FEATURES, cfg),
            "output_plus_representation": _fit_one(population, PLUS_FEATURES, cfg),
        }
    atomic_write_json(output, artifact)
    artifact["artifact_sha256"] = sha256_file(output)
    return artifact


def _score(rows: Sequence[dict[str, Any]], model: dict[str, Any]) -> np.ndarray:
    x = np.asarray([[row[name] for name in model["features"]] for row in rows], dtype=float)
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    coefficient, intercept = np.asarray(model["coefficient"]), float(model["intercept"])
    logits = ((x - mean) / scale) @ coefficient + intercept
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def _verify_frozen_detector(
    detector: dict[str, Any],
    detector_path: str | Path,
    protocol_path: str | Path,
    provenance: dict[str, Any],
) -> None:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    run_key = f"{provenance['model_name']}-seed{provenance['training_seed']}"
    if protocol.get("schema_version") not in {"sac-experiment-protocol-v4", "sac-experiment-protocol-v5"}:
        raise ValueError("detector scoring requires the experiment-level final protocol")
    if protocol.get("detector_sha256", {}).get(run_key) != sha256_file(detector_path):
        raise ValueError("detector differs from the experiment-level freeze")
    if provenance.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("detector evaluation provenance does not match the final protocol")
    if detector.get("control_ecdf_sha256") != protocol.get("control_ecdf_sha256", {}).get(run_key):
        raise ValueError("detector control ECDF differs from the experiment-level freeze")


def evaluate_nested_detectors(
    rows: Sequence[dict[str, Any]],
    detector: dict[str, Any],
    provenance: dict[str, Any],
    min_positives: int,
    *,
    detector_path: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    if detector.get("fit_split") != "calibration" or provenance.get("split_role") != "final_test":
        raise ValueError("detector evaluation requires calibration fit and final_test evaluation")
    for key in ("model_name", "training_seed", "config_sha256", "condition_signature", "checkpoint_sha256", "resolved_run_sha256", "review_freeze_sha256", "control_ecdf_sha256"):
        if detector[key] != provenance[key]:
            raise ValueError(f"detector/test provenance mismatch: {key}")
    _verify_frozen_detector(detector, detector_path, protocol_path, provenance)
    population = _population(rows, "final_test")
    y = np.asarray([row["counterfactual_failure"] for row in population], dtype=int)
    scores: dict[str, np.ndarray] = {
        "confidence_only": np.asarray([row["transformed_uncertainty"] for row in population], dtype=float),
        "entropy_only": np.asarray([row["transformed_entropy"] for row in population], dtype=float),
        "representation_only": np.asarray([row["normalized_representation_instability"] for row in population], dtype=float),
    }
    if detector.get("learned_detectors_defined"):
        models = detector.get("models", {})
        scores["output_only"] = _score(population, models["output_only"])
        scores["output_plus_representation"] = _score(population, models["output_plus_representation"])
    metrics = {name: detection_metrics(y, values, min_positives) for name, values in scores.items()}
    population_rows = []
    for index, row in enumerate(population):
        population_rows.append({
            "pair_id": row["pair_id"], "source_id": row["source_id"], "source_group_id": row["source_group_id"],
            "family": row["family"], "severity": row["severity"], "class_name": row["class_name"],
            "target": int(y[index]), "scores": {name: float(values[index]) for name, values in scores.items()},
        })
    condition_prevalence: dict[str, Any] = {}
    for condition in sorted({f"{row['family']}:{row['severity']}" for row in population}):
        selected = [row for row in population_rows if f"{row['family']}:{row['severity']}" == condition]
        positives = sum(row["target"] for row in selected)
        condition_prevalence[condition] = {"count": len(selected), "positives": positives, "prevalence": positives / len(selected)}
    return {
        "schema_version": "sac-detector-evaluation-v2",
        "model_name": provenance["model_name"],
        "training_seed": provenance["training_seed"],
        "protocol_sha256": provenance["protocol_sha256"],
        "training_wandb_run_id": provenance.get("training_wandb_run_id"),
        "population": {"count": len(population), "positives": int(y.sum()), "prevalence": float(y.mean()), "sham_count": 0},
        "condition_prevalence": condition_prevalence,
        "learned_detector_omitted": not detector.get("learned_detectors_defined", False),
        "omission_reason": detector.get("omission_reason"),
        "models": metrics,
        "population_rows": population_rows,
    }


def load_detector(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "sac-detector-v3":
        raise ValueError("unsupported detector artifact")
    return artifact
