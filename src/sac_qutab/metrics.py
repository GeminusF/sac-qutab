from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


class MetricUndefinedError(ValueError):
    pass


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if array.size == 0:
        raise MetricUndefinedError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def validate_probabilities(probabilities: Any) -> np.ndarray:
    p = _array(probabilities, "probabilities", 2).astype(np.float64)
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must be in [0,1] and sum to one")
    return p


def classification_metrics(probabilities: Any, targets: Any, num_classes: int) -> dict[str, Any]:
    p = validate_probabilities(probabilities)
    y = _array(targets, "targets", 1).astype(int)
    if len(y) != len(p) or np.any((y < 0) | (y >= num_classes)):
        raise ValueError("invalid classification targets")
    predictions = p.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, predictions, labels=np.arange(num_classes), zero_division=0)
    return {
        "count": len(y),
        "accuracy": float(np.mean(predictions == y)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": confusion_matrix(y, predictions, labels=np.arange(num_classes)).tolist(),
    }


def entropy(probabilities: Any) -> np.ndarray:
    p = validate_probabilities(probabilities)
    safe = np.clip(p, np.finfo(np.float64).tiny, 1.0)
    return -np.sum(p * np.log(safe), axis=1)


def js_divergence(p_values: Any, q_values: Any) -> np.ndarray:
    p, q = validate_probabilities(p_values), validate_probabilities(q_values)
    if p.shape != q.shape:
        raise ValueError("JS inputs must have identical shape")
    m = 0.5 * (p + q)
    tiny = np.finfo(np.float64).tiny
    result = 0.5 * np.sum(p * np.log(np.clip(p, tiny, 1) / np.clip(m, tiny, 1)), axis=1) + 0.5 * np.sum(q * np.log(np.clip(q, tiny, 1) / np.clip(m, tiny, 1)), axis=1)
    if np.any(result < -1e-5) or np.any(result > math.log(2) + 1e-5):
        raise RuntimeError("JS divergence escaped its natural-log bounds")
    return np.clip(result, 0.0, math.log(2))


def kl_diagnostic(p_values: Any, q_values: Any) -> np.ndarray:
    p, q = validate_probabilities(p_values), validate_probabilities(q_values)
    if p.shape != q.shape:
        raise ValueError("KL inputs must have identical shape")
    tiny = np.finfo(np.float64).tiny
    return np.sum(p * np.log(np.clip(p, tiny, 1) / np.clip(q, tiny, 1)), axis=1)


def cosine_similarity(a_values: Any, b_values: Any) -> np.ndarray:
    a, b = _array(a_values, "features_a", 2).astype(np.float64), _array(b_values, "features_b", 2).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("pairwise cosine inputs must have identical shape")
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(denominator <= 0):
        raise MetricUndefinedError("cosine is undefined for zero-norm representations")
    return np.sum(a * b, axis=1) / denominator


def linear_cka(x_values: Any, y_values: Any) -> float:
    x, y = _array(x_values, "cka_x", 2).astype(np.float64), _array(y_values, "cka_y", 2).astype(np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise MetricUndefinedError("CKA requires matching sample count >= 2")
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx, yy = x.T @ x, y.T @ y
    denominator = np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro")
    if denominator <= 0:
        raise MetricUndefinedError("CKA is undefined for constant representations")
    return float(np.linalg.norm(cross, "fro") ** 2 / denominator)


def expected_calibration_error(probabilities: Any, targets: Any, bins: int) -> dict[str, Any]:
    p = validate_probabilities(probabilities)
    y = _array(targets, "targets", 1).astype(int)
    if len(y) != len(p) or bins < 2:
        raise ValueError("invalid ECE targets or bin count")
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, bins - 1)
    assignments = np.maximum(assignments, 0)
    rows = []
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_confidence)
        rows.append({"index": index, "lower": float(edges[index]), "upper": float(edges[index + 1]), "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    return {"ece": float(ece), "bins": rows}


def brier_score(probabilities: Any, targets: Any) -> float:
    p = validate_probabilities(probabilities)
    y = _array(targets, "targets", 1).astype(int)
    if len(y) != len(p):
        raise ValueError("Brier targets mismatch")
    one_hot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probabilities: Any, targets: Any) -> float:
    p = validate_probabilities(probabilities)
    y = _array(targets, "targets", 1).astype(int)
    if len(y) != len(p):
        raise ValueError("NLL targets mismatch")
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], np.finfo(np.float64).tiny, 1))))


def fit_control_ecdf(negative_control_dissimilarity: Any) -> dict[str, Any]:
    values = np.sort(_array(negative_control_dissimilarity, "negative_control_dissimilarity", 1).astype(np.float64))
    return {"values": values.tolist(), "count": len(values)}


def apply_control_ecdf(dissimilarity: Any, ecdf: dict[str, Any]) -> np.ndarray:
    query = _array(dissimilarity, "dissimilarity", 1).astype(np.float64)
    values = np.asarray(ecdf["values"], dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise MetricUndefinedError("validation control ECDF is empty or invalid")
    return np.searchsorted(values, query, side="right") / len(values)


def counterfactual_pair_metrics(original_prob: Any, transformed_prob: Any, targets: Any, original_features: Any, transformed_features: Any, control_features: Any, normalized_instability: Any | None = None) -> dict[str, np.ndarray]:
    p, q = validate_probabilities(original_prob), validate_probabilities(transformed_prob)
    y = _array(targets, "targets", 1).astype(int)
    if p.shape != q.shape or len(y) != len(p):
        raise ValueError("pair probability/target mismatch")
    pred_p, pred_q = p.argmax(axis=1), q.argmax(axis=1)
    clean_correct = pred_p == y
    transformed_correct = pred_q == y
    cos_cf = cosine_similarity(original_features, transformed_features)
    cos_control = cosine_similarity(original_features, control_features)
    confidence_change = p.max(axis=1) - q.max(axis=1)
    result = {
        "clean_correct": clean_correct,
        "transformed_correct": transformed_correct,
        "label_consistency": pred_p == pred_q,
        "counterfactual_failure": clean_correct & ~transformed_correct,
        "confidence_change": confidence_change,
        "absolute_confidence_change": np.abs(confidence_change),
        "transformed_entropy": entropy(q),
        "js_divergence": js_divergence(p, q),
        "cosine_similarity": cos_cf,
        "control_cosine_similarity": cos_control,
        "control_margin": cos_cf - cos_control,
        "representation_instability": 1 - cos_cf,
    }
    if normalized_instability is not None:
        normalized = _array(normalized_instability, "normalized_instability", 1)
        if len(normalized) != len(y):
            raise ValueError("normalized instability length mismatch")
        result["normalized_representation_instability"] = normalized
    return result


def summarize_confidence_change(change: Any, failures: Any) -> dict[str, float]:
    delta = _array(change, "confidence_change", 1).astype(float)
    failure = _array(failures, "failures", 1).astype(bool)
    if len(delta) != len(failure):
        raise ValueError("confidence/failure length mismatch")
    failure_count = int(failure.sum())
    return {
        "mean_signed": float(delta.mean()),
        "mean_absolute": float(np.abs(delta).mean()),
        "median": float(np.median(delta)),
        "q25": float(np.quantile(delta, 0.25)),
        "q75": float(np.quantile(delta, 0.75)),
        "increased_confidence_failure_fraction": float(np.mean(delta[failure] < 0)) if failure_count else math.nan,
    }


def detection_metrics(targets: Any, scores: Any, min_positives: int) -> dict[str, Any]:
    y = _array(targets, "detector_targets", 1).astype(int)
    s = _array(scores, "detector_scores", 1).astype(float)
    if len(y) != len(s) or set(np.unique(y)) - {0, 1}:
        raise ValueError("detector targets/scores mismatch or non-binary targets")
    positives = int(y.sum())
    prevalence = positives / len(y)
    if len(np.unique(y)) < 2:
        return {"defined": False, "underpowered": True, "reason": "single target class", "count": len(y), "positives": positives, "prevalence": prevalence}
    estimates = {"auroc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s))}
    if positives < min_positives:
        return {
            "defined": False, "underpowered": True, "reason": f"fewer than {min_positives} positives",
            "count": len(y), "positives": positives, "prevalence": prevalence, "exploratory_estimates": estimates,
        }
    return {
        "defined": True, "underpowered": False, "reason": None, "count": len(y),
        "positives": positives, "prevalence": prevalence, **estimates,
    }
