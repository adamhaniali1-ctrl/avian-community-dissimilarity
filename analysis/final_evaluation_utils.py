from __future__ import annotations

import hashlib

import math

import os

import tempfile

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import kendalltau, spearmanr

HOLDOUTS = ("india", "malaysia_safe", "taiwan", "usa")

CONDITIONS = ("filtered", "author_compatible")

PREDICTIVE_METHODS = ("training_mean", "training_median", "ssi_isotonic", "gdm", "pairwise_random_forest_ssi", "siamese_ensemble_ssi")

RANKING_METHODS = ("ssi_raw_euclidean", "ssi_isotonic", "gdm", "pairwise_random_forest_ssi", "siamese_ensemble_ssi")

ALL_METHODS = ("training_mean", "training_median", "ssi_raw_euclidean", "ssi_isotonic", "gdm", "pairwise_random_forest_ssi", "siamese_ensemble_ssi")

SSI_COLUMNS = [f"ssi_{index:03d}" for index in range(60)]

def sha256_file(path: Path) -> str:
    """Return one SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic CSV atomically from system temporary storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="final_evaluation_", suffix=".tmp")
    os.close(descriptor); temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g", na_rep="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_text(path: Path, text: str) -> None:
    """Write deterministic text atomically."""
    if not text.endswith("\n"): text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="final_evaluation_", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def rank_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, str, float, str]:
    """Return Spearman and Kendall tau-b with explicit undefined reasons."""
    if np.unique(prediction).size < 2: return math.nan, "constant_prediction", math.nan, "constant_prediction"
    if np.unique(target).size < 2: return math.nan, "constant_target", math.nan, "constant_target"
    spearman = float(spearmanr(target, prediction).statistic); kendall = float(kendalltau(target, prediction, variant="b").statistic)
    return spearman, "" if math.isfinite(spearman) else "undefined_spearman", kendall, "" if math.isfinite(kendall) else "undefined_kendall"

def calibration(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float, str]:
    """Return descriptive OLS calibration intercept and slope."""
    centered = prediction - np.mean(prediction); denominator = float(np.sum(centered * centered))
    if denominator == 0: return math.nan, math.nan, "constant_prediction"
    slope = float(np.sum(centered * (target - np.mean(target))) / denominator)
    return float(np.mean(target) - slope * np.mean(prediction)), slope, ""

def point_metrics(target: np.ndarray, prediction: np.ndarray, predictive: bool) -> dict[str, tuple[float, str]]:
    """Calculate locked dataset-level metrics using population standard deviations."""
    target = np.asarray(target, dtype=np.float64); prediction = np.asarray(prediction, dtype=np.float64)
    spearman, sr, kendall, kr = rank_metrics(target, prediction); tsd = float(np.std(target, ddof=0)); psd = float(np.std(prediction, ddof=0)); intercept, slope, cr = calibration(target, prediction)
    values: dict[str, tuple[float, str]] = {
        "spearman": (spearman, sr), "kendall_tau_b": (kendall, kr),
        "prediction_minimum": (float(np.min(prediction)), ""), "prediction_maximum": (float(np.max(prediction)), ""),
        "prediction_mean": (float(np.mean(prediction)), ""), "prediction_standard_deviation": (psd, ""),
        "target_minimum": (float(np.min(target)), ""), "target_maximum": (float(np.max(target)), ""),
        "target_mean": (float(np.mean(target)), ""), "target_standard_deviation": (tsd, ""),
        "prediction_sd_divided_by_target_sd": (psd / tsd if tsd > 0 else math.nan, "" if tsd > 0 else "constant_target"),
        "sample_count": (float(len(target)), ""),
    }
    if predictive:
        errors = target - prediction; denominator = float(np.sum((target - np.mean(target)) ** 2))
        values.update({
            "mae": (float(np.mean(np.abs(errors))), ""), "rmse": (float(np.sqrt(np.mean(errors ** 2))), ""),
            "r2": (1.0 - float(np.sum(errors ** 2)) / denominator if denominator > 0 else math.nan, "" if denominator > 0 else "constant_target"),
            "calibration_intercept": (intercept, cr), "calibration_slope": (slope, cr),
        })
    return values

def weighted_mae(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    """Return integer-replication-equivalent weighted MAE."""
    total = float(np.sum(weights)); return math.nan if total <= 0 else float(np.sum(weights * np.abs(target - prediction)) / total)

def weighted_midrank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return average ranks in the conceptual integer-replicated sample."""
    _, inverse = np.unique(values, return_inverse=True); group_weight = np.bincount(inverse, weights=weights); lower = np.cumsum(group_weight) - group_weight; return (lower + (group_weight + 1.0) / 2.0)[inverse]

def weighted_spearman(target: np.ndarray, score: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted-midrank Spearman equivalent to explicit row replication."""
    positive = weights > 0
    if not np.any(positive): return math.nan
    y, x, w = target[positive], score[positive], weights[positive].astype(np.float64)
    if np.unique(y).size < 2 or np.unique(x).size < 2: return math.nan
    ry, rx = weighted_midrank(y, w), weighted_midrank(x, w); total = float(np.sum(w)); my = float(np.sum(w * ry) / total); mx = float(np.sum(w * rx) / total); dy, dx = ry - my, rx - mx; denominator = math.sqrt(float(np.sum(w * dy * dy) * np.sum(w * dx * dx)))
    return math.nan if denominator == 0 else float(np.sum(w * dy * dx) / denominator)

def fixture_check() -> None:
    """Verify weighted statistics against explicit integer replication."""
    target = np.array([0.0, 0.5, 1.0, 0.5]); score = np.array([0.2, 0.8, 0.7, 0.8]); prediction = np.array([0.1, 0.4, 0.9, 0.6]); weights = np.array([2, 6, 3, 4])
    repeated_target = np.repeat(target, weights); repeated_score = np.repeat(score, weights); repeated_prediction = np.repeat(prediction, weights)
    expected_mae = float(np.mean(np.abs(repeated_target - repeated_prediction))); expected_spearman = float(spearmanr(repeated_target, repeated_score).statistic)
    if not math.isclose(weighted_mae(target, prediction, weights), expected_mae, abs_tol=1e-15, rel_tol=0): raise RuntimeError("Weighted MAE fixture failed")
    if not math.isclose(weighted_spearman(target, score, weights), expected_spearman, abs_tol=1e-15, rel_tol=0): raise RuntimeError("Weighted Spearman fixture failed")

def group_seed(holdout: str, condition: str, role: str, dataset: str) -> int:
    """Return the established deterministic event-node stream seed."""
    text = f"42|event_node_bootstrap|{holdout}|{condition}|{role}|{dataset}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)

def bootstrap_group(frame: pd.DataFrame, values: dict[str, np.ndarray], replicates: int = 2000) -> tuple[dict[tuple[str, str], np.ndarray], int]:
    """Run one established event-node bootstrap stream shared across methods."""
    holdout = str(frame.outer_holdout.iloc[0]); condition = str(frame.data_condition.iloc[0]); role = str(frame.pair_role.iloc[0]); dataset = str(frame.dataset.iloc[0]); events = sorted(set(frame.event_uid_a) | set(frame.event_uid_b)); index = {event: i for i, event in enumerate(events)}; a = frame.event_uid_a.map(index).to_numpy(np.int64); b = frame.event_uid_b.map(index).to_numpy(np.int64); target = frame.true_jaccard_distance.to_numpy(np.float64)
    output = {(method, "mae"): np.full(replicates, np.nan) for method in PREDICTIVE_METHODS}; output.update({(method, "spearman"): np.full(replicates, np.nan) for method in RANKING_METHODS})
    rng = np.random.default_rng(group_seed(holdout, condition, role, dataset))
    for replicate in range(replicates):
        multiplicity = np.bincount(rng.integers(0, len(events), size=len(events)), minlength=len(events)); weights = multiplicity[a] * multiplicity[b]
        for method in PREDICTIVE_METHODS: output[(method, "mae")][replicate] = weighted_mae(target, values[method], weights)
        for method in RANKING_METHODS: output[(method, "spearman")][replicate] = weighted_spearman(target, values[method], weights)
    return output, len(events)

def percentile(values: np.ndarray) -> tuple[float, float, int, int, str]:
    """Return locked linear percentile bounds and replicate counts."""
    valid = values[np.isfinite(values)]
    if not len(valid): return math.nan, math.nan, 0, len(values), "all_replicates_undefined"
    limits = np.quantile(valid, [0.025, 0.975], method="linear")
    return float(limits[0]), float(limits[1]), len(valid), len(values) - len(valid), ""

def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Return the exact float64 empirical-CDF integral."""
    first = np.sort(np.asarray(left, np.float64)); second = np.sort(np.asarray(right, np.float64)); support = np.sort(np.concatenate((first, second))); delta = np.diff(support)
    if not len(delta): return 0.0
    return float(np.sum(np.abs(np.searchsorted(first, support[:-1], side="right") / len(first) - np.searchsorted(second, support[:-1], side="right") / len(second)) * delta))

def compression_label(ratio: float) -> str:
    """Apply the locked project-specific SD-ratio labels."""
    if not math.isfinite(ratio): return "undefined_target_sd"
    if ratio < 0.10: return "severe_compression"
    if ratio < 0.50: return "material_compression"
    return "not_compressed_by_prespecified_rule"
