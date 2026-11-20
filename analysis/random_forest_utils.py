from __future__ import annotations

import functools

import math

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import ParameterGrid

from sklearn.preprocessing import StandardScaler

DATASETS = ("india", "malaysia_safe", "taiwan", "usa")

CONDITIONS = ("author_compatible", "filtered")

FEATURE_REGIMES = ("SSI", "LF", "combined")

FEATURE_FAMILIES = ("SSI", "LF")

METRICS = ("mae", "rmse", "r2", "spearman")

STAGE_NEW_EXACT_PATHS = {
    "config/random_forest_config.yaml",
    "scripts/random_forest_utils.py",
    "scripts/run_pairwise_random_forest.py",
    "scripts/validate_pairwise_random_forest.py",
    "audit_reports/pairwise_random_forest_audit.md",
}

STAGE_NEW_DIRECTORY = "results/pairwise_random_forest"

def feature_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Verify and return the exact SSI and LF feature families."""
    expected_ssi = [f"ssi_{index:03d}" for index in range(60)]
    expected_lf = [f"lf_{index:03d}" for index in range(128)]
    actual_ssi = [column for column in frame.columns if column.startswith("ssi_")]
    actual_lf = [column for column in frame.columns if column.startswith("lf_")]
    if actual_ssi != expected_ssi or actual_lf != expected_lf:
        raise ValueError("The central feature families differ from the audited 60 SSI and 128 LF columns")
    values = frame[[*expected_ssi, *expected_lf]].to_numpy(dtype=float)
    if values.shape[1] != 188 or not np.isfinite(values).all():
        raise ValueError("The central table must contain exactly 188 finite acoustic features")
    return {"SSI": expected_ssi, "LF": expected_lf}

def index_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Index the central table by globally unique event UID."""
    if "event_uid" not in frame or frame["event_uid"].duplicated().any():
        raise ValueError("event_features.csv must contain globally unique event_uid values")
    return frame.set_index("event_uid", drop=False)

def endpoint_arrays(
    pairs: pd.DataFrame,
    feature_index: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Join pair endpoints to one verified feature family."""
    ids_a = pairs["event_uid_a"].astype(str)
    ids_b = pairs["event_uid_b"].astype(str)
    known = set(feature_index.index.astype(str))
    missing = (set(ids_a) | set(ids_b)) - known
    if missing:
        raise ValueError(f"Pair endpoints are absent from the central feature table: {sorted(missing)[:10]}")
    left = feature_index.loc[ids_a, columns].to_numpy(dtype=float)
    right = feature_index.loc[ids_b, columns].to_numpy(dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Pair endpoint features must be finite")
    return left, right

def fit_family_scalers(
    fitting_pairs: pd.DataFrame,
    feature_index: pd.DataFrame,
    columns_by_family: dict[str, list[str]],
    required_role: str,
) -> tuple[dict[str, StandardScaler], int, int]:
    """Fit separate SSI and LF scalers to repeated fitting-pair endpoints."""
    if fitting_pairs.empty or set(fitting_pairs["pair_role"].astype(str)) != {required_role}:
        raise ValueError(f"Scaler fitting accepts non-empty {required_role} pairs only")
    scalers: dict[str, StandardScaler] = {}
    endpoint_rows = 2 * len(fitting_pairs)
    unique_events = len(
        set(fitting_pairs["event_uid_a"].astype(str))
        | set(fitting_pairs["event_uid_b"].astype(str))
    )
    for family in FEATURE_FAMILIES:
        left, right = endpoint_arrays(fitting_pairs, feature_index, columns_by_family[family])
        fit_values = np.vstack((left, right))
        if len(fit_values) != endpoint_rows:
            raise RuntimeError("Scaler fitting did not retain both repeated endpoint occurrences")
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(fit_values)
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.var_).all() or not np.isfinite(scaler.scale_).all():
            raise ValueError(f"{family} scaler parameters are not finite")
        if (scaler.scale_ <= 0.0).any():
            raise ValueError(f"{family} scaler scales must be positive")
        scalers[family] = scaler
    return scalers, endpoint_rows, unique_events

def scaled_endpoint_arrays(
    pairs: pd.DataFrame,
    feature_index: pd.DataFrame,
    columns_by_family: dict[str, list[str]],
    scalers: dict[str, StandardScaler],
    regime: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Scale pair endpoints with already fitted family-specific scalers."""
    if regime not in FEATURE_REGIMES:
        raise ValueError(f"Unknown feature regime: {regime}")
    families = (regime,) if regime in FEATURE_FAMILIES else FEATURE_FAMILIES
    left_blocks: list[np.ndarray] = []
    right_blocks: list[np.ndarray] = []
    names: list[str] = []
    for family in families:
        left, right = endpoint_arrays(pairs, feature_index, columns_by_family[family])
        left_blocks.append(scalers[family].transform(left))
        right_blocks.append(scalers[family].transform(right))
        names.extend(columns_by_family[family])
    left_scaled = np.hstack(left_blocks)
    right_scaled = np.hstack(right_blocks)
    if not np.isfinite(left_scaled).all() or not np.isfinite(right_scaled).all():
        raise ValueError("Scaled pair endpoints must be finite")
    return left_scaled, right_scaled, names

def symmetric_pair_representation(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Construct [absolute_difference, pair_mean] for each unordered pair."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"Endpoint matrices must have the same two-dimensional shape: {left.shape}, {right.shape}")
    representation = np.hstack((np.abs(left - right), (left + right) / 2.0))
    swapped = np.hstack((np.abs(right - left), (right + left) / 2.0))
    if not np.array_equal(representation, swapped):
        raise ValueError("The pair representation is not exactly invariant to endpoint order")
    if not np.isfinite(representation).all():
        raise ValueError("Pair representations must be finite")
    return representation

def build_pair_representation(
    pairs: pd.DataFrame,
    feature_index: pd.DataFrame,
    columns_by_family: dict[str, list[str]],
    scalers: dict[str, StandardScaler],
    regime: str,
) -> tuple[np.ndarray, list[str]]:
    """Build the exact symmetric representation for one feature regime."""
    left, right, original_names = scaled_endpoint_arrays(
        pairs,
        feature_index,
        columns_by_family,
        scalers,
        regime,
    )
    representation = symmetric_pair_representation(left, right)
    expected = {"SSI": 120, "LF": 256, "combined": 376}[regime]
    if representation.shape != (len(pairs), expected):
        raise ValueError(f"{regime} representation shape is {representation.shape}, expected {(len(pairs), expected)}")
    return representation, original_names

def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, tuple[float | None, str]]:
    """Calculate MAE, RMSE, R-squared, and Spearman without p-values."""
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if targets.ndim != 1 or predictions.ndim != 1 or len(targets) != len(predictions) or not len(targets):
        raise ValueError("Metric arrays must be non-empty, one-dimensional, and equally sized")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Metric inputs must be finite")
    if len(targets) < 2:
        spearman: float | None = None
        reason = "insufficient_pairs"
    elif np.ptp(predictions) == 0.0:
        spearman = None
        reason = "constant_prediction"
    elif np.ptp(targets) == 0.0:
        spearman = None
        reason = "constant_target"
    else:
        coefficient = float(spearmanr(targets, predictions).statistic)
        spearman = coefficient if math.isfinite(coefficient) else None
        reason = "" if spearman is not None else "undefined_spearman"
    return {
        "mae": (float(mean_absolute_error(targets, predictions)), ""),
        "rmse": (float(math.sqrt(mean_squared_error(targets, predictions))), ""),
        "r2": (float(r2_score(targets, predictions)), ""),
        "spearman": (spearman, reason),
    }

def candidate_grid(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Create the exact deterministic ParameterGrid candidate sequence."""
    candidates = []
    for index, parameters in enumerate(ParameterGrid(parameter_grid)):
        candidates.append({"candidate_id": f"candidate_{index:03d}", **parameters})
    if len(candidates) != 12:
        raise ValueError(f"ParameterGrid produced {len(candidates)} candidates, expected 12")
    return candidates

def random_forest(
    fixed_parameters: dict[str, Any],
    candidate: dict[str, Any],
) -> RandomForestRegressor:
    """Instantiate one fixed, deterministic RandomForestRegressor candidate."""
    parameters = dict(fixed_parameters)
    parameters.update(
        {
            "max_depth": candidate["max_depth"],
            "min_samples_leaf": int(candidate["min_samples_leaf"]),
            "max_features": candidate["max_features"],
        }
    )
    return RandomForestRegressor(**parameters)

def serialize_max_depth(value: int | None) -> str:
    """Represent an optional maximum depth without CSV missing-value ambiguity."""
    return "none" if value is None else str(int(value))

def serialize_max_features(value: str | float) -> str:
    """Represent mixed max_features settings as stable text."""
    return str(value)

def candidate_comparator(tolerance: float):
    """Return the exact mean-MAE, standard-deviation, candidate-ID comparator."""
    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        mean_difference = float(left["mean_validation_mae"]) - float(right["mean_validation_mae"])
        if abs(mean_difference) > tolerance:
            return -1 if mean_difference < 0.0 else 1
        standard_difference = float(left["std_validation_mae"]) - float(right["std_validation_mae"])
        if standard_difference != 0.0:
            return -1 if standard_difference < 0.0 else 1
        left_id = str(left["candidate_id"])
        right_id = str(right["candidate_id"])
        return -1 if left_id < right_id else (1 if left_id > right_id else 0)
    return compare

def rank_candidate_rows(rows: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    """Rank candidate summaries and mark exactly one selected candidate."""
    ordered = sorted(rows, key=functools.cmp_to_key(candidate_comparator(tolerance)))
    for rank, row in enumerate(ordered, start=1):
        row["selection_rank"] = rank
        row["selected"] = rank == 1
    return ordered

def scaler_parameter_rows(
    outer_holdout: str,
    condition: str,
    fit_scope: str,
    inner_validation_dataset: str,
    family: str,
    columns: list[str],
    scaler: StandardScaler,
    endpoint_rows: int,
    unique_events: int,
    pair_role: str,
) -> list[dict[str, Any]]:
    """Represent one fitted scaler transparently in long form."""
    used_by = "SSI|combined" if family == "SSI" else "LF|combined"
    return [
        {
            "outer_holdout": outer_holdout,
            "data_condition": condition,
            "fit_scope": fit_scope,
            "inner_validation_dataset": inner_validation_dataset,
            "feature_family": family,
            "feature_index": index,
            "feature_name": column,
            "mean": float(scaler.mean_[index]),
            "variance": float(scaler.var_[index]),
            "scale": float(scaler.scale_[index]),
            "n_fit_endpoint_rows": endpoint_rows,
            "n_unique_fit_events": unique_events,
            "permitted_fitting_pair_role": pair_role,
            "used_by_feature_regimes": used_by,
        }
        for index, column in enumerate(columns)
    ]

def feature_importance_rows(
    outer_holdout: str,
    condition: str,
    regime: str,
    original_names: list[str],
    importances: np.ndarray,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Create ranked exploratory impurity-based feature-importance rows."""
    importances = np.asarray(importances, dtype=float)
    expected = 2 * len(original_names)
    if importances.shape != (expected,) or not np.isfinite(importances).all() or (importances < 0.0).any():
        raise ValueError(f"Invalid {regime} feature importances")
    if not math.isclose(float(importances.sum()), 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{regime} feature importances sum to {importances.sum():.17g}, expected 1")
    records: list[dict[str, Any]] = []
    components = ("absolute_difference", "pair_mean")
    for component_index, component in enumerate(components):
        for feature_index, name in enumerate(original_names):
            representation_index = component_index * len(original_names) + feature_index
            records.append(
                {
                    "outer_holdout": outer_holdout,
                    "data_condition": condition,
                    "feature_regime": regime,
                    "pair_component": component,
                    "original_feature_name": name,
                    "representation_index": representation_index,
                    "importance": float(importances[representation_index]),
                    "rank": 0,
                    "importance_method": "exploratory_impurity_based",
                }
            )
    ordered = sorted(records, key=lambda row: (-float(row["importance"]), int(row["representation_index"])))
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered

def ensure_prediction_range(predictions: np.ndarray, label: str) -> None:
    """Reject non-finite random-forest predictions or any value outside [0, 1]."""
    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} predictions contain non-finite values")
    if ((values < 0.0) | (values > 1.0)).any():
        minimum = float(values.min())
        maximum = float(values.max())
        raise ValueError(f"{label} predictions leave [0, 1]: minimum={minimum}, maximum={maximum}")

def safe_result_path(path: Path, results_root: Path) -> Path:
    """Refuse generated destinations outside the random-forest result tree."""
    resolved = path.resolve()
    if not resolved.is_relative_to(results_root.resolve()):
        raise ValueError(f"Refusing to write outside the random-forest results tree: {resolved}")
    return resolved

__all__ = [
    "CONDITIONS",
    "DATASETS",
    "FEATURE_FAMILIES",
    "FEATURE_REGIMES",
    "METRICS",
    "atomic_text_write",
    "build_pair_representation",
    "candidate_grid",
    "dataframe_text",
    "ensure_prediction_range",
    "feature_columns",
    "feature_importance_rows",
    "fit_family_scalers",
    "index_features",
    "protected_input_paths",
    "protected_inventory_digest",
    "protected_inventory_lines",
    "random_forest",
    "rank_candidate_rows",
    "regression_metrics",
    "safe_result_path",
    "scaler_parameter_rows",
    "serialize_max_depth",
    "serialize_max_features",
    "sha256_file",
]
