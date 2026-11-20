from __future__ import annotations

import hashlib

import io

import math

import os

import tempfile

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

PREDICTIVE_METRICS = ("mae", "rmse", "r2", "spearman")

BASELINE_ORDER = (
    "training_mean",
    "author_unscaled_ssi_euclidean",
    "author_unscaled_lf_euclidean",
    "scaled_ssi_isotonic",
    "scaled_lf_isotonic",
)

PROTECTED_EXPLICIT_FILES = (
    "config/experiment_config.yaml",
    "audit_reports/data_audit_report.md",
    "audit_reports/experiment_manifest_audit.md",
    "scripts/data_preparation_script.py",
    "scripts/data_audit_script.py",
    "scripts/create_experiment_manifests.py",
    "scripts/validate_experiment_manifests.py",
)

PROTECTED_DIRECTORY_NAMES = (
    "data_clean",
    "original_downloads",
    "sarabsethi-sscape-avian-div-generalisability-d9f2547",
    "experiments",
)

def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def identify_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify and strictly verify the 60 SSI and 128 LF columns."""
    ssi = [f"ssi_{index:03d}" for index in range(60)]
    lf = [f"lf_{index:03d}" for index in range(128)]
    actual_ssi = [column for column in frame.columns if column.startswith("ssi_")]
    actual_lf = [column for column in frame.columns if column.startswith("lf_")]
    if actual_ssi != ssi:
        raise ValueError(f"SSI feature columns differ from the required 60-column order: {actual_ssi}")
    if actual_lf != lf:
        raise ValueError(f"LF feature columns differ from the required 128-column order: {actual_lf}")
    if len(ssi) + len(lf) != 188:
        raise RuntimeError("Feature-family dimensions do not sum to 188")
    values = frame[[*ssi, *lf]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"Non-finite central feature at row {row}, column {[ *ssi, *lf ][int(column)]}")
    return ssi, lf

def indexed_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Index a verified feature table by globally unique event UID."""
    if "event_uid" not in frame or frame["event_uid"].duplicated().any():
        raise ValueError("Feature table must contain unique event_uid values")
    return frame.set_index("event_uid", drop=False)

def endpoint_arrays(
    pairs: pd.DataFrame,
    feature_index: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Join exact pair endpoint IDs to a chosen feature family."""
    ids_a = pairs["event_uid_a"].astype(str)
    ids_b = pairs["event_uid_b"].astype(str)
    unknown = (set(ids_a) | set(ids_b)) - set(feature_index.index.astype(str))
    if unknown:
        raise ValueError(f"Pair endpoints absent from feature table: {sorted(unknown)[:10]}")
    left = feature_index.loc[ids_a, columns].to_numpy(dtype=float)
    right = feature_index.loc[ids_b, columns].to_numpy(dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite endpoint features encountered")
    return left, right

def fit_endpoint_scaler(
    training_pairs: pd.DataFrame,
    feature_index: pd.DataFrame,
    columns: list[str],
) -> tuple[StandardScaler, int, int]:
    """Fit StandardScaler only to stacked final-training endpoints."""
    if training_pairs.empty:
        raise ValueError("Scaler fitting requires non-empty final_train pairs")
    if "pair_role" not in training_pairs or set(training_pairs["pair_role"].astype(str)) != {"final_train"}:
        raise ValueError("Scaler fitting accepts final_train pairs only")
    left, right = endpoint_arrays(training_pairs, feature_index, columns)
    fit_values = np.vstack((left, right))
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(fit_values)
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("Scaler produced non-finite parameters")
    unique_events = len(set(training_pairs["event_uid_a"]) | set(training_pairs["event_uid_b"]))
    return scaler, len(fit_values), unique_events

def euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Calculate ordinary row-wise Euclidean distances."""
    if left.shape != right.shape:
        raise ValueError(f"Endpoint shapes differ: {left.shape} versus {right.shape}")
    distances = np.linalg.norm(left - right, axis=1)
    if not np.isfinite(distances).all() or (distances < 0.0).any():
        raise ValueError("Euclidean distances must be finite and non-negative")
    return distances

def scaled_euclidean_distances(
    left: np.ndarray,
    right: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a fitted scaler unchanged before Euclidean distance calculation."""
    return euclidean_distances(scaler.transform(left), scaler.transform(right))

def fit_isotonic(distances: np.ndarray, targets: np.ndarray) -> IsotonicRegression:
    """Fit the fixed increasing bounded calibration rule."""
    distances = np.asarray(distances, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if distances.ndim != 1 or targets.ndim != 1 or not len(distances) or len(distances) != len(targets):
        raise ValueError("Isotonic fit arrays must be non-empty, one-dimensional, and equally sized")
    if not np.isfinite(distances).all() or not np.isfinite(targets).all():
        raise ValueError("Isotonic fit inputs must be finite")
    if (distances < 0.0).any():
        raise ValueError("Isotonic distance inputs must be non-negative")
    if ((targets < 0.0) | (targets > 1.0)).any():
        raise ValueError("Isotonic targets must lie in [0, 1]")
    model = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    model.fit(distances, targets)
    return model

def predict_isotonic(model: IsotonicRegression, distances: np.ndarray) -> np.ndarray:
    """Apply an already fitted isotonic mapping and validate its range."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or not np.isfinite(distances).all() or (distances < 0.0).any():
        raise ValueError("Isotonic prediction distances must be finite, one-dimensional, and non-negative")
    predictions = np.asarray(model.predict(distances), dtype=float)
    if not np.isfinite(predictions).all() or ((predictions < 0.0) | (predictions > 1.0)).any():
        raise ValueError("Calibrated predictions must be finite and lie in [0, 1]")
    return predictions

def spearman_value(targets: np.ndarray, scores: np.ndarray) -> tuple[float | None, str]:
    """Calculate Spearman coefficient without returning a pair-dependence p-value."""
    if len(targets) < 2:
        return None, "insufficient_pairs"
    if np.ptp(scores) == 0.0:
        return None, "constant_prediction"
    if np.ptp(targets) == 0.0:
        return None, "constant_target"
    value = float(spearmanr(targets, scores).statistic)
    if not math.isfinite(value):
        return None, "undefined_spearman"
    return value, ""

def predictive_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, tuple[float | None, str]]:
    """Calculate predictive metrics while preserving negative R-squared values."""
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("Metric arrays must be non-empty and equally sized")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Metric arrays must be finite")
    spearman, reason = spearman_value(targets, predictions)
    return {
        "mae": (float(mean_absolute_error(targets, predictions)), ""),
        "rmse": (float(math.sqrt(mean_squared_error(targets, predictions))), ""),
        "r2": (float(r2_score(targets, predictions)), ""),
        "spearman": (spearman, reason),
    }

def dataframe_text(frame: pd.DataFrame, float_format: str = "%.17g") -> str:
    """Serialize a CSV deterministically with round-trip numeric precision."""
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, float_format=float_format, lineterminator="\n")
    return buffer.getvalue()

def atomic_text_write(text: str, destination: Path) -> None:
    """Atomically replace a generated text output."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

def prior_csv_match(folder: Path, csv_texts: dict[str, str]) -> str:
    """Compare a newly calculated complete CSV set to the previous generation."""
    presence = {name: (folder / name).is_file() for name in csv_texts}
    if not any(presence.values()):
        return "not_available"
    if not all(presence.values()):
        return "partial_previous_output"
    return "yes" if all((folder / name).read_bytes() == text.encode("utf-8") for name, text in csv_texts.items()) else "no"
