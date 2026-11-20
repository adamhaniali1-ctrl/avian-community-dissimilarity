from __future__ import annotations

import sys

from pathlib import Path

from typing import Any

sys.dont_write_bytecode = True

import numpy as np

import pandas as pd

from .ssi_preprocessing_comparison_utils import SSI_COLUMNS, PreprocessingFit, apply_preprocessing, fit_preprocessing, inverse_preprocessing, pair_arrays

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = PROJECT_ROOT / "results/ssi_preprocessing_comparison"

PROGRESS_ROOT = RESULT_ROOT / ".fit_progress"

REPORT_PATH = PROJECT_ROOT / "audit_reports/ssi_preprocessing_comparison_audit.md"

EXPECTED_PROTECTED_COUNT = 276

EXPECTED_PROTECTED_DIGEST = "8f3f325caa7b7a30db2df22f3b75411654baf7fafb27a169453a9932bb07c5ff"

OUTPUT_NAMES = (
    "inner_predictions.csv", "inner_metrics.csv", "epoch_history.csv", "preprocessing_parameters.csv",
    "selected_preprocessing.csv", "fit_summary.csv", "audit.md",
)

STAGE_PATHS = {
    "config/ssi_preprocessing_comparison.yaml",
    "scripts/ssi_preprocessing_comparison_utils.py",
    "scripts/run_ssi_preprocessing_comparison.py",
    "scripts/validate_ssi_preprocessing_comparison.py",
    "audit_reports/ssi_preprocessing_comparison_audit.md",
    *(f"results/ssi_preprocessing_comparison/{name}" for name in OUTPUT_NAMES),
}

TRAINING_SETTINGS = {
    "torch_threads": 4,
    "batch_size": 256,
    "validation_batch_size": 1024,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "maximum_epochs": 200,
    "patience": 15,
    "minimum_improvement": 0.0001,
}

def prepare_fold(
    pairs: pd.DataFrame, validation_dataset: str, feature_index: pd.DataFrame, method: str
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessingFit, tuple[np.ndarray, ...], list[dict[str, Any]]]:
    """Create one leakage-safe fold and its transparent preprocessing rows."""
    fold = pairs.loc[pairs["inner_validation_dataset"].eq(validation_dataset)].copy()
    training_pairs = fold.loc[fold["pair_role"].eq("train")].reset_index(drop=True)
    validation_pairs = fold.loc[fold["pair_role"].eq("validation")].reset_index(drop=True)
    training_ids = sorted(set(training_pairs["event_uid_a"]) | set(training_pairs["event_uid_b"]))
    validation_ids = sorted(set(validation_pairs["event_uid_a"]) | set(validation_pairs["event_uid_b"]))
    if set(training_ids) & set(validation_ids):
        raise ValueError("Inner train/validation event leakage")
    if set(validation_pairs["dataset"]) != {validation_dataset}:
        raise ValueError("Validation ecosystem differs")
    unique_values = feature_index.loc[training_ids, SSI_COLUMNS].to_numpy(dtype=np.float64)
    fit = fit_preprocessing(method, unique_values, training_ids)
    encoded = apply_preprocessing(unique_values, fit)
    reconstructed = inverse_preprocessing(encoded, fit)
    inverse_error = np.max(np.abs(reconstructed - unique_values), axis=0)
    train_arrays = pair_arrays(training_pairs, feature_index, fit)
    validation_arrays = pair_arrays(validation_pairs, feature_index, fit)
    parameter_rows: list[dict[str, Any]] = []
    for index, feature in enumerate(SSI_COLUMNS):
        parameter_rows.append({
            "method": method,
            "feature_name": feature,
            "feature_index": index,
            "asinh_scale": fit.asinh_scale[index],
            "standardizer_mean": fit.mean[index],
            "standardizer_variance": fit.variance[index],
            "standardizer_scale": fit.scale[index],
            "unique_training_events": fit.unique_training_events,
            "training_event_uid_sha256": fit.training_event_hash,
            "maximum_inverse_reconstruction_error": inverse_error[index],
            "fit_role": "inner_train_unique_events",
            "validation_values_used": False,
        })
    return training_pairs, validation_pairs, fit, (*train_arrays, *validation_arrays), parameter_rows
