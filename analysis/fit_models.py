from __future__ import annotations

import csv

import math

import os

import subprocess

import tempfile

import time

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

from .final_fit_utils import ARCHITECTURES, SSI_COLUMNS, RawLogitSiamese, all_parameters_finite, apply_preprocessing, fit_preprocessing, initialize_output, predict_model, set_deterministic, sha256_file, state_hash

ROOT = Path(__file__).resolve().parents[1]

HOLDOUTS = ("india", "malaysia_safe", "taiwan", "usa")

CONDITIONS = ("author_compatible", "filtered")



GDM_PRED_COLUMNS = ["outer_holdout", "data_condition", "dataset", "pair_role", "pair_id", "event_uid_a", "event_uid_b", "model_id", "prediction"]

SEED_PRED_COLUMNS = ["outer_holdout", "data_condition", "dataset", "pair_role", "pair_id", "event_uid_a", "event_uid_b", "model_id", "architecture", "preprocessing", "seed", "epoch_count", "prediction"]


def load_manifest(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load metadata for all roles while converting targets only for final_train."""
    metadata: list[dict[str, Any]] = []
    train_targets: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: header.index(name) for name in header}
        required = ["pair_id", "outer_holdout", "data_condition", "pair_role", "dataset", "event_uid_a", "event_uid_b", "jaccard_distance"]
        if any(name not in positions for name in required):
            raise ValueError(f"Invalid manifest schema: {path}")
        for row in reader:
            role = row[positions["pair_role"]]
            metadata.append({name: row[positions[name]] for name in required if name != "jaccard_distance"})
            if role == "final_train":
                target = float(row[positions["jaccard_distance"]])
                if not math.isfinite(target) or not 0.0 <= target <= 1.0:
                    raise ValueError("Invalid final-training target")
                train_targets.append(target)
    frame = pd.DataFrame(metadata)
    if frame["pair_id"].duplicated().any() or set(frame["pair_role"]) != {"final_train", "familiar_test", "unseen_test"}:
        raise ValueError("Manifest pair identifiers or roles are invalid")
    if len(train_targets) != int((frame["pair_role"] == "final_train").sum()):
        raise ValueError("Training-target isolation failed")
    return frame, np.asarray(train_targets, dtype=np.float32)

def endpoint_arrays(pairs: pd.DataFrame, feature_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ordered endpoint SSI arrays."""
    try:
        a = feature_index.loc[pairs["event_uid_a"], SSI_COLUMNS].to_numpy(dtype=np.float64)
        b = feature_index.loc[pairs["event_uid_b"], SSI_COLUMNS].to_numpy(dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"Unknown pair endpoint: {error}") from error
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("Nonfinite SSI endpoint")
    return a, b

def checkpoint_to_temp(payload: dict[str, Any], target: Path) -> None:
    """Save a PyTorch checkpoint atomically from system temporary storage."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="siamese_", suffix=".pt")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

def train_siamese(model: RawLogitSiamese, a: np.ndarray, b: np.ndarray, targets: np.ndarray, spec: Any, seed: int, epochs: int, fit_id: str) -> list[dict[str, Any]]:
    """Train for the fixed locked epoch count without validation or early stopping."""
    dataset = TensorDataset(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(targets), torch.arange(len(targets)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator, num_workers=0, drop_last=False)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter(); model.train(); loss_sum = 0.0; count = 0; gradient_finite = True
        for endpoint_a, endpoint_b, target, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logit = model(endpoint_a, endpoint_b)
            loss = criterion(logit, target)
            if not torch.isfinite(logit).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"{fit_id} nonfinite training value")
            loss.backward()
            gradient_finite = gradient_finite and all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
            if not gradient_finite:
                raise FloatingPointError(f"{fit_id} nonfinite gradient")
            optimizer.step()
            if not all_parameters_finite(model):
                raise FloatingPointError(f"{fit_id} nonfinite parameter")
            loss_sum += float(loss.detach()) * len(target); count += len(target)
        history.append({"model_id": fit_id, "model_family": "siamese", "epoch": epoch, "training_loss": loss_sum / count, "all_parameters_finite": True, "all_gradients_finite": gradient_finite, "fit_duration_seconds": time.perf_counter() - started})
    return history

def gdm_table(pairs: pd.DataFrame, feature_index: pd.DataFrame, targets: np.ndarray | None) -> pd.DataFrame:
    """Construct the official GDM site-pair table using raw SSI values."""
    a, b = endpoint_arrays(pairs, feature_index)
    distance = targets.astype(np.float64) if targets is not None else np.full(len(pairs), 0.5, dtype=np.float64)
    values: dict[str, Any] = {"distance": distance, "weights": np.ones(len(pairs)), "s1.xCoord": np.zeros(len(pairs)), "s1.yCoord": np.zeros(len(pairs)), "s2.xCoord": np.zeros(len(pairs)), "s2.yCoord": np.zeros(len(pairs))}
    for index, name in enumerate(SSI_COLUMNS): values[f"s1.{name}"] = a[:, index]
    for index, name in enumerate(SSI_COLUMNS): values[f"s2.{name}"] = b[:, index]
    return pd.DataFrame(values)

def run_gdm_fit(cell: tuple[str, str, str, str, int], manifest: pd.DataFrame, targets: np.ndarray, features: pd.DataFrame, checkpoint: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run exactly one official R GDM fit and its save/reload check."""
    holdout, condition, _, _, _ = cell; train = manifest[manifest.pair_role == "final_train"].reset_index(drop=True); evaluation = manifest[manifest.pair_role != "final_train"].reset_index(drop=True)
    model_id = f"gdm__holdout_{holdout}__{condition}"
    with tempfile.TemporaryDirectory(prefix="final_gdm_") as directory:
        temporary = Path(directory); train_path = temporary / "train.csv"; evaluation_path = temporary / "evaluation.csv"; model_path = temporary / "model.rds"; prediction_path = temporary / "prediction.csv"; parameter_path = temporary / "parameters.csv"
        gdm_table(train, features, targets).to_csv(train_path, index=False, float_format="%.17g")
        gdm_table(evaluation, features, None).to_csv(evaluation_path, index=False, float_format="%.17g")
        subprocess.run(["Rscript", str(ROOT / "analysis/fit_gdm.R"), str(train_path), str(evaluation_path), str(model_path), str(prediction_path), str(parameter_path)], check=True, cwd=ROOT)
        checkpoint.parent.mkdir(parents=True, exist_ok=True); os.replace(model_path, checkpoint)
        prediction = pd.read_csv(prediction_path)["prediction"].to_numpy(dtype=np.float64)
        parameters = pd.read_csv(parameter_path)
    if len(prediction) != len(evaluation) or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Invalid GDM predictions")
    rows = evaluation[["outer_holdout", "data_condition", "dataset", "pair_role", "pair_id", "event_uid_a", "event_uid_b"]].copy(); rows["model_id"] = model_id; rows["prediction"] = prediction; rows = rows[GDM_PRED_COLUMNS]
    parameters.insert(0, "model_id", model_id); parameters.insert(0, "data_condition", condition); parameters.insert(0, "outer_holdout", holdout)
    summary = {"model_id": model_id, "model_family": "gdm", "outer_holdout": holdout, "data_condition": condition, "seed": "", "architecture": "", "preprocessing": "RAW_SSI", "epoch_count": "", "training_pair_count": len(train), "prediction_count": len(evaluation), "predictor_count": 60, "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256_file(checkpoint), "model_state_sha256": "", "status": "validated", "fits_executed_this_run": 1}
    return rows, parameters, summary

def run_siamese_fit(cell: tuple[str, str, str, str, int], seed: int, manifest: pd.DataFrame, targets: np.ndarray, features: pd.DataFrame, checkpoint: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Run one locked final Siamese fit, save it, reload it, and reproduce predictions."""
    holdout, condition, preprocessing, architecture, epochs = cell; train = manifest[manifest.pair_role == "final_train"].reset_index(drop=True); evaluation = manifest[manifest.pair_role != "final_train"].reset_index(drop=True); model_id = f"siamese__holdout_{holdout}__{condition}__seed_{seed}"; spec = ARCHITECTURES[architecture]
    unique_ids = sorted(set(train.event_uid_a) | set(train.event_uid_b)); unique_values = features.loc[unique_ids, SSI_COLUMNS].to_numpy(dtype=np.float64); fit = fit_preprocessing(preprocessing, unique_values)
    train_a_raw, train_b_raw = endpoint_arrays(train, features); eval_a_raw, eval_b_raw = endpoint_arrays(evaluation, features)
    train_a = apply_preprocessing(preprocessing, train_a_raw, fit); train_b = apply_preprocessing(preprocessing, train_b_raw, fit); eval_a = apply_preprocessing(preprocessing, eval_a_raw, fit); eval_b = apply_preprocessing(preprocessing, eval_b_raw, fit)
    set_deterministic(seed); model = RawLogitSiamese(spec); output_bias = initialize_output(model, float(np.mean(targets, dtype=np.float64))); history = train_siamese(model, train_a, train_b, targets, spec, seed, epochs, model_id)
    before_logits, before_prediction, before_swap = predict_model(model, eval_a, eval_b)
    if before_swap != 0.0 or not all_parameters_finite(model) or np.any((before_prediction < 0) | (before_prediction > 1)) or float(np.max(before_prediction) - np.min(before_prediction)) <= 1.0e-8:
        raise RuntimeError(f"{model_id} failed structural validation")
    model_hash = state_hash(model)
    payload = {"format_version": 1, "model_id": model_id, "outer_holdout": holdout, "data_condition": condition, "architecture": architecture, "preprocessing": preprocessing, "seed": seed, "epoch_count": epochs, "ssi_columns": SSI_COLUMNS, "state_dict": model.state_dict(), "model_state_sha256": model_hash, "preprocessing_parameters": {key: value.tolist() for key, value in fit.items()}, "unique_training_event_ids": unique_ids, "training_target_mean": float(np.mean(targets, dtype=np.float64)), "output_bias": output_bias}
    checkpoint_to_temp(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False); reloaded = RawLogitSiamese(spec); reloaded.load_state_dict(loaded["state_dict"]); after_logits, after_prediction, after_swap = predict_model(reloaded, eval_a, eval_b)
    if model_hash != state_hash(reloaded) or not np.array_equal(before_logits, after_logits) or not np.array_equal(before_prediction, after_prediction) or after_swap != 0.0:
        raise RuntimeError(f"{model_id} failed exact checkpoint reproduction")
    rows = evaluation[["outer_holdout", "data_condition", "dataset", "pair_role", "pair_id", "event_uid_a", "event_uid_b"]].copy(); rows["model_id"] = model_id; rows["architecture"] = architecture; rows["preprocessing"] = preprocessing; rows["seed"] = seed; rows["epoch_count"] = epochs; rows["prediction"] = after_prediction.astype(np.float64); rows = rows[SEED_PRED_COLUMNS]
    parameter_rows = []
    for index, feature in enumerate(SSI_COLUMNS):
        parameter_rows.append({"model_id": model_id, "outer_holdout": holdout, "data_condition": condition, "fit_scope": "final_train_unique_events", "feature_name": feature, "preprocessing": preprocessing, "asinh_scale": fit["feature_scale"][index], "training_mean": fit["mean"][index], "training_variance": fit["variance"][index], "training_scale": fit["scale"][index], "unique_training_event_count": len(unique_ids), "permitted_pair_role": "final_train", "clipping_applied": False})
    summary = {"model_id": model_id, "model_family": "siamese", "outer_holdout": holdout, "data_condition": condition, "seed": seed, "architecture": architecture, "preprocessing": preprocessing, "epoch_count": epochs, "training_pair_count": len(train), "prediction_count": len(evaluation), "predictor_count": 60, "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256_file(checkpoint), "model_state_sha256": model_hash, "status": "validated", "fits_executed_this_run": 1}
    return rows, pd.DataFrame(parameter_rows), history, summary
