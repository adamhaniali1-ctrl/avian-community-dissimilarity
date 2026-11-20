from __future__ import annotations

import copy

import hashlib

import json

import math

import os

import random

import tempfile

import time

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import torch

from scipy.stats import spearmanr

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

SSI_COLUMNS = [f"ssi_{index:03d}" for index in range(60)]

PREPROCESSING_METHODS = ("STANDARD", "ASINH_STANDARD")


HOLDOUTS = ("india", "malaysia_safe", "taiwan", "usa")

CONDITIONS = ("author_compatible", "filtered")

@dataclass(frozen=True)
class PreprocessingFit:
    """Saved training-only parameters for one fold and method."""

    method: str
    asinh_scale: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    unique_training_events: int
    training_event_hash: str

class RawLogitCandidateA(nn.Module):
    """Fixed Candidate A shared encoder with a symmetric raw-logit head."""

    def __init__(self, input_dimension: int = 60) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward_components(
        self, endpoint_a: torch.Tensor, endpoint_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw logits, both embeddings and the symmetric comparison."""
        embedding_a = self.encoder(endpoint_a)
        embedding_b = self.encoder(endpoint_b)
        comparison = torch.cat(
            (torch.abs(embedding_a - embedding_b), embedding_a * embedding_b), dim=1
        )
        logit = self.regression_head(comparison).squeeze(1)
        return logit, embedding_a, embedding_b, comparison

    def forward(self, endpoint_a: torch.Tensor, endpoint_b: torch.Tensor) -> torch.Tensor:
        """Return unrestricted raw logits."""
        return self.forward_components(endpoint_a, endpoint_b)[0]

def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical_json(value: Any) -> str:
    """Serialize strict deterministic JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a stable CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g", na_rep="")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a pickle-free compressed progress record atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def set_deterministic(seed: int, threads: int = 4) -> None:
    """Configure an isolated deterministic CPU fit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)

def state_hash(model: nn.Module) -> str:
    """Hash a model state deterministically."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()

def initialize_output(model: RawLogitCandidateA, target_mean: float) -> dict[str, float]:
    """Initialize the final layer at the clamped training-target mean."""
    clamped = min(max(float(target_mean), 0.000001), 0.999999)
    bias = math.log(clamped / (1.0 - clamped))
    output = model.regression_head[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("Candidate A final head module is not Linear")
    with torch.no_grad():
        output.weight.zero_()
        output.bias.fill_(bias)
    return {"training_target_mean": float(target_mean), "clamped_mean": clamped, "output_bias": bias}

def fit_preprocessing(method: str, unique_training_values: np.ndarray, event_ids: list[str]) -> PreprocessingFit:
    """Fit the exact requested method using each training event once."""
    values = np.asarray(unique_training_values, dtype=np.float64)
    if method not in PREPROCESSING_METHODS or values.ndim != 2 or values.shape[1] != 60:
        raise ValueError("Invalid preprocessing request")
    if len(values) != len(event_ids) or len(event_ids) != len(set(event_ids)) or not np.isfinite(values).all():
        raise ValueError("Preprocessing fit population is invalid")
    if event_ids != sorted(event_ids):
        raise ValueError("Training event IDs must have deterministic ordering")
    if method == "STANDARD":
        asinh_scale = np.full(60, np.nan, dtype=np.float64)
        transformed = values
    else:
        asinh_scale = np.empty(60, dtype=np.float64)
        for index in range(60):
            nonzero = np.abs(values[:, index][values[:, index] != 0.0])
            asinh_scale[index] = float(np.median(nonzero)) if nonzero.size else 1.0
        transformed = np.arcsinh(values / asinh_scale)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(transformed)
    event_hash = hashlib.sha256("".join(f"{event_id}\n" for event_id in event_ids).encode()).hexdigest()
    return PreprocessingFit(
        method=method,
        asinh_scale=asinh_scale,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        variance=np.asarray(scaler.var_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        unique_training_events=len(event_ids),
        training_event_hash=event_hash,
    )

def apply_preprocessing(values: np.ndarray, fit: PreprocessingFit) -> np.ndarray:
    """Apply saved training-only parameters without refitting."""
    raw = np.asarray(values, dtype=np.float64)
    transformed = raw if fit.method == "STANDARD" else np.arcsinh(raw / fit.asinh_scale)
    result = (transformed - fit.mean) / fit.scale
    if not np.isfinite(result).all():
        raise FloatingPointError("Preprocessing produced a nonfinite value")
    return result.astype(np.float32)

def inverse_preprocessing(values: np.ndarray, fit: PreprocessingFit) -> np.ndarray:
    """Invert the saved transform in float64."""
    standardized = np.asarray(values, dtype=np.float64)
    transformed = fit.mean + fit.scale * standardized
    return transformed if fit.method == "STANDARD" else fit.asinh_scale * np.sinh(transformed)

def pair_arrays(
    pairs: pd.DataFrame, feature_index: pd.DataFrame, fit: PreprocessingFit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join endpoints to SSI features and transform them."""
    endpoint_a = feature_index.loc[pairs["event_uid_a"], SSI_COLUMNS].to_numpy(dtype=np.float64)
    endpoint_b = feature_index.loc[pairs["event_uid_b"], SSI_COLUMNS].to_numpy(dtype=np.float64)
    targets = pairs["jaccard_distance"].to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all() or np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("Inner Jaccard target is invalid")
    return apply_preprocessing(endpoint_a, fit), apply_preprocessing(endpoint_b, fit), targets

def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Calculate regression metrics without p-values."""
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise FloatingPointError("Metric input is nonfinite")
    if np.unique(p).size < 2:
        spearman = math.nan
        reason = "constant_prediction"
    elif np.unique(y).size < 2:
        spearman = math.nan
        reason = "constant_target"
    else:
        spearman = float(spearmanr(y, p).statistic)
        reason = "" if math.isfinite(spearman) else "undefined_spearman"
    return {
        "mae": float(mean_absolute_error(y, p)),
        "mse": float(mean_squared_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "spearman": spearman,
        "spearman_missing_reason": reason,
    }

def gradient_norms(model: nn.Module) -> dict[str, float]:
    """Return L2 gradient norms for the requested parameter groups."""
    groups: dict[str, list[torch.Tensor]] = {"global": [], "encoder": [], "head": [], "output": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        groups["global"].append(gradient)
        if name.startswith("encoder."):
            groups["encoder"].append(gradient)
        if name.startswith("regression_head."):
            groups["head"].append(gradient)
        if name.startswith("regression_head.5."):
            groups["output"].append(gradient)
    result: dict[str, float] = {}
    for group, tensors in groups.items():
        total = sum(float(torch.sum(tensor.double() ** 2)) for tensor in tensors)
        result[f"gradient_{group}"] = math.sqrt(total)
    return result

def all_parameters_finite(model: nn.Module) -> bool:
    """Check every model parameter."""
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())

def evaluate(
    model: RawLogitCandidateA, loader: DataLoader, criterion: nn.Module, swap_check: bool = False
) -> dict[str, Any]:
    """Evaluate one model deterministically."""
    model.eval()
    logits: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    bce_sum = 0.0
    count = 0
    swap_maximum = 0.0
    with torch.no_grad():
        for endpoint_a, endpoint_b, target in loader:
            logit, embedding_a, embedding_b, comparison = model.forward_components(endpoint_a, endpoint_b)
            prediction = torch.sigmoid(logit)
            tensors = (logit, prediction, embedding_a, embedding_b, comparison)
            if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
                raise FloatingPointError("Evaluation contains a nonfinite tensor")
            if swap_check:
                swapped = model(endpoint_b, endpoint_a)
                swap_maximum = max(swap_maximum, float(torch.max(torch.abs(logit - swapped))))
            loss = criterion(logit, target)
            rows = len(target)
            bce_sum += float(loss) * rows
            count += rows
            logits.append(logit.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    logit_values = np.concatenate(logits)
    prediction_values = np.concatenate(predictions)
    target_values = np.concatenate(targets)
    metrics = regression_metrics(target_values, prediction_values)
    result = {
        "validation_bce": bce_sum / count,
        **metrics,
        "prediction_minimum": float(np.min(prediction_values)),
        "prediction_maximum": float(np.max(prediction_values)),
        "prediction_mean": float(np.mean(prediction_values, dtype=np.float64)),
        "prediction_standard_deviation": float(np.std(prediction_values.astype(np.float64), ddof=0)),
        "logit_minimum": float(np.min(logit_values)),
        "logit_maximum": float(np.max(logit_values)),
        "fraction_below": float(np.mean(prediction_values < 0.000001)),
        "fraction_above": float(np.mean(prediction_values > 0.999999)),
        "all_finite": bool(np.isfinite(logit_values).all() and np.isfinite(prediction_values).all()),
        "swap_maximum_absolute_difference": swap_maximum,
        "logits": logit_values,
        "predictions": prediction_values,
        "targets": target_values,
    }
    return result

def fit_identifier(holdout: str, condition: str, validation: str, method: str, seed: int) -> str:
    """Create a deterministic fit identifier."""
    return f"holdout_{holdout}__{condition}__validate_{validation}__{method}__seed_{seed}"

def run_fit(
    fit_id: str,
    seed: int,
    train_a: np.ndarray,
    train_b: np.ndarray,
    train_targets: np.ndarray,
    validation_a: np.ndarray,
    validation_b: np.ndarray,
    validation_targets: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Train one fixed Candidate A model and restore its best validation checkpoint."""
    set_deterministic(seed, int(settings["torch_threads"]))
    model = RawLogitCandidateA(60)
    initialization = initialize_output(model, float(np.mean(train_targets, dtype=np.float64)))
    initial_state_hash = state_hash(model)
    train_dataset = TensorDataset(
        torch.from_numpy(train_a), torch.from_numpy(train_b), torch.from_numpy(train_targets),
        torch.arange(len(train_targets), dtype=torch.int64),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_a), torch.from_numpy(validation_b), torch.from_numpy(validation_targets)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=int(settings["batch_size"]), shuffle=True, generator=generator,
        num_workers=0, drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(settings["validation_batch_size"]), shuffle=False,
        num_workers=0, drop_last=False,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    initial = evaluate(model, validation_loader, criterion)
    initial_difference = float(np.max(np.abs(initial["predictions"] - initialization["clamped_mean"])))
    if initial_difference > 1.0e-6:
        raise RuntimeError("Initial prediction differs from training-target mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    history: list[dict[str, Any]] = []
    best_mae = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    early_stopping_epoch: int | None = None
    started = time.perf_counter()
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        orders: list[np.ndarray] = []
        gradient_rows: list[dict[str, float]] = []
        finite = True
        for endpoint_a, endpoint_b, target, indices in train_loader:
            orders.append(indices.numpy())
            optimizer.zero_grad(set_to_none=True)
            logit, embedding_a, embedding_b, comparison = model.forward_components(endpoint_a, endpoint_b)
            loss = criterion(logit, target)
            finite = finite and all(torch.isfinite(value).all().item() for value in (logit, embedding_a, embedding_b, comparison, loss))
            if not finite:
                raise FloatingPointError(f"{fit_id} produced a nonfinite training value")
            loss.backward()
            norms = gradient_norms(model)
            if not all(math.isfinite(value) for value in norms.values()):
                raise FloatingPointError(f"{fit_id} produced a nonfinite gradient")
            gradient_rows.append(norms)
            optimizer.step()
            if not all_parameters_finite(model):
                raise FloatingPointError(f"{fit_id} produced a nonfinite parameter")
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        order = np.concatenate(orders).astype(np.int64, copy=False)
        if len(order) != len(train_dataset) or len(np.unique(order)) != len(train_dataset):
            raise RuntimeError("Deterministic shuffle did not cover every pair once")
        validation = evaluate(model, validation_loader, criterion)
        improved = best_state is None or validation["mae"] < best_mae - float(settings["minimum_improvement"])
        if improved:
            best_mae = float(validation["mae"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        row: dict[str, Any] = {
            "epoch": epoch,
            "training_loss": loss_sum / count,
            "validation_bce": validation["validation_bce"],
            "validation_mae": validation["mae"],
            "validation_rmse": validation["rmse"],
            "validation_r2": validation["r2"],
            "validation_spearman": validation["spearman"],
            "spearman_missing_reason": validation["spearman_missing_reason"],
            "prediction_minimum": validation["prediction_minimum"],
            "prediction_maximum": validation["prediction_maximum"],
            "prediction_mean": validation["prediction_mean"],
            "prediction_standard_deviation": validation["prediction_standard_deviation"],
            "logit_minimum": validation["logit_minimum"],
            "logit_maximum": validation["logit_maximum"],
            "fraction_below": validation["fraction_below"],
            "fraction_above": validation["fraction_above"],
            "gradient_global_mean": float(np.mean([item["gradient_global"] for item in gradient_rows])),
            "gradient_encoder_mean": float(np.mean([item["gradient_encoder"] for item in gradient_rows])),
            "gradient_head_mean": float(np.mean([item["gradient_head"] for item in gradient_rows])),
            "gradient_output_mean": float(np.mean([item["gradient_output"] for item in gradient_rows])),
            "all_finite": bool(finite and validation["all_finite"] and all_parameters_finite(model)),
            "improved_best_mae": improved,
            "patience_state": patience,
            "shuffle_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        }
        history.append(row)
        if patience >= int(settings["patience"]):
            early_stopping_epoch = epoch
            break
    if best_state is None:
        raise RuntimeError("No best state was saved")
    model.load_state_dict(best_state)
    restored = evaluate(model, validation_loader, criterion, swap_check=True)
    prediction_range = restored["prediction_maximum"] - restored["prediction_minimum"]
    degenerate_reasons: list[str] = []
    if prediction_range <= 1.0e-8:
        degenerate_reasons.append("prediction_range_le_1e-8")
    if restored["prediction_standard_deviation"] <= 1.0e-8:
        degenerate_reasons.append("prediction_sd_le_1e-8")
    if restored["fraction_below"] > 0.999 or restored["fraction_above"] > 0.999:
        degenerate_reasons.append("single_extreme_tail_gt_0.999")
    if not restored["all_finite"] or not all_parameters_finite(model):
        degenerate_reasons.append("nonfinite")
    for row in history:
        row["selected_best_epoch"] = best_epoch
        row["early_stopping_epoch"] = early_stopping_epoch
        row["is_selected_best_epoch"] = row["epoch"] == best_epoch
        row["restored_best_state_hash"] = state_hash(model)
    return {
        "fit_id": fit_id,
        "seed": seed,
        "history": history,
        "restored": restored,
        "best_epoch": best_epoch,
        "early_stopping_epoch": early_stopping_epoch,
        "epochs_completed": len(history),
        "state_hash": state_hash(model),
        "initial_state_hash": initial_state_hash,
        "initialization": initialization,
        "initial_prediction_difference": initial_difference,
        "degenerate": bool(degenerate_reasons),
        "degenerate_reasons": ";".join(degenerate_reasons),
        "fit_seconds": time.perf_counter() - started,
    }
