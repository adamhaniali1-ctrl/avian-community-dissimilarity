from __future__ import annotations

import copy

import hashlib

import math

import time

from dataclasses import dataclass

from typing import Any

import numpy as np

import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

from .ssi_preprocessing_comparison_utils import all_parameters_finite, gradient_norms, regression_metrics, set_deterministic, state_hash

SSI_COLUMNS = [f"ssi_{index:03d}" for index in range(60)]

HOLDOUTS = ("india", "malaysia_safe", "taiwan", "usa")

CONDITIONS = ("author_compatible", "filtered")

SEEDS = (42, 43, 44)

CANDIDATE_ORDER = ("A", "B", "C", "D")

@dataclass(frozen=True)
class CandidateSpec:
    """One prespecified corrected plain-Siamese architecture."""

    candidate: str
    hidden_1: int
    hidden_2: int
    embedding: int
    dropout: float
    learning_rate: float
    weight_decay: float

CANDIDATES = {
    "A": CandidateSpec("A", 128, 64, 32, 0.10, 0.0003, 0.0001),
    "B": CandidateSpec("B", 256, 128, 64, 0.20, 0.0003, 0.0001),
    "C": CandidateSpec("C", 128, 64, 32, 0.20, 0.0003, 0.0001),
    "D": CandidateSpec("D", 256, 128, 64, 0.30, 0.0003, 0.001),
}

class RawLogitSiamese(nn.Module):
    """Shared encoder and symmetric raw-logit regression head."""

    def __init__(self, spec: CandidateSpec, input_dimension: int = 60) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, spec.hidden_1), nn.ReLU(), nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_1, spec.hidden_2), nn.ReLU(), nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_2, spec.embedding),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(2 * spec.embedding, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward_components(
        self, endpoint_a: torch.Tensor, endpoint_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, embeddings and the symmetric comparison vector."""
        embedding_a = self.encoder(endpoint_a)
        embedding_b = self.encoder(endpoint_b)
        comparison = torch.cat(
            (torch.abs(embedding_a - embedding_b), embedding_a * embedding_b), dim=1
        )
        return self.regression_head(comparison).squeeze(1), embedding_a, embedding_b, comparison

    def forward(self, endpoint_a: torch.Tensor, endpoint_b: torch.Tensor) -> torch.Tensor:
        """Return unrestricted raw logits."""
        return self.forward_components(endpoint_a, endpoint_b)[0]

def initialize_output(model: RawLogitSiamese, target_mean: float) -> dict[str, float]:
    """Initialize the final layer at the clamped inner-training target mean."""
    clamped = min(max(float(target_mean), 0.000001), 0.999999)
    bias = math.log(clamped / (1.0 - clamped))
    output = model.regression_head[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("Final head module is not Linear")
    with torch.no_grad():
        output.weight.zero_()
        output.bias.fill_(bias)
    return {"training_target_mean": float(target_mean), "clamped_mean": clamped, "output_bias": bias}

def evaluate(
    model: RawLogitSiamese, loader: DataLoader, criterion: nn.Module, swap_check: bool = False
) -> dict[str, Any]:
    """Evaluate deterministically and retain transparent numerical diagnostics."""
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
            if any(not torch.isfinite(value).all().item() for value in (logit, prediction, embedding_a, embedding_b, comparison)):
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
    logit_values = np.concatenate(logits).astype(np.float32, copy=False)
    prediction_values = np.concatenate(predictions).astype(np.float32, copy=False)
    target_values = np.concatenate(targets).astype(np.float32, copy=False)
    metrics = regression_metrics(target_values, prediction_values)
    return {
        "validation_bce": bce_sum / count, **metrics,
        "prediction_minimum": float(np.min(prediction_values)),
        "prediction_maximum": float(np.max(prediction_values)),
        "prediction_mean": float(np.mean(prediction_values, dtype=np.float64)),
        "prediction_standard_deviation": float(np.std(prediction_values.astype(np.float64), ddof=0)),
        "logit_minimum": float(np.min(logit_values)), "logit_maximum": float(np.max(logit_values)),
        "fraction_below": float(np.mean(prediction_values < 0.000001)),
        "fraction_above": float(np.mean(prediction_values > 0.999999)),
        "all_finite": bool(np.isfinite(logit_values).all() and np.isfinite(prediction_values).all()),
        "swap_maximum_absolute_difference": swap_maximum,
        "logits": logit_values, "predictions": prediction_values, "targets": target_values,
    }

def fit_identifier(
    holdout: str, condition: str, validation: str, candidate: str, preprocessing: str, seed: int
) -> str:
    """Return a deterministic new-fit identifier."""
    return f"holdout_{holdout}__{condition}__validate_{validation}__candidate_{candidate}__{preprocessing}__seed_{seed}"

def run_fit(
    fit_id: str,
    spec: CandidateSpec,
    seed: int,
    train_a: np.ndarray,
    train_b: np.ndarray,
    train_targets: np.ndarray,
    validation_a: np.ndarray,
    validation_b: np.ndarray,
    validation_targets: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Train one corrected candidate and restore its best validation checkpoint."""
    set_deterministic(seed, int(settings["torch_threads"]))
    model = RawLogitSiamese(spec, 60)
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
        raise RuntimeError("Initial prediction differs from the inner-training target mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    history: list[dict[str, Any]] = []
    best_mae = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    early_stopping_epoch: int | None = None
    fit_started = time.perf_counter()
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        epoch_started = time.perf_counter()
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
        history.append({
            "epoch": epoch, "training_loss": loss_sum / count,
            "validation_bce": validation["validation_bce"], "validation_mae": validation["mae"],
            "validation_rmse": validation["rmse"], "validation_r2": validation["r2"],
            "validation_spearman": validation["spearman"],
            "spearman_missing_reason": validation["spearman_missing_reason"],
            "prediction_minimum": validation["prediction_minimum"], "prediction_maximum": validation["prediction_maximum"],
            "prediction_mean": validation["prediction_mean"],
            "prediction_standard_deviation": validation["prediction_standard_deviation"],
            "logit_minimum": validation["logit_minimum"], "logit_maximum": validation["logit_maximum"],
            "fraction_below": validation["fraction_below"], "fraction_above": validation["fraction_above"],
            "gradient_global_mean": float(np.mean([item["gradient_global"] for item in gradient_rows])),
            "gradient_encoder_mean": float(np.mean([item["gradient_encoder"] for item in gradient_rows])),
            "gradient_head_mean": float(np.mean([item["gradient_head"] for item in gradient_rows])),
            "gradient_output_mean": float(np.mean([item["gradient_output"] for item in gradient_rows])),
            "all_finite": bool(finite and validation["all_finite"] and all_parameters_finite(model)),
            "improved_best_mae": improved, "patience_state": patience,
            "shuffle_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        if patience >= int(settings["patience"]):
            early_stopping_epoch = epoch
            break
    if best_state is None:
        raise RuntimeError("No best state was saved")
    model.load_state_dict(best_state)
    restored = evaluate(model, validation_loader, criterion, swap_check=True)
    reasons: list[str] = []
    if restored["prediction_maximum"] - restored["prediction_minimum"] <= 1.0e-8:
        reasons.append("prediction_range_le_1e-8")
    if restored["prediction_standard_deviation"] <= 1.0e-8:
        reasons.append("prediction_sd_le_1e-8")
    if restored["fraction_below"] > 0.999 or restored["fraction_above"] > 0.999:
        reasons.append("single_extreme_tail_gt_0.999")
    if not restored["all_finite"] or not all_parameters_finite(model):
        reasons.append("nonfinite")
    restored_hash = state_hash(model)
    for row in history:
        row.update({
            "selected_best_epoch": best_epoch, "early_stopping_epoch": early_stopping_epoch,
            "is_selected_best_epoch": row["epoch"] == best_epoch,
            "restored_best_state_hash": restored_hash,
        })
    return {
        "fit_id": fit_id, "candidate": spec.candidate, "seed": seed, "history": history,
        "restored": restored, "best_epoch": best_epoch, "early_stopping_epoch": early_stopping_epoch,
        "epochs_completed": len(history), "state_hash": restored_hash, "initial_state_hash": initial_state_hash,
        "initialization": initialization, "initial_prediction_difference": initial_difference,
        "degenerate": bool(reasons), "degenerate_reasons": ";".join(reasons),
        "fit_seconds": time.perf_counter() - fit_started,
    }

def parameter_count(spec: CandidateSpec) -> int:
    """Return the exact trainable parameter count for 60 SSI inputs."""
    return sum(parameter.numel() for parameter in RawLogitSiamese(spec, 60).parameters())
