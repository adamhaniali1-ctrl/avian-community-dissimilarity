from __future__ import annotations

import hashlib

import math

import os

import random

import tempfile

from dataclasses import dataclass

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from torch import nn

SSI_COLUMNS = [f"ssi_{index:03d}" for index in range(60)]

STAGE_PREFIXES = (
    "config/final_fit_and_seal.yaml",
    "scripts/final_fit_utils.py",
    "scripts/run_final_fit_and_seal.py",
    "scripts/fit_final_gdm.R",
    "scripts/validate_final_fit_and_seal.py",
    "scripts/validate_final_gdm.R",
    "audit_reports/final_fit_and_seal_audit.md",
    "results/final_fit_and_seal",
)

@dataclass(frozen=True)
class Architecture:
    """One locked Siamese architecture."""

    name: str
    hidden_1: int
    hidden_2: int
    embedding: int
    dropout: float
    learning_rate: float
    weight_decay: float

ARCHITECTURES = {
    "A": Architecture("A", 128, 64, 32, 0.10, 0.0003, 0.0001),
    "B": Architecture("B", 256, 128, 64, 0.20, 0.0003, 0.0001),
    "C": Architecture("C", 128, 64, 32, 0.20, 0.0003, 0.0001),
    "D": Architecture("D", 256, 128, 64, 0.30, 0.0003, 0.001),
}

class RawLogitSiamese(nn.Module):
    """Validated shared-weight SSI Siamese regressor."""

    def __init__(self, spec: Architecture) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(60, spec.hidden_1), nn.ReLU(), nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_1, spec.hidden_2), nn.ReLU(), nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_2, spec.embedding),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(2 * spec.embedding, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, endpoint_a: torch.Tensor, endpoint_b: torch.Tensor) -> torch.Tensor:
        embedding_a = self.encoder(endpoint_a)
        embedding_b = self.encoder(endpoint_b)
        comparison = torch.cat(
            (torch.abs(embedding_a - embedding_b), embedding_a * embedding_b), dim=1
        )
        return self.regression_head(comparison).squeeze(1)

def sha256_file(path: Path) -> str:
    """Return a file SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_bytes(path: Path, payload: bytes) -> None:
    """Commit bytes from a system temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="final_fit_", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_text(path: Path, payload: str) -> None:
    """Commit UTF-8 text atomically."""
    if not payload.endswith("\n"):
        payload += "\n"
    atomic_bytes(path, payload.encode("utf-8"))

def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Commit a deterministic CSV atomically."""
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n", float_format="%.17g"))

def set_deterministic(seed: int) -> None:
    """Configure the validated deterministic CPU execution policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

def state_hash(model: nn.Module) -> str:
    """Hash a model state canonically."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()

def initialize_output(model: RawLogitSiamese, target_mean: float) -> float:
    """Initialize the output at the clamped final-training target mean."""
    clamped = min(max(float(target_mean), 0.000001), 0.999999)
    bias = math.log(clamped / (1.0 - clamped))
    output = model.regression_head[-1]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.fill_(bias)
    return bias

def fit_preprocessing(method: str, values: np.ndarray) -> dict[str, np.ndarray]:
    """Fit locked preprocessing on unique final-training events."""
    raw = np.asarray(values, dtype=np.float64)
    if method == "STANDARD":
        feature_scale = np.ones(raw.shape[1], dtype=np.float64)
        transformed = raw
    elif method == "ASINH_STANDARD":
        feature_scale = np.empty(raw.shape[1], dtype=np.float64)
        for index in range(raw.shape[1]):
            nonzero = np.abs(raw[:, index][raw[:, index] != 0.0])
            feature_scale[index] = float(np.median(nonzero)) if len(nonzero) else 1.0
        transformed = np.arcsinh(raw / feature_scale)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    mean = np.mean(transformed, axis=0, dtype=np.float64)
    variance = np.var(transformed, axis=0, ddof=0, dtype=np.float64)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    if not all(np.isfinite(item).all() for item in (feature_scale, mean, variance, scale)):
        raise FloatingPointError("Nonfinite preprocessing parameter")
    return {"feature_scale": feature_scale, "mean": mean, "variance": variance, "scale": scale}

def apply_preprocessing(method: str, values: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    """Apply locked preprocessing without refitting or clipping."""
    raw = np.asarray(values, dtype=np.float64)
    transformed = raw if method == "STANDARD" else np.arcsinh(raw / fit["feature_scale"])
    result = ((transformed - fit["mean"]) / fit["scale"]).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite transformed SSI value")
    return result

def predict_model(model: RawLogitSiamese, a: np.ndarray, b: np.ndarray, batch: int = 1024) -> tuple[np.ndarray, np.ndarray, float]:
    """Return logits, predictions, and exact endpoint-swap difference."""
    model.eval()
    logits = []
    predictions = []
    swap_maximum = 0.0
    with torch.no_grad():
        for start in range(0, len(a), batch):
            ta = torch.from_numpy(a[start:start + batch])
            tb = torch.from_numpy(b[start:start + batch])
            logit = model(ta, tb)
            swapped = model(tb, ta)
            swap_maximum = max(swap_maximum, float(torch.max(torch.abs(logit - swapped))))
            pred = torch.sigmoid(logit)
            if not torch.isfinite(logit).all() or not torch.isfinite(pred).all():
                raise FloatingPointError("Nonfinite saved-model prediction")
            logits.append(logit.numpy())
            predictions.append(pred.numpy())
    return np.concatenate(logits), np.concatenate(predictions), swap_maximum

def dataframe_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash selected rows and columns deterministically."""
    text = frame.loc[:, columns].to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(text.encode()).hexdigest()

def all_parameters_finite(model: nn.Module) -> bool:
    """Return whether every parameter is finite."""
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())

def parameter_count(spec: Architecture) -> int:
    """Return trainable parameter count."""
    return sum(parameter.numel() for parameter in RawLogitSiamese(spec).parameters())
