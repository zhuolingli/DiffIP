"""Alternating closed-form representation reversion."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes

from ._validation import validate_step
from .types import FloatArray, StepAlignmentResult


def _objective(
    suspect_columns: FloatArray,
    victim_columns: FloatArray,
    rotation: FloatArray,
    scale: FloatArray,
) -> float:
    residual = rotation @ (scale[:, None] * suspect_columns) - victim_columns
    return float(np.sum(residual * residual))


def fit_step_alignment(
    suspect: npt.ArrayLike,
    victim: npt.ArrayLike,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    rcond: float | None = None,
) -> StepAlignmentResult:
    """Fit representation reversion to two paired sample matrices."""
    suspect_array = validate_step("suspect", suspect)
    victim_array = validate_step("victim", victim)
    if suspect_array.shape != victim_array.shape:
        raise ValueError("suspect and victim steps must have identical shapes")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    suspect_pairs, victim_pairs = suspect_array, victim_array
    suspect_mean = suspect_pairs.mean(axis=0)
    victim_mean = victim_pairs.mean(axis=0)
    suspect_columns = (suspect_pairs - suspect_mean).T
    victim_columns = (victim_pairs - victim_mean).T
    features = suspect_array.shape[1]
    rotation = np.eye(features, dtype=np.float64)
    scale = np.ones(features, dtype=np.float64)
    history: list[float] = []
    converged = False

    for _ in range(max_iterations):
        scaled_rows = (scale[:, None] * suspect_columns).T
        row_rotation, _ = orthogonal_procrustes(scaled_rows, victim_columns.T)
        rotation = row_rotation.T

        normal = (rotation.T @ rotation) * (suspect_columns @ suspect_columns.T)
        right_hand_side = np.diag(
            suspect_columns @ victim_columns.T @ rotation
        )
        scale = np.linalg.lstsq(normal, right_hand_side, rcond=rcond)[0]
        current = _objective(
            suspect_columns, victim_columns, rotation, scale
        )
        history.append(current)
        if len(history) > 1 and abs(history[-2] - current) <= tolerance:
            converged = True
            break

    sample_pairs = suspect_pairs.shape[0]
    return StepAlignmentResult(
        distance=history[-1] / sample_pairs,
        rotation=rotation,
        scale=scale,
        suspect_mean=suspect_mean,
        victim_mean=victim_mean,
        iterations=len(history),
        converged=converged,
        objective_history=tuple(history),
    )
