"""Public result types for DiffIP comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class StepAlignmentResult:
    """Optimal reversion map and diagnostics for one time-step pair."""

    distance: float
    rotation: FloatArray
    scale: FloatArray
    suspect_mean: FloatArray
    victim_mean: FloatArray
    iterations: int
    converged: bool
    objective_history: tuple[float, ...]

    def transform(self, samples: npt.ArrayLike) -> FloatArray:
        """Map suspect samples into the victim representation space."""
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.scale.size:
            raise ValueError(
                "samples must have shape (samples, features) matching the fitted map"
            )
        if not np.isfinite(values).all():
            raise ValueError("samples must contain only finite values")
        centered = values - self.suspect_mean
        return (centered * self.scale) @ self.rotation.T + self.victim_mean


@dataclass(frozen=True)
class DiffIPResult:
    """Final DiffIP comparison scores."""

    similarity: float
    distance: float
