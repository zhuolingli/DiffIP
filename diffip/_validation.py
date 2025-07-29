"""Internal input validation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .types import FloatArray


def validate_sequence(name: str, values: npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (time_steps, samples, features)")
    time_steps, samples, features = array.shape
    if time_steps < 1 or samples < 2 or features < 1:
        raise ValueError(
            f"{name} requires at least 1 time step, 2 samples, and 1 feature"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_step(name: str, values: npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, features)")
    samples, features = array.shape
    if samples < 2 or features < 1:
        raise ValueError(f"{name} requires at least 2 samples and 1 feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array
