"""Dynamic-programming alignment of diffusion time steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .types import FloatArray


@dataclass(frozen=True)
class DynamicAlignmentResult:
    distance: float
    accumulated_costs: FloatArray
    path: tuple[tuple[int, int], ...]


def align_local_distances(local_distances: npt.ArrayLike) -> DynamicAlignmentResult:
    """Find the minimum-cost monotone path through a local-distance matrix."""
    local = np.asarray(local_distances, dtype=np.float64)
    if local.ndim != 2 or min(local.shape, default=0) < 1:
        raise ValueError("local_distances must be a non-empty two-dimensional matrix")
    if not np.isfinite(local).all():
        raise ValueError("local_distances must contain only finite values")
    if (local < 0).any():
        raise ValueError("local_distances cannot contain negative values")

    rows, columns = local.shape
    padded = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    padded[0, 0] = 0.0
    parents = np.full((rows, columns), -1, dtype=np.int8)

    for row in range(rows):
        for column in range(columns):
            candidates = (
                padded[row, column],
                padded[row, column + 1],
                padded[row + 1, column],
            )
            parent = int(np.argmin(candidates))
            parents[row, column] = parent
            padded[row + 1, column + 1] = local[row, column] + candidates[parent]

    row, column = rows - 1, columns - 1
    reversed_path = [(row, column)]
    while row > 0 or column > 0:
        parent = parents[row, column]
        if parent == 0:
            row -= 1
            column -= 1
        elif parent == 1:
            row -= 1
        else:
            column -= 1
        reversed_path.append((row, column))

    return DynamicAlignmentResult(
        distance=float(padded[rows, columns]),
        accumulated_costs=padded[1:, 1:],
        path=tuple(reversed(reversed_path)),
    )
