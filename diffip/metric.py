"""High-level DiffIP representation-fingerprint metric."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .dynamic import align_local_distances
from .preprocessing import project_sequences
from .reversion import fit_step_alignment
from .types import DiffIPResult


class DiffIPMetric:
    """Compare two diffusion-process representation sequences."""

    def __init__(
        self,
        *,
        n_components: int | None = 20,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
        rcond: float | None = None,
    ) -> None:
        if n_components is not None and (
            isinstance(n_components, bool)
            or not isinstance(n_components, int)
            or n_components < 1
        ):
            raise ValueError("n_components must be a positive integer or None")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.n_components = n_components
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.rcond = rcond

    def compare(
        self, suspect: npt.ArrayLike, victim: npt.ArrayLike
    ) -> DiffIPResult:
        """Return the final DiffIP similarity and distance."""
        projected = project_sequences(suspect, victim, self.n_components)
        suspect_steps = projected.suspect.shape[0]
        victim_steps = projected.victim.shape[0]
        local_rows: list[list[float]] = []

        for suspect_index in range(suspect_steps):
            distance_row: list[float] = []
            for victim_index in range(victim_steps):
                alignment = fit_step_alignment(
                    projected.suspect[suspect_index],
                    projected.victim[victim_index],
                    tolerance=self.tolerance,
                    max_iterations=self.max_iterations,
                    rcond=self.rcond,
                )
                distance_row.append(alignment.distance)
            local_rows.append(distance_row)

        local_distances = np.asarray(local_rows, dtype=np.float64)
        temporal = align_local_distances(local_distances)
        normalization = math.sqrt(suspect_steps * victim_steps)
        similarity = math.exp(-temporal.distance / normalization)
        return DiffIPResult(
            similarity=similarity,
            distance=temporal.distance,
        )
