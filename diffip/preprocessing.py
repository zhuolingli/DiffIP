"""Independent PCA projection for suspect and victim sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy.typing as npt
from sklearn.decomposition import PCA

from ._validation import validate_sequence
from .types import FloatArray


@dataclass(frozen=True)
class ProjectionResult:
    suspect: FloatArray
    victim: FloatArray
    n_components: int


def project_sequences(
    suspect: npt.ArrayLike,
    victim: npt.ArrayLike,
    n_components: int | None = 20,
) -> ProjectionResult:
    """Validate two sequences and optionally project each with its own PCA."""
    suspect_array = validate_sequence("suspect", suspect)
    victim_array = validate_sequence("victim", victim)
    if suspect_array.shape[1] != victim_array.shape[1]:
        raise ValueError("suspect and victim must contain the same number of samples")

    if n_components is None:
        if suspect_array.shape[2] != victim_array.shape[2]:
            raise ValueError(
                "feature dimensions must match when PCA projection is disabled"
            )
        return ProjectionResult(
            suspect=suspect_array.copy(),
            victim=victim_array.copy(),
            n_components=suspect_array.shape[2],
        )
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise TypeError("n_components must be a positive integer or None")
    if n_components < 1:
        raise ValueError("n_components must be positive")

    suspect_flat = suspect_array.reshape(-1, suspect_array.shape[2])
    victim_flat = victim_array.reshape(-1, victim_array.shape[2])
    used_components = min(
        n_components,
        suspect_flat.shape[0] - 1,
        victim_flat.shape[0] - 1,
        suspect_flat.shape[1],
        victim_flat.shape[1],
    )
    if used_components < 1:
        raise ValueError("not enough observations for PCA projection")

    suspect_projected = PCA(n_components=used_components).fit_transform(suspect_flat)
    victim_projected = PCA(n_components=used_components).fit_transform(victim_flat)
    return ProjectionResult(
        suspect=suspect_projected.reshape(
            suspect_array.shape[0], suspect_array.shape[1], used_components
        ),
        victim=victim_projected.reshape(
            victim_array.shape[0], victim_array.shape[1], used_components
        ),
        n_components=used_components,
    )
