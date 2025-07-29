"""Visualize the three key properties of DiffIP."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from diffip import DiffIPMetric, align_local_distances, fit_step_alignment


def unreverted_similarity(suspect: np.ndarray, victim: np.ndarray) -> float:
    distance = np.mean(np.sum((suspect - victim) ** 2, axis=1))
    return math.exp(-float(distance))


def main(output_dir: Path) -> None:
    generator = np.random.default_rng(18)
    victim_step = generator.normal(size=(48, 6))
    suspect_step = victim_step + generator.normal(scale=0.02, size=victim_step.shape)
    metric = DiffIPMetric(n_components=None)

    permutation = np.array([3, 0, 5, 1, 4, 2])
    permuted_step = suspect_step[:, permutation]
    unreverted_permutation = [
        unreverted_similarity(suspect_step, victim_step),
        unreverted_similarity(permuted_step, victim_step),
    ]
    diffip_permutation = [
        metric.compare(suspect_step[None], victim_step[None]).similarity,
        metric.compare(permuted_step[None], victim_step[None]).similarity,
    ]

    scaling_strengths = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    unreverted_scaling = []
    diffip_scaling = []
    for strength in scaling_strengths:
        column_scales = np.ones(victim_step.shape[1])
        column_scales[0] = strength
        scaled_step = suspect_step * column_scales
        unreverted_scaling.append(unreverted_similarity(scaled_step, victim_step))
        diffip_scaling.append(
            metric.compare(scaled_step[None], victim_step[None]).similarity
        )

    temporal_generator = np.random.default_rng(21)
    time_steps = 11
    suspect_steps = 7
    samples = 24
    features = 6
    sample_latents = temporal_generator.normal(size=(samples, features))
    victim = np.empty((time_steps, samples, features), dtype=np.float64)
    for index, phase in enumerate(np.linspace(0.0, 1.0, time_steps)):
        weights = 0.45 + 0.8 * phase + 0.15 * np.sin(
            np.arange(features) + 2.0 * np.pi * phase
        )
        drift = 0.3 * np.cos(np.arange(features) * 0.7 + phase * np.pi)
        victim[index] = 2.5 * (sample_latents * weights + drift)

    rotation, _ = np.linalg.qr(
        temporal_generator.normal(size=(features, features))
    )
    scale = np.linspace(0.65, 1.35, features)
    translation = temporal_generator.normal(scale=0.3, size=features)
    normalized_time = np.linspace(0.0, 1.0, suspect_steps)
    warp = np.rint(normalized_time**1.35 * (time_steps - 1)).astype(int)
    fewer_step_suspect = (victim[warp] @ rotation) / scale + translation
    fewer_step_suspect += temporal_generator.normal(
        scale=0.02, size=fewer_step_suspect.shape
    )
    local_distances = np.empty((suspect_steps, time_steps), dtype=np.float64)
    for suspect_index in range(suspect_steps):
        for victim_index in range(time_steps):
            local_distances[suspect_index, victim_index] = fit_step_alignment(
                fewer_step_suspect[suspect_index], victim[victim_index]
            ).distance
    temporal_alignment = align_local_distances(local_distances)
    path_y, path_x = zip(*temporal_alignment.path, strict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["hatch.linewidth"] = 0.5
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))
    without_reversion_color = "#4c78a8"
    diffip_color = "#e06c9f"
    method_colors = [without_reversion_color, diffip_color]

    permutation_methods = ["DiffIP w/o reversion", "DiffIP"]
    original_scores = [
        unreverted_permutation[0],
        diffip_permutation[0],
    ]
    permuted_scores = [
        unreverted_permutation[1],
        diffip_permutation[1],
    ]
    positions = np.arange(len(permutation_methods))
    width = 0.34
    axes[0].bar(
        positions - width / 2,
        original_scores,
        width,
        color=method_colors,
        edgecolor="black",
        linewidth=0.9,
    )
    axes[0].bar(
        positions + width / 2,
        permuted_scores,
        width,
        color=method_colors,
        edgecolor="black",
        linewidth=0.9,
        hatch="//",
    )
    axes[0].set_xticks(positions, permutation_methods)
    axes[0].set_ylim(0.0, 1.10)
    axes[0].set_ylabel("Similarity")
    axes[0].set_title("(a) Dimension permutation")
    bar_groups = (
        (positions - width / 2, original_scores),
        (positions + width / 2, permuted_scores),
    )
    for bar_positions, scores in bar_groups:
        for position, score in zip(bar_positions, scores, strict=True):
            axes[0].annotate(
                f"{score:.2f}",
                (position, score),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="black",
                fontweight="bold",
            )
    state_handles = [
        Patch(facecolor="black", edgecolor="black", label="Original"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Permuted"),
    ]
    axes[0].legend(
        handles=state_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
    )

    axes[1].plot(
        scaling_strengths,
        unreverted_scaling,
        color=without_reversion_color,
        marker="o",
        linewidth=2,
        label="DiffIP w/o reversion",
    )
    axes[1].plot(
        scaling_strengths,
        diffip_scaling,
        color=diffip_color,
        marker="o",
        linewidth=2,
        label="DiffIP",
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Column-scaling strength")
    axes[1].set_ylabel("Similarity")
    axes[1].set_title("(b) Column-wise scaling")
    axes[1].legend(frameon=False, loc="lower right")

    image = axes[2].imshow(
        local_distances, cmap="Greens", aspect="auto"
    )
    axes[2].plot(path_x, path_y, color="black", marker="o", linewidth=2)
    axes[2].set_xlabel("Full-step victim")
    axes[2].set_ylabel("Fewer-step suspect")
    axes[2].set_title("(c) Temporal alignment")
    figure.colorbar(image, ax=axes[2], label="Local distance", fraction=0.046)

    for axis in axes[:2]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(
        output_dir / "key_properties.png",
        dpi=200,
        metadata={"Software": None},
    )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    arguments = parser.parse_args()
    main(arguments.output_dir)
