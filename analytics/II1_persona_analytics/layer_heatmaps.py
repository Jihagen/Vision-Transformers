"""
Layer-wise heatmaps: locate where persona-feature vectors appear in the model.

For each metric (GDV, UMAP trustworthiness, t-SNE trustworthiness, probe accuracy,
projection AUROC) plot the value as a function of layer index, separately for
vision, projector, and language model components.

Migrated from analysis/plot_layer_curves.py.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

# import matplotlib.pyplot as plt


def plot_layer_curve(
    values_by_layer: dict[str, float],   # {layer_key: metric_value}
    metric_name: str,
    title: str = "",
    save_path: str | Path | None = None,
):
    """
    Line plot of metric value vs layer index, with vision / projector / LLM
    sections visually separated.
    """
    raise NotImplementedError


def plot_layer_heatmap(
    values_by_condition_and_layer: dict[str, dict[str, float]],
    # {condition_name: {layer_key: value}}
    metric_name: str,
    save_path: str | Path | None = None,
):
    """
    2D heatmap: conditions (rows) × layers (columns), cell = metric value.

    Useful for comparing where each persona feature has its strongest signal.
    """
    raise NotImplementedError


def run_layer_heatmap_pipeline(
    gdv_results: dict[str, dict],           # {condition: {layer: {metric: value}}}
    umap_results: dict[str, dict] | None = None,
    tsne_results: dict[str, dict] | None = None,
    probe_results=None,
    save_dir: str | Path | None = None,
):
    """
    Generate all layer-curve and heatmap figures for a set of conditions.
    """
    raise NotImplementedError
