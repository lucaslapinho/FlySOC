"""Headless, scientifically labeled matplotlib artifacts."""

import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_plots(output: Path, metrics: dict[str, Any], neighbor_scores: dict[str, np.ndarray],
                 distributions: dict[str, np.ndarray], unseen: np.ndarray | None) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, item in metrics["retrieval"].items():
        if "by_k" in item:
            ks = [int(k) for k in item["by_k"]]
            ax.plot(ks, [v["precision_at_k"] for v in item["by_k"].values()], marker="o", label=name.upper())
    ax.set(xlabel="Retrieved neighbors K", ylabel="Precision@K", ylim=(0, 1.03),
           title="Chronological retrieval • known families only")
    if ax.lines:
        ax.legend()
    save(fig, output / "retrieval_precision.png")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, values in neighbor_scores.items():
        ax.hist(values, bins=np.linspace(0, 1, 31), histtype="step", linewidth=2, label=name.upper())
    ax.set(xlabel="Top-1 similarity (Fly: Jaccard; PN: cosine)", ylabel="Test alerts",
           title="Similarity to historical alerts • all test families")
    ax.legend()
    save(fig, output / "similarity_distribution.png")

    if unseen is not None and unseen.any() and (~unseen).any():
        fig, axes = plt.subplots(1, len(distributions), figsize=(15, 4), layout="constrained")
        for ax, (name, values) in zip(np.atleast_1d(axes), distributions.items()):
            bins = np.linspace(min(values.min(), 0), max(values.max(), .001), 26)
            ax.hist(values[~unseen], bins=bins, density=True, alpha=.6, label=f"Known (n={(~unseen).sum()})")
            ax.hist(values[unseen], bins=bins, density=True, alpha=.6, label=f"Unseen (n={unseen.sum()})")
            ax.axvline(metrics["novelty"][name]["threshold"], color="black", linestyle="--", label="Training threshold")
            ax.set(title=name, xlabel="Novelty score (higher = less familiar)", ylabel="Density")
            ax.legend(fontsize=8)
        save(fig, output / "novelty_distributions.png")

        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        names = list(distributions)
        x = np.arange(len(names))
        ax.bar(x - .18, [metrics["novelty"][n]["auroc"] for n in names], .36, label="AUROC")
        ax.bar(x + .18, [metrics["novelty"][n]["auprc_average_precision"] for n in names], .36, label="AUPRC (AP)")
        ax.set(xticks=x, xticklabels=names, ylim=(0, 1.08), ylabel="Score", title="Held-out family novelty comparison")
        ax.legend()
        save(fig, output / "novelty_comparison.png")

    class_names = [name for name, item in metrics["classification"].items() if "test" in item]
    if class_names:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for ax, name in zip(axes.flat, class_names):
            item = metrics["classification"][name]["test"]
            cm = np.array(item["confusion_matrix"])
            ax.imshow(cm, cmap="Blues")
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            labels = [label.replace("TRUE_POSITIVE", "TP").replace("FALSE_POSITIVE", "FP") for label in item["labels"]]
            ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                   xlabel="Predicted verdict", ylabel="Actual verdict", title=name)
        save(fig, output / "confusion_matrices.png")
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        ax.bar(class_names, [metrics["classification"][name]["test"]["report"]["macro avg"]["f1-score"] for name in class_names])
        ax.set(ylabel="Macro F1", ylim=(0, 1.04), title="Downstream classification • full chronological test")
        ax.tick_params(axis="x", labelrotation=25)
        save(fig, output / "classification_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    names = list(metrics["deduplication"])
    for ax, key, title in zip(axes, ["reduction_percentage", "cluster_purity", "false_merge_rate"],
                             ["Alert count reduction (%)", "Weighted cluster purity", "Unrelated collapsed pair fraction"]):
        values = [metrics["deduplication"][name].get(key) for name in names]
        ax.bar(names, [value if value is not None else 0 for value in values])
        for i, value in enumerate(values):
            ax.text(i, (value if value is not None else 0) + (1.5 if key == "reduction_percentage" else .02),
                    f"{value:.4f}" if value is not None else "N/A", ha="center", fontsize=9)
        ax.set(title=title, ylim=(0, 100 if key == "reduction_percentage" else 1))
    threshold = metrics["deduplication"][names[0]]["threshold"]
    fig.suptitle(f"Exemplar deduplication • threshold {threshold:g} is not a matched operating point")
    save(fig, output / "alert_reduction.png")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    names = list(metrics["performance"]["retrieval"])
    ax.bar(names, [metrics["performance"]["retrieval"][name]["individual_queries"]["median_ms"] for name in names])
    ax.set(ylabel="Median warm query latency (ms)", title="Exact Top-5 retrieval • 30 individual test queries")
    save(fig, output / "query_latency.png")
