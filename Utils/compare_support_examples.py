import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from tslearn.metrics import dtw

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Utils.load_data import load_dataset, load_dataset_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_rules", type=int, default=2)
    parser.add_argument("--prototype_mode", type=str, default="baseline", choices=["baseline", "rulebased"])
    parser.add_argument("--random_mode", type=str, default="baselineNoPrototype", choices=["baselineNoPrototype", "noPrototype"])
    parser.add_argument("--prototype_repetition", type=int, default=None)
    parser.add_argument("--random_repetition", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_pdf", type=str, default=None)
    return parser.parse_args()


def load_runs(dataset: str, mode: str, k: int, num_rules: int) -> list[dict]:
    path = f"results/llm_results/{dataset}_{mode}_{k}_{num_rules}_llm_results.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    runs = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            if row["k"] == k and row["num_rules"] == num_rules:
                runs.append(row)
    return runs


def pick_run(runs: list[dict], repetition: int | None=None) -> dict:
    if repetition is not None:
        matching = [run for run in runs if run.get("repetition") == repetition]
        if not matching:
            raise ValueError(f"No run found for repetition={repetition}")
        run = matching[-1]
        if not run.get("support_examples"):
            raise ValueError(f"Run repetition={repetition} does not contain support_examples. Rerun the experiment with the current code.")
        return run

    for run in reversed(runs):
        if run.get("support_examples"):
            return run

    raise ValueError("No run with support_examples was found. Rerun the experiment with the current code.")


def compute_class_medoids(dataset: str, data_type: str="TRAIN_normalized") -> tuple[np.ndarray, np.ndarray, dict]:
    dataset_ts = load_dataset(dataset, data_type=data_type)
    labels = np.array(load_dataset_labels(dataset, data_type=data_type))
    medoids = {}

    for label in np.unique(labels):
        label = int(label)
        class_indices = np.where(labels == label)[0]
        class_ts = dataset_ts[class_indices]
        km = KMedoids(n_clusters=1, metric=dtw, init="random", random_state=42)  # type: ignore
        km.fit(class_ts)
        local_medoid_idx = int(km.medoid_indices_[0])
        global_medoid_idx = int(class_indices[local_medoid_idx])
        center_ts = dataset_ts[global_medoid_idx]
        center_distances = np.array([dtw(ts, center_ts) for ts in class_ts])
        medoids[label] = {
            "global_idx": global_medoid_idx,
            "center_ts": center_ts,
            "class_indices": class_indices,
            "class_ts": class_ts,
            "center_distances": center_distances,
        }

    return dataset_ts, labels, medoids


def support_examples_to_map(run: dict) -> dict[int, list[int]]:
    return {int(item["class_label"]): [int(idx) for idx in item["indices"]] for item in run["support_examples"]}


def build_distance_table(dataset: str, prototype_run: dict, random_run: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    dataset_ts, labels, medoids = compute_class_medoids(dataset)
    prototype_map = support_examples_to_map(prototype_run)
    random_map = support_examples_to_map(random_run)
    rows = []

    for selection_name, support_map in [("prototype", prototype_map), ("random", random_map)]:
        for label, support_indices in support_map.items():
            center_ts = medoids[label]["center_ts"]
            class_ts = medoids[label]["class_ts"]
            center_distances = medoids[label]["center_distances"]
            prototype_indices = prototype_map[label]
            prototype_ts = [dataset_ts[idx] for idx in prototype_indices]
            nearest_prototype_distances = np.array([min(dtw(ts, proto) for proto in prototype_ts) for ts in class_ts])

            for support_idx in support_indices:
                support_ts = dataset_ts[support_idx]
                dist_to_center = dtw(support_ts, center_ts)
                dist_to_nearest_prototype = min(dtw(support_ts, proto) for proto in prototype_ts)
                rows.append({
                    "selection": selection_name,
                    "class_label": label,
                    "train_idx": int(support_idx),
                    "dist_to_center": float(dist_to_center),
                    "center_percentile": float((center_distances <= dist_to_center).mean()),
                    "dist_to_nearest_prototype": float(dist_to_nearest_prototype),
                    "nearest_prototype_percentile": float((nearest_prototype_distances <= dist_to_nearest_prototype).mean()),
                })

    return pd.DataFrame(rows), medoids, dataset_ts


def add_distance_plots(pdf: PdfPages, distance_df: pd.DataFrame, dataset: str, prototype_mode: str, random_mode: str, k: int):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    labels = sorted(distance_df["class_label"].unique())
    colors = {"prototype": "#1f77b4", "random": "#d62728"}
    offsets = {"prototype": -0.08, "random": 0.08}

    for selection, group in distance_df.groupby("selection"):
        x = np.array(group["class_label"], dtype=float) + offsets[selection]
        axes[0].scatter(x, group["dist_to_center"], alpha=0.9, label=selection, color=colors[selection])
        axes[1].scatter(x, group["center_percentile"], alpha=0.9, label=selection, color=colors[selection])
        axes[2].scatter(x, group["dist_to_nearest_prototype"], alpha=0.9, label=selection, color=colors[selection])

    axes[0].set_title("DTW Distance to Class Medoid")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("DTW distance")
    axes[1].set_title("Position Within Class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Percentile vs class medoid")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_title("DTW Distance to Nearest Prototype")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("DTW distance")

    for ax in axes:
        ax.set_xticks(labels)
        ax.grid(alpha=0.2, linestyle="--")

    axes[0].legend(loc="best")
    fig.suptitle(f"{dataset} - {prototype_mode} vs {random_mode} support examples (k={k})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_support_grid(pdf: PdfPages, dataset_ts: np.ndarray, medoids: dict, prototype_run: dict, random_run: dict, dataset: str):
    prototype_map = support_examples_to_map(prototype_run)
    random_map = support_examples_to_map(random_run)
    labels = sorted(medoids.keys())
    prototype_k = max(len(idxs) for idxs in prototype_map.values())
    random_k = max(len(idxs) for idxs in random_map.values())
    total_cols = 1 + prototype_k + random_k

    fig, axes = plt.subplots(len(labels), total_cols, figsize=(3.4 * total_cols, 2.6 * len(labels)), squeeze=False)
    for row_idx, label in enumerate(labels):
        center_idx = medoids[label]["global_idx"]
        center_ax = axes[row_idx][0]
        center_ax.plot(dataset_ts[center_idx])
        center_ax.set_title(f"Class {label} - Medoid\nidx {center_idx}")
        center_ax.set_xticks([])
        center_ax.set_yticks([])

        for col_idx, support_idx in enumerate(prototype_map[label], start=1):
            ax = axes[row_idx][col_idx]
            ax.plot(dataset_ts[support_idx])
            ax.set_title(f"P{col_idx}\nidx {support_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

        for col_idx in range(len(prototype_map[label]) + 1, prototype_k + 1):
            axes[row_idx][col_idx].axis("off")

        random_start = 1 + prototype_k
        for offset, support_idx in enumerate(random_map[label]):
            ax = axes[row_idx][random_start + offset]
            ax.plot(dataset_ts[support_idx])
            ax.set_title(f"R{offset + 1}\nidx {support_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

        for col_idx in range(random_start + len(random_map[label]), total_cols):
            axes[row_idx][col_idx].axis("off")

    fig.suptitle(f"{dataset} - medoids, prototypes, and random support examples")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    args = parse_args()
    prototype_runs = load_runs(args.dataset, args.prototype_mode, args.k, args.num_rules)
    random_runs = load_runs(args.dataset, args.random_mode, args.k, args.num_rules)
    prototype_run = pick_run(prototype_runs, repetition=args.prototype_repetition)
    random_run = pick_run(random_runs, repetition=args.random_repetition)

    distance_df, medoids, dataset_ts = build_distance_table(args.dataset, prototype_run, random_run)
    output_prefix = f"{args.dataset}_{args.prototype_mode}_vs_{args.random_mode}_k{args.k}_r{args.num_rules}"
    output_csv = args.output_csv or f"results/llm_explore/{output_prefix}.csv"
    output_pdf = args.output_pdf or f"results/llm_explore/{output_prefix}.pdf"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    distance_df.to_csv(output_csv, index=False)

    with PdfPages(output_pdf) as pdf:
        add_distance_plots(pdf, distance_df, args.dataset, args.prototype_mode, args.random_mode, args.k)
        add_support_grid(pdf, dataset_ts, medoids, prototype_run, random_run, args.dataset)

    print(f"Saved distance summary to {output_csv}")
    print(f"Saved comparison plots to {output_pdf}")


if __name__ == "__main__":
    main()
