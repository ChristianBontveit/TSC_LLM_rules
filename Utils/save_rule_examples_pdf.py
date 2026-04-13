import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Utils.load_data import load_dataset_labels
from Utils.selectPrototypes import select_prototypes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_rules", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--all_rules", action="store_true")
    return parser.parse_args()


def load_runs(dataset: str, mode: str, k: int, num_rules: int) -> list[dict]:
    path = f"results/llm_results/{dataset}_{mode}_{k}_{num_rules}_llm_results.jsonl"
    runs = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            if row["k"] == k and row["num_rules"] == num_rules:
                runs.append(row)
    return runs


def rules_to_text(rules_dict: dict) -> str:
    keys = sorted(rules_dict.keys(), key=lambda x: int(x.split("_")[-1]))
    lines = []
    for key in keys:
        label = key.split("_")[-1]
        lines.append(f"Class {label}:")
        lines.append(rules_dict[key].strip())
        lines.append("")
    return "\n".join(lines).strip()


def add_prototypes_page(pdf: PdfPages, dataset: str, k: int):
    prototypes = select_prototypes(dataset, num_instances=k, data_type="TRAIN_normalized")
    num_labels = len(set(load_dataset_labels(dataset, data_type="TRAIN_normalized")))

    fig, axes = plt.subplots(num_labels, k, figsize=(4 * k, 2.6 * num_labels), squeeze=False)
    idx = 0
    for label in range(num_labels):
        for proto_idx in range(k):
            ax = axes[label][proto_idx]
            ax.plot(prototypes[idx])
            ax.set_title(f"Class {label} - P{proto_idx + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1

    fig.suptitle(f"{dataset} - Rulebased prototypes (k={k})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_rules_page(pdf: PdfPages, title: str, rules_text: str):
    height = max(3.0, 1.2 + 0.32 * len((title + "\n" + rules_text).splitlines()))
    fig = plt.figure(figsize=(8.27, height))
    fig.text(0.05, 0.97, title, fontsize=15, va="top")
    fig.text(0.05, 0.93, rules_text, fontsize=10, family="monospace", va="top", wrap=True)
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    args = parse_args()
    output = args.output or f"results/llm_results/{args.dataset}_rule_examples_k{args.k}_r{args.num_rules}.pdf"

    rulebased_runs = load_runs(args.dataset, "rulebased", args.k, args.num_rules)
    no_prototype_runs = load_runs(args.dataset, "noPrototype", args.k, args.num_rules)

    if not args.all_rules:
        rulebased_runs = [rulebased_runs[-1]]
        no_prototype_runs = [no_prototype_runs[-1]]

    with PdfPages(output) as pdf:
        for idx, run in enumerate(no_prototype_runs, start=1):
            add_rules_page(
                pdf,
                f"{args.dataset} - noPrototype rules #{idx} (k={args.k}, num_rules={args.num_rules}, acc={run['accuracy']:.2f})",
                rules_to_text(run["extracted_rules"]),
            )

        add_prototypes_page(pdf, args.dataset, args.k)

        for idx, run in enumerate(rulebased_runs, start=1):
            add_rules_page(
                pdf,
                f"{args.dataset} - rulebased rules #{idx} (k={args.k}, num_rules={args.num_rules}, acc={run['accuracy']:.2f})",
                rules_to_text(run["extracted_rules"]),
            )


if __name__ == "__main__":
    main()
