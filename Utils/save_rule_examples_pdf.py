import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Utils.load_data import load_dataset, load_dataset_labels
from Utils.selectPrototypes import select_prototypes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_rules", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--all_rules", action="store_true")
    parser.add_argument("--prompt_version", type=str, default="promptV0")
    return parser.parse_args()


# def load_runs(dataset: str, mode: str, prompt_version: str ,k: int, num_rules: int) -> list[dict]:
#     path = f"results/llm_results/{dataset}_{mode}_{prompt_version}_{k}_{num_rules}_llm_results.jsonl"
#     runs = []

#     if not os.path.exists(path):
#         print(f"Warning: No results found for {dataset} with mode {mode}, k={k}, num_rules={num_rules}")
#         return runs
    
#     with open(path, "r") as f:
#         for line in f:
#             row = json.loads(line)
#             if row["k"] == k and row["num_rules"] == num_rules:
#                 runs.append(row)
#     return runs

from pathlib import Path

def load_runs(dataset: str, mode: str, prompt_version: str | None = None, k: int = 3, num_rules: int = 3) -> list[dict]:
    base = Path("results/llm_results")
    patterns = []
    # If a specific prompt_version is requested, only include that exact prompt file
    # (older naming convention). Avoid the wildcard pattern which would
    # match other prompt versions (e.g., promptV2 when requesting promptV3).
    if prompt_version:
        patterns.append(f"{dataset}_{mode}_{prompt_version}_{k}_{num_rules}_llm_results.jsonl")
    else:
        # No prompt specified: include both legacy and any prompt-versioned files
        patterns.append(f"{dataset}_{mode}_{k}_{num_rules}_llm_results.jsonl")
        patterns.append(f"{dataset}_{mode}_*_{k}_{num_rules}_llm_results.jsonl")

    seen = set()
    runs = []
    for pat in patterns:
        for path in sorted(base.glob(pat)):
            if path in seen:
                continue
            seen.add(path)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("k") == k and row.get("num_rules") == num_rules:
                        runs.append(row)

    if not runs:
        print(f"Warning: No results found for {dataset} with mode {mode}, k={k}, num_rules={num_rules}")
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
            # Replace the two lines below in both plotting functions
            # ax.set_xticks([])
            # ax.set_yticks([])

            ax.tick_params(axis="both", which="both", labelsize=8)
            idx += 1

    fig.suptitle(f"{dataset} - Rulebased prototypes (k={k})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_support_examples_page(pdf: PdfPages, dataset: str, support_examples: list[dict], title: str, data_type: str="TRAIN_normalized"):
    if not support_examples:
        return

    support_examples = sorted(support_examples, key=lambda x: x["class_label"])
    dataset_ts = load_dataset(dataset, data_type=data_type)
    k = max(len(item["indices"]) for item in support_examples)

    fig, axes = plt.subplots(len(support_examples), k, figsize=(4 * k, 2.6 * len(support_examples)), squeeze=False)
    for row_idx, item in enumerate(support_examples):
        label = item["class_label"]
        indices = item["indices"]
        for col_idx in range(k):
            ax = axes[row_idx][col_idx]
            if col_idx < len(indices):
                ts_idx = indices[col_idx]
                ax.plot(dataset_ts[ts_idx])
                ax.set_title(f"Class {label} - idx {ts_idx}")
            else:
                ax.axis("off")
            # Replace the two lines below in both plotting functions
            # ax.set_xticks([])
            # ax.set_yticks([])

                ax.tick_params(axis="both", which="both", labelsize=8)

    fig.suptitle(title)
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

    if args.prompt_version:
        output = args.output or f"results/llm_results/{args.dataset}_{args.prompt_version}_rule_examples_k{args.k}_r{args.num_rules}.pdf"
    else:
        output = args.output or f"results/llm_results/{args.dataset}_rule_examples_k{args.k}_r{args.num_rules}.pdf"

    rulebased_runs = load_runs(args.dataset, "rulebased", prompt_version=args.prompt_version, k=args.k, num_rules=args.num_rules)
    no_prototype_runs = load_runs(args.dataset, "noPrototype", prompt_version=args.prompt_version, k=args.k, num_rules=args.num_rules)
    baseline_no_prototype_runs = load_runs(args.dataset, "baselineNoPrototype", prompt_version=args.prompt_version, k=args.k, num_rules=args.num_rules)

    if not args.all_rules:
        rulebased_runs = [rulebased_runs[-1]]
        no_prototype_runs = [no_prototype_runs[-1]]
        if baseline_no_prototype_runs:
            baseline_no_prototype_runs = [baseline_no_prototype_runs[-1]]

    with PdfPages(output) as pdf:
        for idx, run in enumerate(baseline_no_prototype_runs, start=1):
            add_support_examples_page(
                pdf,
                args.dataset,
                run.get("support_examples", []),
                f"{args.dataset} - baselineNoPrototype support examples #{idx} (k={args.k}, acc={run['accuracy']:.2f})",
            )

        for idx, run in enumerate(no_prototype_runs, start=1):
            add_support_examples_page(
                pdf,
                args.dataset,
                run.get("support_examples", []),
                f"{args.dataset} - noPrototype support examples #{idx} (k={args.k}, num_rules={args.num_rules}, acc={run['accuracy']:.2f})",
            )
            add_rules_page(
                pdf,
                f"{args.dataset} - noPrototype rules #{idx} (k={args.k}, num_rules={args.num_rules}, acc={run['accuracy']:.2f})",
                rules_to_text(run["extracted_rules"]),
            )

        add_prototypes_page(pdf, args.dataset, args.k)

        for idx, run in enumerate(rulebased_runs, start=1):
            add_support_examples_page(
                pdf,
                args.dataset,
                run.get("support_examples", []),
                f"{args.dataset} - rulebased support examples #{idx} (k={args.k}, num_rules={args.num_rules}, acc={run['accuracy']:.2f})",
            )
            add_rules_page(
                pdf,
                f"{args.dataset} - rulebased rules #{idx} (k={args.k}, num_rules={args.num_rules}, acc={run['accuracy']:.2f})",
                rules_to_text(run["extracted_rules"]),
            )


if __name__ == "__main__":
    main()
