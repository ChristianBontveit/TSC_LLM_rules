# TSC_LLM_rules

Research code for **rule-based interpretability of Time Series Classification (TSC)** with help from Large Language Models (LLMs).

This repo is connected to the Obsidian project note:
- `~/Documents/Obsedian/Side Research/XAI_TS/Rule-based Interpretability of TSC with LLMs.md`

## Idea (high level)

Use an LLM as a **rule synthesizer** to produce simple, human-readable decision rules for time series classification.

Typical setup:
- select a small set of **prototypes/examples** per class
- prompt an LLM to produce either:
  - a natural-language rule, or
  - minimal Python code implementing the rule
- evaluate that rule on held-out samples

Metrics you care about (from the Obsidian note):
- accuracy
- coverage
- confidence
- simplicity

## Repo layout

- `train_models.py` — trains baseline models (CNN, decision tree, logistic regression, etc.)
- `prompt_simplifications.py` — prompts an LLM with prototype plots and evaluates predictions
- `Utils/` — dataset loading, metrics, plotting, helpers
- `data/` — datasets
- `results/` — result tables/notebooks

## Installation

Python project defined in `pyproject.toml`.

- Python: `>=3.13,<3.15`

### Poetry

```bash
cd TSC_LLM_rules
poetry install
poetry shell
```

## Data layout (what the code expects)

Datasets live under:

- `data/<DatasetName>/`

And are stored as numpy arrays with the label in the **first column**:

- `data/<DatasetName>/<DatasetName>_TRAIN.npy`
- `data/<DatasetName>/<DatasetName>_TEST.npy`
- optionally:
  - `data/<DatasetName>/<DatasetName>_VALIDATION.npy`

The helper `Utils/load_data.py` loads those files and then removes column 0 for features, and uses column 0 as labels.

If you want normalization, the code expects normalized variants:
- `<DatasetName>_TRAIN_normalized.npy`, etc.

## Baselines: training models

Run on one dataset:

```bash
python train_models.py --datasets Chinatown --model_type cnn --normalized
```

Run on several:

```bash
python train_models.py --datasets Chinatown ECG200 ItalyPowerDemand --model_type decision-tree --normalized
```

If you omit `--datasets`, it will run over every folder under `./data/`.

Model types supported in the training script include:
- `cnn`
- `miniRocket`
- `decision-tree`
- `logistic-regression`
- `knn`

Save trained models and per-dataset metrics:

```bash
python train_models.py --datasets Chinatown --model_type cnn --normalized --save_model
```

Saved artifacts (by convention in the code):
- models under `models/<DatasetName>/...`
- metrics CSV under `results/<DatasetName>/models.csv`

## LLM prompting experiments

`prompt_simplifications.py` builds a prompt using prototype plots and asks an LLM to classify 10 random test samples.

It expects an `.env` file in repo root with at least:
- `API_KEY`
- `API_BASE`

Run:

```bash
python prompt_simplifications.py --dataset Chinatown --classifier cnn --llm gpt4o --k 3
```

Options:
- `--interactive` prints the prompt structure and pauses between steps.

## What I intentionally did NOT include here

- A promise of reproducibility across machines (LLM endpoints + API_BASE make this environment-specific)
- A single "official" dataset list (the scripts can run on any dataset that follows the `data/<name>/<name>_*.npy` convention)

If you want, next step is to add:
- a `scripts/download_ucr.sh` (or similar) to fetch datasets automatically,
- a single `run_all.sh` that produces a consolidated `results/results.csv`.
