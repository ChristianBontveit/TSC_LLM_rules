# TSC_LLM_rules

Research code for **rule-based interpretability of Time Series Classification (TSC)** with help from Large Language Models (LLMs).

This repo is connected to the Obsidian project note:
- `~/Documents/Obsedian/Side Research/XAI_TS/Rule-based Interpretability of TSC with LLMs.md`

## Idea (high level)

Use an LLM as a "rule synthesizer" to produce **simple, human-readable decision rules** for time series classification.

Typical setup:
- Provide the LLM with a small set of **prototypes/examples** from each class
- Ask it to generate either:
  - a natural-language rule, or
  - minimal Python code implementing the rule
- Evaluate the rule on a held-out sample of time series and measure:
  - **accuracy**
  - **coverage** (how often the rule applies)
  - **confidence** (consistency)
  - **simplicity** (rule complexity)

## Repo layout

- `train_models.py` — trains baseline models / runs experiments (multiple classifier options)
- `prompt_simplifications.py` — prompting utilities for simpler rules
- `Utils/` — dataset loading, metrics, plotting, model helpers
- `data/` — datasets / summaries
- `results/` — result tables + notebooks

## Requirements

- Python `>=3.13,<3.15` (per `pyproject.toml`)

## Install

This is a `pyproject.toml` project.

### Option 1: Poetry

```bash
cd TSC_LLM_rules
poetry install
poetry shell
```

### Option 2: pip (less reproducible)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run (typical)

Train/evaluate baseline models:

```bash
python train_models.py --help
```

Generate / simplify prompted rules:

```bash
python prompt_simplifications.py --help
```

(Exact flags depend on your dataset setup; see the scripts and `Utils/load_data.py`.)

## Results

- Summary CSV: `results/results.csv`
- Analysis notebook: `results/global_results.ipynb`

## Notes

This is research code (in-progress). If you want this to be fully reproducible, the next step is to document:
- which datasets are expected under `data/`
- how prototypes are selected
- what the exact prompt formats are
- what evaluation split is used
