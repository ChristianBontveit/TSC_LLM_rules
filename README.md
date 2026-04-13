# TSC_LLM_rules

Research code for **rule-based interpretability of Time Series Classification (TSC)** with Large Language Models (LLMs).

## Idea (high level)

Use an LLM as a **rule synthesizer** to produce simple, human-readable decision rules for time series classification.

Typical setup:
- select a small set of **prototypes/examples** per class
- prompt an LLM to produce either:
  - a natural-language rule
- evaluate that rule on held-out samples

Metrics you care about:
- accuracy
- coverage
- complexity

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

## Data layout

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

## LLM Rules for TSC

`llm_rules_tsc.py` prompts an LLM to create subrules for each class using prototype plots and evaluates them on sampled test instances.

It expects an `.env` file in repo root with at least:
- `API_KEY`

Run:

Current behavior in the main loop:
- the script runs `10` experiment repetitions
- each repetition samples `100` test instances from the full normalized test set
- in `rulebased` and `noPrototype` modes, a fresh rule set is generated on each repetition
- classification is done in batches of `10` instances per LLM call

Variables which affect the run size are in the main loop:
- `for n in range(10)` controls how many repetitions are executed
- `rand_ts_idx = np.random.randint(0, full_test_ts_norm.shape[0], size=(100))` controls how many test instances are classified per repetition
- `batch_size=10` controls how many instances are sent in each classification prompt

```bash
python llm_rules_tsc.py --dataset Chinatown --classifier miniRocket --llm gpt-5.1 --k 3 --rules 2
```

Options:
- `--mode` chooses between `baseline`, `rulebased` and `noPrototype`.
- `--mode` default is `rulebased`.
- `--k` is the number of prototypes per class.
- `--rules` is the maximum number of sub-rules per class requested from the LLM.

Results are saved in:
- `results/llm_results/<dataset>_<mode>_<k>_<rules>_llm_results.jsonl`

Notes on saved results:
- the file is opened in append mode, so previous results are not deleted
- each line in the `.jsonl` file represents one repetition of the experiment
- with the current loop, one command execution appends `10` lines to the matching results file

Reading the results from the jsonl
- each line represents one complete run over `100` sampled test instances
- It is represented as such:
  - `{"dataset": "Chinatown", "mode": "rulebased", "classifier": "miniRocket", "llm": "gpt-5.1", "k": 3, "num_rules": 3, "accuracy": 0.6, "extracted_rules": {...}, "instance": [{"instance_id": 1, "ts_idx": 93, "true_label": 1, "predicted_label": 1, "status": "MATCH"}]}`
