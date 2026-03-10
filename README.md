# CS639 Hallucinations Project

This repository collects our project code for studying hallucination-related signals in large language models, with a focus on ideas inspired by INSIDE / EigenScore and the HALoGEN prompt benchmark.

At a high level, the repo contains:

- A standalone script, `halogen_inside_implementation.py`, for running a portable HALoGEN-style generation pipeline and saving uncertainty-style metrics.
- A larger research pipeline under `pipeline/` that can run generation experiments across multiple QA datasets.
- A simple exploratory data analysis script for the HALoGEN dataset.
- Local copies of the project papers in `Papers/`.

This is research code rather than a polished package, so the README below is written to help you get oriented quickly and avoid the common setup pitfalls.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `halogen_inside_implementation.py` | Standalone HALoGEN + INSIDE-style generation script. |
| `pipeline/generate.py` | Main experiment runner for multiple datasets. |
| `eda_halogen.py` | Exploratory data analysis for the HALoGEN prompts dataset. |
| `dataeval/` | Dataset loaders and generation-stop settings for HALoGEN, CoQA, TriviaQA, NQ-Open, and SQuAD. |
| `func/metric.py` | Metric helpers used by the generation pipeline. |
| `models/` | Model and tokenizer loading helpers. |
| `utils/` | Seeding, JSON helpers, and simple parallel task utilities. |
| `data/` | Default storage area for datasets, model weights, logs, and outputs. |
| `Papers/` | PDFs for the papers used in the project background. |

## What This Repo Can Do

There are two main workflows:

### 1. Portable HALoGEN + INSIDE-style run

`halogen_inside_implementation.py` is the easiest entry point if you want a single-file experiment. It:

- Loads a language model from a Hugging Face model ID or a local path.
- Loads HALoGEN-style prompts from Hugging Face or a local CSV / JSON / JSONL file.
- Generates one greedy answer plus several sampled answers per prompt.
- Saves proxy metrics such as:
  - `perplexity`
  - `energy`
  - `entropy`
  - `lexical_similarity`
  - `eigenIndicator`
  - `eigenIndicatorOutput` if a sentence-transformer encoder is available

It writes the results as a pickle file plus a JSON copy of the run arguments.

### 2. Multi-dataset experiment pipeline

`pipeline/generate.py` is the original experiment runner. It supports:

- `halogen`
- `coqa`
- `triviaqa`
- `nq_open`
- `SQuAD`
- `TruthfulQA` only if the optional module exists locally

This pipeline caches generations under `data/output/`, writes log files, and uses helper modules in `dataeval/`, `models/`, and `func/`.

## Setup

### Recommended environment

Python 3.8+ is recommended.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install seaborn
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install seaborn
```

Why `seaborn` separately:

- `eda_halogen.py` imports `seaborn`
- `run.sh` installs it implicitly through a minimal EDA setup
- `requirements.txt` currently does not include it

## Data and Model Paths

The project uses `_settings.py` to define storage locations. By default:

- models live under `data/weights`
- local datasets live under `data/datasets`
- generated outputs live under `data/output`

You can override those locations with environment variables:

- `EIGENSCORE_DATA_ROOT`
- `EIGENSCORE_MODEL_PATH`
- `EIGENSCORE_DATA_FOLDER`
- `EIGENSCORE_GENERATION_FOLDER`

The HALoGEN dataset loader also supports:

- `HALOGEN_SOURCE`
- `HALOGEN_SPLIT`
- `HALOGEN_CATEGORY`

## Quick Start

### Run the standalone HALoGEN script

Use this when you want the cleanest path to a result:

```powershell
.\.venv\Scripts\python.exe .\halogen_inside_implementation.py `
  --model mistralai/Mistral-7B-v0.1 `
  --halogen_source lasha-nlp/HALoGEN-prompts `
  --split train `
  --category biography `
  --num_generations_per_prompt 10 `
  --max_new_tokens 64 `
  --device cuda:0
```

If you want to use a local file instead of Hugging Face:

```powershell
.\.venv\Scripts\python.exe .\halogen_inside_implementation.py `
  --model .\path\to\local-model `
  --halogen_source .\data\my_halogen_prompts.jsonl `
  --device cpu `
  --limit 25
```

What gets written:

- A pickle file in `halogen_outputs/<model>_halogen_<project_ind>/` unless you pass `--output_dir` or `--output_file`
- A matching JSON file containing the run arguments

Each saved record includes fields such as:

- `prompt`
- `question`
- `answer`
- `most_likely_generation`
- `generations`
- `perplexity`
- `energy`
- `entropy`
- `lexical_similarity`
- `eigenIndicator`
- `eigenIndicatorOutput`

### Run the main experiment pipeline

Run this from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pipeline.generate `
  --model mistralai/Mistral-7B-v0.1 `
  --dataset halogen `
  --device cuda:0 `
  --fraction_of_data_to_use 0.1 `
  --num_generations_per_prompt 10 `
  --temperature 0.5 `
  --top_p 0.99 `
  --top_k 10 `
  --project_ind 0
```

Notes:

- Use `python -m pipeline.generate`, not `python pipeline/generate.py`, so imports resolve from the repo root.
- This script expects the dependencies in `requirements.txt`, including optional research helpers like `ipdb`, `sentence-transformers`, `torchmetrics`, and `selfcheckgpt`.
- Results are written under `data/output/<model>_<dataset>_<project_ind>/`.
- Logs are written under `data/output/logs/`.

### Run HALoGEN EDA

```powershell
.\.venv\Scripts\python.exe .\eda_halogen.py
```

This creates an `eda_results/` folder with:

- `category_distribution.png`
- `prompt_lengths.png`
- `dataset_summary.csv`

### Run the convenience shell script

On Unix-like systems:

```bash
bash run.sh
```

This script is only a lightweight EDA helper. It does not install the full project stack for model generation experiments.

## Dataset Notes

### HALoGEN

- `dataeval/halogen.py` can load from the Hugging Face dataset `lasha-nlp/HALoGEN-prompts`
- It can also load from a local directory or local `.json`, `.jsonl`, `.csv`, or `.parquet` file if `HALOGEN_SOURCE` is set

### CoQA

- Expects a local `coqa-dev-v1.0.json` under the configured dataset folder
- Saves a processed dataset cache to disk for reuse

### TriviaQA and NQ-Open

- Loaded via Hugging Face `datasets`
- Require internet access unless already cached locally

### SQuAD

- The current `dataeval/SQuAD.py` loader contains a hard-coded local path and is not portable as-is
- Expect to edit that file before using SQuAD on a different machine

## Practical Caveats

This repo is useful, but it is not a fully productionized evaluation framework. A few things to keep in mind:

- Some scripts assume GPU usage by default. On a CPU-only machine, pass `--device cpu`.
- Large models will be slow or may not fit in memory without a GPU.
- The standalone HALoGEN script produces proxy uncertainty signals; it does not automatically convert them into a final binary hallucination label.
- Several dataset loaders and metrics rely on external downloads or locally prepared files.
- `run.sh` is narrower than the rest of the repo and should not be treated as the full setup path.

## Suggested First Run

If you just want to confirm the repo is wired correctly, do this:

1. Create and activate the virtual environment.
2. Install `requirements.txt` plus `seaborn`.
3. Run `eda_halogen.py`.
4. Run `halogen_inside_implementation.py` on a very small subset using `--limit 5`.
5. Move to `pipeline.generate` only after the standalone script is working.

## References

The `Papers/` directory includes the project reading material:

- `INSIDE.pdf`
- `HALoGEN_PAPER.pdf`
- `why an how LLMs Hallucinate.pdf`
- `why language models hallucinate.pdf`

The project was also informed by the public EigenScore implementation:

- https://github.com/D2I-ai/eigenscore
