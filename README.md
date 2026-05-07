# CS639 Hallucination Detection Project

This repo contains the code we used for our CS639 final project on hallucination detection signals for language model outputs. The main experiment uses HALoGEN prompts and computes INSIDE/EigenScore-style hidden-state scores along with uncertainty and consistency baselines.

The code is organized around one main runner, two helper scripts for checking and analyzing results, and four experiment scripts.

## Files

| File or folder | What it is for |
| --- | --- |
| `halogen_inside_implementation.py` | Main HALoGEN experiment runner. |
| `validate_experiments.py` | Checks that saved `.pkl` result files have the expected fields, generation counts, and valid metrics. |
| `analyze_results.py` | Converts `.pkl` result files into CSV tables and plots. |
| `scripts/` | Bash scripts for the experiments used in the report/presentation. |
| `eda_halogen.py` | Basic HALoGEN dataset exploration. |
| `requirements.txt` | Python dependencies. |
| `Papers/` | Paper PDFs used for background reading. |

Run outputs are written to `halogen_outputs/`, `analysis_outputs/`, or `eda_results/`. These folders are ignored by git.

## Setup

Use Python 3.8 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, run the `.sh` experiment scripts from Git Bash or WSL.

The scripts load HALoGEN from Hugging Face:

```text
lasha-nlp/HALoGEN-prompts
```

If needed, the main runner can also load a local CSV, JSON, or JSONL file with `--halogen_source`.

## Small Test Commands

Before running a full experiment, run a small test.

OPT on CPU:

```bash
python halogen_inside_implementation.py \
  --model facebook/opt-125m \
  --limit 5 \
  --device cpu \
  --num_generations_per_prompt 3
```

TinyLlama on CUDA:

```bash
python halogen_inside_implementation.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --limit 5 \
  --device cuda:0 \
  --num_generations_per_prompt 3 \
  --trust_remote_code
```

Then validate the output file:

```bash
python validate_experiments.py --input <path-to-pkl> --expected_k 3
```

## Experiments

Run all commands from the repo root.

| Script | Purpose | Model | Size |
| --- | --- | --- | --- |
| `scripts/experiment_1_opt_categories.sh` | Compare HALoGEN categories and feature clipping. | `facebook/opt-125m` | 100 prompts per run, K=10 |
| `scripts/experiment_1_tinyllama_extensions.sh` | Extend Experiment 1 to TinyLlama across biographies, code, and references with feature clipping on/off. | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 100 prompts per category/run, K=10 |
| `scripts/experiment_2_opt_prompt_length.sh` | Compare short and long prompts. | `facebook/opt-125m` | 100 prompts per run, K=10 |
| `scripts/experiment_3_tinyllama_subset.sh` | Run TinyLlama on a small HALoGEN subset. | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 5 prompts, K=3 |
| `scripts/experiment_4_llama_full_halogen.sh` | Run TinyLlama on the full HALoGEN split. | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Full split, K=10 |


The scripts use:

```text
temperature = 0.5
top_p = 0.99
top_k = 10
max_new_tokens = 64
```

Run the OPT experiments:

```bash
bash scripts/experiment_1_opt_categories.sh
bash scripts/experiment_2_opt_prompt_length.sh
```

Run the TinyLlama subset experiment:

```bash
bash scripts/experiment_3_tinyllama_subset.sh
```

Run TinyLlama on the full HALoGEN split:

```bash
bash scripts/experiment_4_llama_full_halogen.sh
```

Each script clears only its own previous output folder before running. For example, Experiment 2 writes to:

```text
halogen_outputs/experiment_2_opt_prompt_length/
analysis_outputs/experiment_2_opt_prompt_length/
```

Each run folder contains `0.pkl` and `0_args.json`.

## Validation

The experiment scripts call `validate_experiments.py` automatically, but it can also be run manually.

Single result file:

```bash
python validate_experiments.py \
  --input halogen_outputs/experiment_2_opt_prompt_length/short_0_100_words/0.pkl \
  --expected_k 10
```

Whole experiment folder:

```bash
python validate_experiments.py \
  --input halogen_outputs/experiment_2_opt_prompt_length \
  --expected_k 10
```

The validator checks:

- required result fields
- number of records
- categories found
- number of sampled generations per prompt
- empty or degenerate generations
- missing, NaN, or infinite metric values
- summary statistics for each metric

`PASS` means the result files look clean. `WARN` usually means the model produced some empty or weak sampled outputs. `FAIL` means a required field or file structure is broken.

## Analysis Outputs

The experiment scripts call `analyze_results.py` automatically. To rerun analysis manually:

```bash
python analyze_results.py \
  --input halogen_outputs/experiment_2_opt_prompt_length \
  --output_dir analysis_outputs/experiment_2_opt_prompt_length \
  --plots
```

The main output files are:

| Output file | Use |
| --- | --- |
| `results_flat.csv` | One row per prompt. Use this for detailed checks. |
| `run_summary.csv` | One row per experiment run. Use this for high-level comparison. |
| `category_summary.csv` | Category-level summary table. |
| `length_summary.csv` | Short-vs-long prompt summary table. |
| `feature_clip_summary.csv` | Comparison of feature clipping on/off when both are present. |
| `eigenscore_by_category.png` | EigenScore distributions by HALoGEN category. |
| `eigenscore_by_prompt_length.png` | EigenScore distributions for short and long prompts. |
| `metric_summary_by_run.png` | Bar plots for the main metrics by run. |
| `metric_summary_by_prompt_length.png` | Bar plots for the short-vs-long prompt experiment. |

For slides, the most useful files are usually `run_summary.csv`, `category_summary.csv`, `length_summary.csv`, `metric_summary_by_run.png`, and `metric_summary_by_prompt_length.png`.

## Metrics

The saved result records include:

- `prompt`
- `category`
- `most_likely_generation`
- `generations`
- `perplexity`
- `energy`
- `entropy`
- `lexical_similarity`
- `eigenIndicator`
- `eigenIndicatorOutput`

Metric notes:

- `eigenIndicator` is the hidden-state EigenScore-style signal. Values closer to zero mean more dispersion across sampled generations.
- `lexical_similarity` measures how similar the sampled generations are to each other.
- `perplexity`, `entropy`, and `energy` are confidence-style proxy metrics from model logits.
- These metrics are detection signals, not official HALoGEN hallucination labels.

Optional answer-similarity evaluation:

```bash
python analyze_results.py \
  --input <path-to-pkl-or-folder> \
  --output_dir analysis_outputs/eval \
  --evaluate_correctness
```

This writes `detection_summary.csv`. It uses answer containment and ROUGE-L style similarity as a rough correctness proxy. It is not the official HALoGEN judge.

## INSIDE/EigenScore Details

The main method follows the INSIDE/EigenScore idea:

1. Generate K sampled responses for each prompt.
2. Extract hidden states during generation.
3. Use a middle-layer last-token hidden state as the response embedding.
4. Compute a covariance / singular-value score over the K embeddings.

The code also includes optional test-time feature clipping:

```bash
--enable_feature_clipping \
--feature_clip_memory_size 3000 \
--feature_clip_percentile 0.2
```

Feature clipping builds a memory bank of prompt-token activations, uses the penultimate layer, and clips activations outside the `[0.2, 99.8]` percentile range during generation. In this project, the memory bank is built from the filtered experiment prompts rather than a separate external corpus.

## HALoGEN Filtering

Useful flags:

```bash
--category biographies
--limit 100
--shuffle
--min_prompt_words 101
--max_prompt_words 100
```

Examples:

```bash
python halogen_inside_implementation.py \
  --model facebook/opt-125m \
  --category biographies \
  --shuffle \
  --limit 100
```

```bash
python halogen_inside_implementation.py \
  --model facebook/opt-125m \
  --shuffle \
  --max_prompt_words 100 \
  --limit 100
```

## Model Notes

- `facebook/opt-125m` works for CPU tests and the OPT report experiments.
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` is used for LLaMA-style experiments.
- Larger models need more GPU memory.
- The runner uses float32 on CPU and float16 on CUDA.
- If the tokenizer has no pad token, the runner uses EOS as padding when possible.
- Use `--max_num_gen_once` to split K generations into smaller batches if memory is tight.

## EDA

```bash
python eda_halogen.py
```

This writes basic HALoGEN plots and a dataset summary to `eda_results/`.

## Team Contributions

Project work was split across paper review, implementation, experiment runs, analysis, slides, and report writing.

| Member | Contribution |
| --- | --- |
| Joshua Ho | Paper review, slides, Experiment 1 discussion |
| Swapnil Gore | Slides, Experiment 1 runs |
| Nithya Krishna | Slides, recap, Experiment 1 discussion |
| Aarav Agrawal | Method planning, Experiment 2 analysis |
| Parith Reddy | Experiment 2 runs, analysis, experiment scripts, presentation support |

## References

- INSIDE: LLMs Internal States Retain the Power of Hallucination Detection
- HALoGEN prompt benchmark
- Public EigenScore implementation: https://github.com/D2I-ai/eigenscore
