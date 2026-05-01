#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
ROOT_OUT="${ROOT_OUT:-halogen_outputs/experiment_1_opt_categories}"
ANALYSIS_OUT="${ANALYSIS_OUT:-analysis_outputs/experiment_1_opt_categories}"

rm -rf "$ROOT_OUT" "$ANALYSIS_OUT"

run_category() {
  local name="$1"
  shift
  local out_dir="$ROOT_OUT/$name"

  python halogen_inside_implementation.py \
    --model facebook/opt-125m \
    --halogen_source lasha-nlp/HALoGEN-prompts \
    --split train \
    --shuffle \
    --limit 100 \
    --device "$DEVICE" \
    --num_generations_per_prompt 10 \
    --max_num_gen_once 10 \
    --temperature 0.5 \
    --top_p 0.99 \
    --top_k 10 \
    --max_new_tokens 64 \
    --output_file "$out_dir/0.pkl" \
    "$@"

  python validate_experiments.py --input "$out_dir" --expected_k 10
}

run_category "biographies" --category biography --category biographies
run_category "biographies_feature_clipping" \
  --category biography \
  --category biographies \
  --enable_feature_clipping \
  --feature_clip_memory_size 3000 \
  --feature_clip_percentile 0.2
run_category "code" --category code
run_category "references" --category reference --category references --category "scientific references"

python analyze_results.py --input "$ROOT_OUT" --output_dir "$ANALYSIS_OUT" --plots
