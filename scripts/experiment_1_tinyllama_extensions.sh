#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
ROOT_OUT="${ROOT_OUT:-halogen_outputs/experiment_1_tinyllama_extensions}"
ANALYSIS_OUT="${ANALYSIS_OUT:-analysis_outputs/experiment_1_tinyllama_extensions}"

rm -rf "$ROOT_OUT" "$ANALYSIS_OUT"

run_category() {
  local name="$1"
  shift
  local out_dir="$ROOT_OUT/$name"

  mkdir -p "$out_dir"

  python halogen_inside_implementation.py \
    --model "$MODEL" \
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
    --trust_remote_code \
    --output_file "$out_dir/0.pkl" \
    "$@"

  python validate_experiments.py --input "$out_dir" --expected_k 10
}

run_category "biographies" \
  --category biographies

run_category "biographies_feature_clipping" \
  --category biographies \
  --enable_feature_clipping \
  --feature_clip_memory_size 3000 \
  --feature_clip_percentile 0.2

run_category "code" \
  --category code

run_category "code_feature_clipping" \
  --category code \
  --enable_feature_clipping \
  --feature_clip_memory_size 3000 \
  --feature_clip_percentile 0.2

run_category "references" \
  --category references

run_category "references_feature_clipping" \
  --category references \
  --enable_feature_clipping \
  --feature_clip_memory_size 3000 \
  --feature_clip_percentile 0.2

python analyze_results.py \
  --input "$ROOT_OUT" \
  --output_dir "$ANALYSIS_OUT" \
  --plots \
  --evaluate_correctness
