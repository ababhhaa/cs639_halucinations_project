#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-halogen_outputs/experiment_4_llama_full_halogen}"
ANALYSIS_OUT="${ANALYSIS_OUT:-analysis_outputs/experiment_4_llama_full_halogen}"

rm -rf "$OUTPUT_DIR" "$ANALYSIS_OUT"

python3 halogen_inside_implementation.py \
  --model "$MODEL" \
  --halogen_source lasha-nlp/HALoGEN-prompts \
  --split train \
  --device "$DEVICE" \
  --num_generations_per_prompt 10 \
  --max_num_gen_once 10 \
  --temperature 0.5 \
  --top_p 0.99 \
  --top_k 10 \
  --max_new_tokens 64 \
  --trust_remote_code \
  --output_file "$OUTPUT_DIR/0.pkl"

python3 validate_experiments.py --input "$OUTPUT_DIR" --expected_k 10
python3 analyze_results.py --input "$OUTPUT_DIR" --output_dir "$ANALYSIS_OUT" --plots
