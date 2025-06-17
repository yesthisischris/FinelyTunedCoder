#!/usr/bin/env bash
# Usage: bash eval/run_lm_eval.sh merged/h3-v4

MODEL_PATH=$1
lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL_PATH" \
  --tasks "humaneval,mbpp" \
  --batch_size 16 \
  --output_path eval/results.json
