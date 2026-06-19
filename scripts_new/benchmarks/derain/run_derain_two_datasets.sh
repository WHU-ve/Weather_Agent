#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"

MAIN_ENV="${WEATHER_AGENT_ENV:-weather_agent}"
PROFILE="${1:-quality}"
OUTPUT_ROOT="${2:-outputs_derain_benchmark}"
LIMIT="${3:-0}"
PARALLEL_WORKERS="${4:-1}"
PARALLEL_GPU_IDS="${5:-}"

if [[ "$PROFILE" != "fast" && "$PROFILE" != "balanced" && "$PROFILE" != "quality" ]]; then
  echo "PROFILE must be one of: fast | balanced | quality"
  exit 1
fi

echo "[Derain Benchmark] env=$MAIN_ENV profile=$PROFILE output=$OUTPUT_ROOT limit=$LIMIT workers=$PARALLEL_WORKERS gpu_ids=${PARALLEL_GPU_IDS:-none}"

conda run -n "$MAIN_ENV" python scripts/benchmarks/derain/eval_derain_two_datasets.py \
  --dataset_root dataset/rain \
  --datasets Rain100H rain100H_train \
  --profile "$PROFILE" \
  --output_root "$OUTPUT_ROOT" \
  --limit "$LIMIT" \
  --parallel_workers "$PARALLEL_WORKERS" \
  --parallel_gpu_ids "$PARALLEL_GPU_IDS"

echo "[Done] results at: $OUTPUT_ROOT"
