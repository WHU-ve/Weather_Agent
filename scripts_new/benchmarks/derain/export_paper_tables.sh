#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"

MAIN_ENV="${WEATHER_AGENT_ENV:-weather_agent}"
SUMMARY_JSON="${1:-outputs_derain_benchmark_full/overall_summary.json}"
METHOD_NAME="${2:-WeatherAgent (Ours)}"
OUTPUT_DIR="${3:-}"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(dirname "$SUMMARY_JSON")"
fi

conda run -n "$MAIN_ENV" python scripts/benchmarks/derain/export_paper_tables.py \
  --summary_json "$SUMMARY_JSON" \
  --method_name "$METHOD_NAME" \
  --output_dir "$OUTPUT_DIR"

echo "[Done] paper tables exported to: $OUTPUT_DIR"
