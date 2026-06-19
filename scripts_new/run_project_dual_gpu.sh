#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_PATH="${1:-dataset/LQ/example.png}"
OUTPUT_DIR="${2:-outputs_run_dual_gpu}"
PLANNER_MODE="${3:-clip_vlm_fallback}"
PROFILE_MODE="${4:-auto}"
PROFILE_MANUAL="${5:-}"
MAIN_ENV="${WEATHER_AGENT_ENV:-weather_agent}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. GPU check skipped."
else
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${gpu_count:-0}" -lt 2 ]]; then
    echo "Warning: fewer than 2 GPUs detected. Dual-GPU layout may not apply."
  fi
fi

# Perception model (Llama-3.2-Vision) pinned to GPU 0.
export WEATHER_PERCEPTION_DEVICE="${WEATHER_PERCEPTION_DEVICE:-cuda:0}"
export WEATHER_PERCEPTION_QUANTIZATION="${WEATHER_PERCEPTION_QUANTIZATION:-4bit}"
export WEATHER_PERCEPTION_MODEL_SOURCE="${WEATHER_PERCEPTION_MODEL_SOURCE:-modelscope}"
export WEATHER_PERCEPTION_MODELSCOPE_ID="${WEATHER_PERCEPTION_MODELSCOPE_ID:-LLM-Research/Llama-3.2-11B-Vision-Instruct}"
export WEATHER_PERCEPTION_MODEL_ID="${WEATHER_PERCEPTION_MODEL_ID:-meta-llama/Llama-3.2-11B-Vision-Instruct}"
# Planning VLM pinned to GPU 1.
export WEATHER_VLM_INPUT_DEVICE="${WEATHER_VLM_INPUT_DEVICE:-cuda:1}"
export WEATHER_VLM_MODEL_ID="${WEATHER_VLM_MODEL_ID:-Qwen/Qwen2.5-VL-14B-Instruct}"
# Disable automatic model sharding to keep planner on a single GPU.
export WEATHER_VLM_DEVICE_MAP="${WEATHER_VLM_DEVICE_MAP:-none}"

# Keep default behavior unchanged unless user explicitly enables denoise.
export ENABLE_DENOISE_STEP="${ENABLE_DENOISE_STEP:-0}"

source scripts/run_project.sh "$INPUT_PATH" "$OUTPUT_DIR" "$PROFILE_MODE" "$PROFILE_MANUAL"
