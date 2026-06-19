#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_DIR="${1:-dataset/LQ}"
OUTPUT_ROOT="${2:-outputs_batch}"
PROFILE_MODE="${3:-auto}"
PROFILE_MANUAL="${4:-}"
MAIN_ENV="${WEATHER_AGENT_ENV:-weather_agent}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR"
  exit 1
fi

resolve_profile_for_dir() {
  local mode="$1"
  local manual="$2"
  local input_dir="$3"

  if [[ "$mode" == "manual" ]]; then
    if [[ "$manual" != "fast" && "$manual" != "balanced" && "$manual" != "quality" ]]; then
      echo "错误：manual 模式下需要指定档位 fast|balanced|quality" >&2
      return 1
    fi
    echo "$manual"
    return
  fi

  if [[ "$mode" != "auto" ]]; then
    echo "错误：第三个参数必须是 auto 或 manual" >&2
    return 1
  fi

  local selected
  selected=$(conda run -n "$MAIN_ENV" python - "$input_dir" <<'PY' 2>/dev/null || true
from PIL import Image
from pathlib import Path
import sys

folder = Path(sys.argv[1])
files = []
for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
    files.extend(folder.glob(ext))
files = sorted(files)[:20]
if not files:
    print('balanced')
    raise SystemExit

pixels = []
for p in files:
    try:
        with Image.open(p) as im:
            w, h = im.size
            pixels.append(w * h)
    except Exception:
        continue

if not pixels:
    print('balanced')
    raise SystemExit

avg_pix = sum(pixels) / len(pixels)
if avg_pix >= 2560 * 1440:
    print('fast')
elif avg_pix <= 1280 * 720:
    print('quality')
else:
    print('balanced')
PY
)
  if [[ "$selected" != "fast" && "$selected" != "balanced" && "$selected" != "quality" ]]; then
    selected="balanced"
  fi
  echo "$selected"
}

PROFILE="$(resolve_profile_for_dir "$PROFILE_MODE" "$PROFILE_MANUAL" "$INPUT_DIR")" || exit 1
source scripts/apply_runtime_profile.sh "$PROFILE"

mkdir -p "$OUTPUT_ROOT"
LOG_FILE="$OUTPUT_ROOT/run_batch.log"
SUMMARY_FILE="$OUTPUT_ROOT/run_batch_summary.txt"

echo "Batch inference started at $(date)" | tee "$LOG_FILE"
echo "Input directory: $INPUT_DIR" | tee -a "$LOG_FILE"
echo "Output root: $OUTPUT_ROOT" | tee -a "$LOG_FILE"
echo "Mode: $PROFILE_MODE" | tee -a "$LOG_FILE"
echo "Profile: $PROFILE" | tee -a "$LOG_FILE"
echo "Main env: $MAIN_ENV" | tee -a "$LOG_FILE"

total=0
success=0
failed=0

while IFS= read -r -d '' image_path; do
  total=$((total + 1))
  image_name="$(basename "$image_path")"
  stem="${image_name%.*}"
  sample_output_dir="$OUTPUT_ROOT/$stem"
  final_output="$sample_output_dir/final_output.png"

  if [[ -f "$final_output" ]]; then
    echo "[SKIP] $image_name -> $final_output" | tee -a "$LOG_FILE"
    success=$((success + 1))
    continue
  fi

  mkdir -p "$sample_output_dir"
  echo "[RUN ] $image_name" | tee -a "$LOG_FILE"

  if conda run -n "$MAIN_ENV" python main.py --input "$image_path" --output "$sample_output_dir" >> "$LOG_FILE" 2>&1; then
    if [[ -f "$final_output" ]]; then
      echo "[PASS] $image_name -> $final_output" | tee -a "$LOG_FILE"
      success=$((success + 1))
    else
      echo "[FAIL] $image_name -> final_output.png not found" | tee -a "$LOG_FILE"
      failed=$((failed + 1))
    fi
  else
    echo "[FAIL] $image_name -> main.py returned non-zero" | tee -a "$LOG_FILE"
    failed=$((failed + 1))
  fi
done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.tiff' \) -print0 | sort -z)

{
  echo "Batch inference finished at $(date)"
  echo "Total: $total"
  echo "Success: $success"
  echo "Failed: $failed"
  echo "Log: $LOG_FILE"
} | tee "$SUMMARY_FILE"
