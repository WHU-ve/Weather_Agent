#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_PATH="${1:-dataset/LQ/example.png}"
OUTPUT_DIR="${2:-outputs_oneclick}"
PROFILE_MODE="${3:-auto}"
PROFILE_MANUAL="${4:-}"

MAIN_ENV="${WEATHER_AGENT_ENV:-weather_agent}"
RIDCP_ENV="${WEATHER_AGENT_RIDCP_ENV:-weather_agent_ridcp}"
NAFNET_ENV="${WEATHER_AGENT_NAFNET_ENV:-weather_agent_nafnet}"

echo "[1/3] 准备三环境: $MAIN_ENV / $RIDCP_ENV / $NAFNET_ENV"
bash scripts/setup_weather_all_envs.sh "$MAIN_ENV" "$RIDCP_ENV" "$NAFNET_ENV"

resolve_profile() {
	local mode="$1"
	local manual="$2"
	local input_path="$3"

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
	selected=$(conda run -n "$MAIN_ENV" python - "$input_path" <<'PY' 2>/dev/null || true
from PIL import Image
import sys

path = sys.argv[1]
img = Image.open(path)
w, h = img.size
pix = w * h
if pix >= 2560 * 1440:
	print('fast')
elif pix <= 1280 * 720:
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

PROFILE="$(resolve_profile "$PROFILE_MODE" "$PROFILE_MANUAL" "$INPUT_PATH")" || exit 1
source scripts/apply_runtime_profile.sh "$PROFILE"

echo "[2/3] 运行主流程推理"
conda run -n "$MAIN_ENV" python main.py --input "$INPUT_PATH" --output "$OUTPUT_DIR"

echo "[3/3] 完成"
echo "输入: $INPUT_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "模式: $PROFILE_MODE"
echo "运行档位: $PROFILE"
