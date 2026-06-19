#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_PATH="${1:-dataset/LQ/example.png}"
OUTPUT_DIR="${2:-outputs_run}"
PROFILE_MODE="${3:-auto}"
PROFILE_MANUAL="${4:-}"

MAIN_ENV="weather_agent"

# 检查环境是否存在
if ! conda info --envs | grep -q "^${MAIN_ENV}\s"; then
    echo "错误：未找到主环境 ${MAIN_ENV}。请先执行安装脚本创建环境。"
    exit 1
fi

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

echo "=========================================="
echo "启动自动气象恢复系统"
echo "输入图像: $INPUT_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "模式:     $PROFILE_MODE"
echo "运行档位: $PROFILE"
echo "主环境:   $MAIN_ENV"
echo "=========================================="

echo "[状态] 自动激活主环境并启动核心调度程序 (main.py)..."
echo "提示: main.py 中的调度器会在需要时自动静默调用 RIDCP 和 NAFNet 的专用环境"
echo ""

# 使用 conda run 来确保在主环境中执行 Python
conda run -n "$MAIN_ENV" python main.py --input "$INPUT_PATH" --output "$OUTPUT_DIR"

echo "=========================================="
echo "执行完成！结果已保存至: $OUTPUT_DIR"
