#!/bin/bash
set -e

ENV_NAME=${1:-weather_agent_nafnet}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[1/6] Create NAFNet env: $ENV_NAME"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists, skipping creation and installation."
    exit 0
fi

conda create -n "$ENV_NAME" python=3.10 -y || true

echo "[1.5/6] Install CUDA-compatible torch stack"
conda run -n "$ENV_NAME" pip install --extra-index-url https://download.pytorch.org/whl/cu116 \
    torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116

echo "[2/6] Install project base requirements needed by NAFNet"
conda run -n "$ENV_NAME" pip install --extra-index-url https://download.pytorch.org/whl/cu116 -r requirements/requirements_weather_agent_nafnet.txt

echo "[3/6] Install NAFNet requirements"
conda run -n "$ENV_NAME" pip install -r utils/denoising/tools/NAFNet/requirements.txt

echo "[4/6] Use local NAFNet code via PYTHONPATH (no package install needed)"

echo "[5/6] Verify NAFNet imports"
conda run -n "$ENV_NAME" env PYTHONPATH="$PROJECT_ROOT/utils/denoising/tools/NAFNet" python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
from basicsr.data import create_dataloader, create_dataset
print('NAFNet basicsr import OK')
PY

echo "[6/6] Done"
echo "Export if needed: export WEATHER_AGENT_NAFNET_ENV=$ENV_NAME"
