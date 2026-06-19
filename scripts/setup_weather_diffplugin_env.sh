#!/bin/bash
set -e

ENV_NAME=${1:-weather_agent_diffplugin}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[1/4] Create DiffPlugin env: $ENV_NAME"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists, skipping creation and installation."
    exit 0
fi

conda create -n "$ENV_NAME" python=3.10 -y || true

echo "[2/4] Install torch 2.1 CUDA stack for DiffPlugin"
conda run -n "$ENV_NAME" pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118

echo "[3/4] Install DiffPlugin requirements"
conda run -n "$ENV_NAME" pip install -r requirements/requirements_weather_agent_diffplugin.txt

echo "[4/4] Verify DiffPlugin critical imports"
conda run -n "$ENV_NAME" python - <<'PY'
import torch
import diffusers
import transformers
import huggingface_hub
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
print('huggingface_hub:', huggingface_hub.__version__)
print('DiffPlugin deps import OK')
PY

echo "Done"
echo "Export if needed: export WEATHER_AGENT_DIFFPLUGIN_ENV=$ENV_NAME"
