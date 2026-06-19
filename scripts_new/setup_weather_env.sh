#!/bin/bash
set -e

ENV_NAME=${1:-weather_agent}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[1/7] Create single env: $ENV_NAME"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists, skipping creation and installation."
    exit 0
fi
conda create -n "$ENV_NAME" python=3.10 -y || true

echo "[2/7] Install project requirements"
conda run -n "$ENV_NAME" pip install --extra-index-url https://download.pytorch.org/whl/cu116 -r requirements/requirements_weather_agent.txt

echo "[3/7] Install basicsr and add torchvision compatibility shim"
conda run -n "$ENV_NAME" pip install --upgrade --force-reinstall basicsr==1.4.2
SITE_PACKAGES=$(conda run -n "$ENV_NAME" python -c "import site; print(site.getsitepackages()[0])" | tail -n 1)
cat > "$SITE_PACKAGES/torchvision/transforms/functional_tensor.py" <<'PY'
from torchvision.transforms.functional import rgb_to_grayscale

__all__ = ['rgb_to_grayscale']
PY

# Tool-specific requirements (common experts only)
echo "[4/7] Install tool-specific requirements"
conda run -n "$ENV_NAME" pip install -r utils/deraining/tools/X-Restormer/requirements.txt
conda run -n "$ENV_NAME" pip install -r utils/dehazing/tools/DehazeFormer/requirements.txt
conda run -n "$ENV_NAME" pip install -r utils/dehazing/tools/maxim/requirements.txt

# Setup MAXIM local package
# MAXIM
conda run -n "$ENV_NAME" pip install "$PROJECT_ROOT/utils/dehazing/tools/maxim"

echo "[5/7] Install remaining utility deps"
conda run -n "$ENV_NAME" pip install ftfy

echo "[6/7] Verify critical imports"
conda run -n "$ENV_NAME" python - <<'PY'
import torch
import torchvision
import basicsr
from torchvision.transforms.functional_tensor import rgb_to_grayscale
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('basicsr import OK')
PY

echo "[7/7] Done"
echo "Use this env with: conda activate $ENV_NAME"
echo "Tool runner uses WEATHER_AGENT_ENV=$ENV_NAME for common experts."
