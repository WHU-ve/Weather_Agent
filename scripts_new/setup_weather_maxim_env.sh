#!/bin/bash
set -e

ENV_NAME=${1:-weather_agent_maxim}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[1/4] Create MAXIM env: $ENV_NAME"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists, skipping creation and installation."
    exit 0
fi

conda create -n "$ENV_NAME" python=3.10 -y || true

echo "[2/4] Install MAXIM requirements"
conda run -n "$ENV_NAME" pip install -r requirements/requirements_weather_agent_maxim.txt

echo "[3/4] Skip upstream unpinned requirements to keep MAXIM env stable"

echo "[4/4] Verify critical MAXIM imports"
conda run -n "$ENV_NAME" python - <<'PY'
import tensorflow as tf
import jax
import flax
import ml_collections
from PIL import Image
print('tensorflow:', tf.__version__)
print('jax:', jax.__version__)
print('flax:', flax.__version__)
print('MAXIM deps import OK')
PY

echo "Done"
echo "Export if needed: export WEATHER_AGENT_MAXIM_ENV=$ENV_NAME"
