#!/bin/bash
set -e

ENV_NAME=${1:-weather_agent_jstasr}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[1/4] Create JSTASR env: $ENV_NAME"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists, skipping creation and installation."
    exit 0
fi

conda create -n "$ENV_NAME" python=3.7.16 -y || true

echo "[2/4] Install JSTASR requirements"
conda run -n "$ENV_NAME" pip install -r requirements/requirements_weather_agent_jstasr.txt

echo "[3/4] Verify JSTASR critical imports"
conda run -n "$ENV_NAME" python - <<'PY'
import tensorflow as tf
import keras
import numpy
import scipy
import cv2
print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('opencv:', cv2.__version__)
print('JSTASR deps import OK')
PY

echo "[4/4] Done"
echo "Export if needed: export WEATHER_AGENT_JSTASR_ENV=$ENV_NAME"
