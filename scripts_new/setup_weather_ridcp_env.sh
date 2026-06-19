#!/bin/bash
set -e

ENV_NAME=${1:-weather_agent_ridcp}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[1/6] Create RIDCP env: $ENV_NAME"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists, skipping creation and installation."
    exit 0
fi

conda create -n "$ENV_NAME" python=3.10 -y || true

echo "[1.5/6] Install CUDA-compatible compiler toolchain (gcc/g++ 10)"
conda install -n "$ENV_NAME" -y -c conda-forge gcc_linux-64=10 gxx_linux-64=10 ninja

echo "[2/6] Install project base requirements needed by RIDCP"
conda run -n "$ENV_NAME" pip install --extra-index-url https://download.pytorch.org/whl/cu116 -r requirements/requirements_weather_agent_ridcp.txt
conda run -n "$ENV_NAME" pip install --upgrade --force-reinstall "numpy<2"

echo "[3/6] Install RIDCP toolbox requirements"
conda run -n "$ENV_NAME" pip install -r utils/dehazing/tools/RIDCP_dehazing/requirements.txt
conda run -n "$ENV_NAME" pip install --upgrade --force-reinstall "numpy<2"

echo "[3.5/6] Populate missing RIDCP DCN source files"
conda run -n "$ENV_NAME" python - <<'PY'
from pathlib import Path
import shutil
import basicsr

project_root = Path.cwd()
ridcp_src = project_root / "utils/dehazing/tools/RIDCP_dehazing/basicsr/ops/dcn/src"
site_src = Path(basicsr.__file__).resolve().parent / "ops/dcn/src"

if not ridcp_src.exists():
  ridcp_src.parent.mkdir(parents=True, exist_ok=True)
  shutil.copytree(site_src, ridcp_src)
  print(f"Copied DCN src: {site_src} -> {ridcp_src}")
else:
  print(f"RIDCP DCN src already exists: {ridcp_src}")
PY

echo "[4/6] Build RIDCP with CUDA DCN extension"
(
  cd "$PROJECT_ROOT/utils/dehazing/tools/RIDCP_dehazing"
  conda run -n "$ENV_NAME" env BASICSR_EXT=True FORCE_CUDA=1 CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++ python setup.py build_ext --inplace
)

echo "[5/6] Verify critical RIDCP imports"
conda run -n "$ENV_NAME" env PYTHONPATH="$PROJECT_ROOT/utils/dehazing/tools/RIDCP_dehazing" python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
from basicsr.ops.dcn import deform_conv
print('RIDCP DCN import OK')
PY

echo "[6/6] Done"
echo "Export if needed: export WEATHER_AGENT_RIDCP_ENV=$ENV_NAME"
