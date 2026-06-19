# 大语言/视觉模型 / LLM & VLM Models

本目录存放项目的 LLM/VLM 模型权重，总大小约 **40GB**，不上传 GitHub。

---

## 模型列表

### 1. Llama-3.2-11B-Vision-Instruct（感知模块）

- **用途**: 零-shot 多标签退化分析（perception_module.py）
- **大小**: ~22GB
- **默认路径**: `models/Llama-3.2-Vision-11B-Instruct`
- **环境变量**: `WEATHER_PERCEPTION_MODEL_DIR`

**下载（二选一）**:

```bash
# HuggingFace（需申请访问权限）
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir models/Llama-3.2-Vision-11B-Instruct

# ModelScope（国内推荐，无需申请）
python -c "
from modelscope import snapshot_download
snapshot_download('LLM-Research/Llama-3.2-11B-Vision-Instruct',
                  local_dir='models/Llama-3.2-Vision-11B-Instruct')
"
```

- HuggingFace: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- ModelScope: https://modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct

### 2. Qwen2.5-VL-7B-Instruct（任务规划器）

- **用途**: VLM 任务规划（vlm_planner.py）
- **大小**: ~16GB
- **默认路径**: `models/Qwen2.5-VL-7B-Instruct`

**下载（二选一）**:

```bash
# HuggingFace
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen2.5-VL-7B-Instruct

# ModelScope（国内推荐）
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',
                  local_dir='models/Qwen2.5-VL-7B-Instruct')
"
```

- HuggingFace: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- ModelScope: https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct

### 3. CLIP ViT-B/32（IQA 评估）

- **用途**: CLIP-IQA 质量评估指标
- **大小**: ~340MB
- **自动下载**: pyiqa 首次运行时自动从 HuggingFace 下载
- HuggingFace: https://huggingface.co/openai/clip-vit-base-patch32

---

## 下载后目录结构

```
models/
├── Llama-3.2-Vision-11B-Instruct/
│   ├── config.json
│   ├── tokenizer.json
│   └── model-*.safetensors
├── Qwen2.5-VL-7B-Instruct/
│   ├── config.json
│   ├── tokenizer_config.json
│   └── model-*.safetensors
└── clip/
    └── ViT-B-32.pt
```
