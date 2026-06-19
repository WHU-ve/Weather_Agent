# 模型下载说明 / Model Downloads Guide

本项目依赖大量预训练模型权重，总大小约 **100GB+**，不可上传至 GitHub（单文件限制 100MB）。
以下列出所有需要的模型及其下载方式。

---

## 一、环境变量

项目通过以下环境变量定位模型目录，下载前可先确认：

```bash
WEATHER_CKPT_DIR        # 恢复专家模型目录，默认: pretrained_ckpts/
WEATHER_UTILS_DIR       # 工具箱代码目录，默认: utils/
WEATHER_PERCEPTION_MODEL_DIR  # 感知模型目录，默认: models/Llama-3.2-Vision-11B-Instruct
```

---

## 二、恢复专家模型（~37G）

所有去雨/去雾/去噪/去雪专家模型由 [4KAgent](https://github.com/taco-group/4KAgent) 项目统一打包发布在 HuggingFace。

**下载源**: `YSZuo/4KAgent-Toolbox-Pretrained-Models`
**HuggingFace URL**: https://huggingface.co/YSZuo/4KAgent-Toolbox-Pretrained-Models
**论文**: [4KAgent: Agentic Any Image to 4K Super-Resolution](https://arxiv.org/abs/2507.07105)

### 一键下载脚本

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载整个 tar.gz（约 37G）
huggingface-cli download YSZuo/4KAgent-Toolbox-Pretrained-Models \
  4KAgent_toolbox_pretrained_ckpts.tar.gz \
  --local-dir . --repo-type model

# 解压到 pretrained_ckpts/
mkdir -p pretrained_ckpts
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts

# 清理 tar.gz
rm 4KAgent_toolbox_pretrained_ckpts.tar.gz
```

### 按需解压（仅本项目用到的模型）

如果不想解压全部，可以只解压需要的子集：

```bash
mkdir -p pretrained_ckpts

# ===== 去雨 (Deraining) =====
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts X-Restormer/derain_155k.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Restormer/deraining.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MPRNet/model_deraining.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MAXIM/maxim_ckpt_Deraining_Rain13k_checkpoint.npz
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Diff-Plugin/derain

# ===== 去雾 (Dehazing) =====
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts X-Restormer/dehaze_300k.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MAXIM/maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts RIDCP_dehazing
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts DehazeFormer
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Diff-Plugin/dehaze

# ===== 去噪 (Denoising) =====
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts X-Restormer/denoise_300k.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Restormer/real_denoising.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MPRNet/model_denoising.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts MAXIM/maxim_ckpt_Denoising_SIDD_checkpoint.npz
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts NAFNet/NAFNet-SIDD-width64.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth

# ===== 去雪 (Desnowing) =====
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts Diff-Plugin/desnow
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts DesnowNet
```

### 工具箱中各模型详情

| 专家模型 | 任务 | 论文 / 仓库 |
|---|---|---|
| **X-Restormer** | 去雨/去雾/去噪 | [GitHub](https://github.com/taco-group/X-Restormer) |
| **Restormer** | 去雨/去噪 | [GitHub](https://github.com/swz30/Restormer) |
| **MPRNet** | 去雨/去噪 | [GitHub](https://github.com/swz30/MPRNet) |
| **MAXIM** | 去雨/去雾/去噪 | [GitHub](https://github.com/google-research/maxim) |
| **SwinIR** | 去噪 | [GitHub](https://github.com/JingyunLiang/SwinIR) |
| **NAFNet** | 去噪 | [GitHub](https://github.com/megvii-research/NAFNet) |
| **RIDCP** | 去雾 | [GitHub](https://github.com/RQ-Wu/RIDCP_dehazing) |
| **DehazeFormer** | 去雾 | [GitHub](https://github.com/IDKiro/DehazeFormer) |
| **Diff-Plugin** | 去雨/去雾/去雪 | [GitHub](https://github.com/taco-group/Diff-Plugin) |
| **DesnowNet** | 去雪 | [GitHub](https://github.com/taco-group/JSTASR-DesnowNet-ECCV-2020) |

---

## 三、大语言/视觉模型

### 3.1 Llama-3.2-11B-Vision-Instruct（感知模块）

- **用途**: 零-shot 多标签退化分析
- **大小**: ~22G
- **默认路径**: `models/Llama-3.2-Vision-11B-Instruct`

**下载方式（二选一）**:

```bash
# HuggingFace（需要申请访问权限）
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir models/Llama-3.2-Vision-11B-Instruct

# ModelScope（国内推荐，无需申请）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('LLM-Research/Llama-3.2-11B-Vision-Instruct',
                  local_dir='models/Llama-3.2-Vision-11B-Instruct')
"
```

- HuggingFace: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- ModelScope: https://modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct

### 3.2 Qwen2.5-VL-7B-Instruct（任务规划器）

- **用途**: VLM 任务规划
- **大小**: ~16G
- **默认路径**: `models/Qwen2.5-VL-7B-Instruct`

**下载方式（二选一）**:

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

### 3.3 CLIP ViT-L/14（Diff-Plugin 依赖）

- **用途**: Diff-Plugin 的 prompt selector / 特征提取
- **大小**: ~1.7G
- **被 Stable Diffusion 自动下载**

HuggingFace: https://huggingface.co/openai/clip-vit-large-patch14

---

## 四、Stable Diffusion v1.4（Diff-Plugin 依赖）

- **用途**: Diff-Plugin 的图像恢复基础生成模型
- **大小**: ~4G

```bash
# Diff-Plugin 首次运行时会自动从 HuggingFace 下载
# 也可手动下载：
huggingface-cli download CompVis/stable-diffusion-v1-4 \
  --local-dir pretrained_ckpts/Diff-Plugin/stable-diffusion-v1-4
```

- HuggingFace: https://huggingface.co/CompVis/stable-diffusion-v1-4

---

## 五、快速下载全部

项目提供了快速下载脚本（仅下载本项目实际使用的模型）：

```bash
bash scripts/download_models.py
```

---

## 六、下载后验证

下载完成后，目录结构应如下：

```
pretrained_ckpts/
├── X-Restormer/
│   ├── derain_155k.pth
│   ├── dehaze_300k.pth
│   └── denoise_300k.pth
├── Restormer/
│   ├── deraining.pth
│   └── real_denoising.pth
├── MPRNet/
│   ├── model_deraining.pth
│   └── model_denoising.pth
├── MAXIM/
│   ├── maxim_ckpt_Deraining_Rain13k_checkpoint.npz
│   ├── maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz
│   └── maxim_ckpt_Denoising_SIDD_checkpoint.npz
├── NAFNet/
│   └── NAFNet-SIDD-width64.pth
├── RIDCP_dehazing/
│   ├── pretrained_RIDCP.pth
│   └── weight_for_matching_dehazing_Flickr.pth
├── Diff-Plugin/
│   ├── derain/tpb.pt
│   ├── dehaze/tpb.pt
│   ├── desnow/tpb.pt
│   └── stable-diffusion-v1-4/
├── DehazeFormer/
│   └── dehazeformer-b.pth
├── DesnowNet/
│   └── model.pth
└── SwinIR/model_zoo/swinir/
    ├── 005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth
    └── 005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth

models/
├── Llama-3.2-Vision-11B-Instruct/
└── Qwen2.5-VL-7B-Instruct/
```
