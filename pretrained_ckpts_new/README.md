# 预训练模型权重 / Pretrained Model Checkpoints

本目录存放恢复专家模型的预训练权重，总大小约 **37GB**，不上传 GitHub。

---

## 下载方式

所有模型由 [4KAgent](https://github.com/taco-group/4KAgent) 项目统一打包，发布在 HuggingFace：

> **https://huggingface.co/YSZuo/4KAgent-Toolbox-Pretrained-Models**

```bash
pip install huggingface_hub
huggingface-cli download YSZuo/4KAgent-Toolbox-Pretrained-Models \
  4KAgent_toolbox_pretrained_ckpts.tar.gz --local-dir . --repo-type model
mkdir -p pretrained_ckpts
tar -xzf 4KAgent_toolbox_pretrained_ckpts.tar.gz -C pretrained_ckpts
rm 4KAgent_toolbox_pretrained_ckpts.tar.gz
```

> 详细说明见根目录的 [MODEL_DOWNLOADS.md](../MODEL_DOWNLOADS.md)

---

## 本项目实际使用的专家模型

### 🌧️ 去雨 (Deraining) — 5 个专家

| 专家 | 模型文件 | 论文/仓库 |
|---|---|---|
| **MPRNet** | `MPRNet/model_deraining.pth` | [GitHub](https://github.com/swz30/MPRNet) |
| **MAXIM** | `MAXIM/maxim_ckpt_Deraining_Rain13k_checkpoint.npz` | [GitHub](https://github.com/google-research/maxim) |
| **X-Restormer** | `X-Restormer/derain_155k.pth` | [GitHub](https://github.com/taco-group/X-Restormer) |
| **Restormer** | `Restormer/deraining.pth` | [GitHub](https://github.com/swz30/Restormer) |
| **Diff-Plugin** | `Diff-Plugin/derain/tpb.pt` + SD v1.4 | [GitHub](https://github.com/taco-group/Diff-Plugin) |

### 🌫️ 去雾 (Dehazing) — 5 个专家

| 专家 | 模型文件 | 论文/仓库 |
|---|---|---|
| **DehazeFormer** | `DehazeFormer/outdoor/dehazeformer-b.pth` | [GitHub](https://github.com/IDKiro/DehazeFormer) |
| **MAXIM** | `MAXIM/maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz` | [GitHub](https://github.com/google-research/maxim) |
| **RIDCP** | `RIDCP_dehazing/pretrained_RIDCP.pth` + `weight_for_matching_dehazing_Flickr.pth` | [GitHub](https://github.com/RQ-Wu/RIDCP_dehazing) |
| **X-Restormer** | `X-Restormer/dehaze_300k.pth` | [GitHub](https://github.com/taco-group/X-Restormer) |
| **Diff-Plugin** | `Diff-Plugin/dehaze/tpb.pt` | [GitHub](https://github.com/taco-group/Diff-Plugin) |

### ❄️ 去雪 (Desnowing) — 4 个专家

| 专家 | 模型文件 | 论文/仓库 |
|---|---|---|
| **Star-Net** | `StarNet/model.ckpt.*` | [论文](https://arxiv.org/abs/2507.07105) |
| **Diff-Plugin** | `Diff-Plugin/desnow/tpb.pt` | [GitHub](https://github.com/taco-group/Diff-Plugin) |
| **DesnowNet** | `DesnowNet/model.pth` | [GitHub](https://github.com/taco-group/JSTASR-DesnowNet-ECCV-2020) |
| **DDMSNet** | `DDMSNet/snow100k_DDMSNet/` + `cityscapes_best.pth` | [论文](https://arxiv.org/abs/2507.07105) |

### 🔊 去噪 (Denoising) — 6 个专家

| 专家 | 模型文件 | 论文/仓库 |
|---|---|---|
| **MPRNet** | `MPRNet/model_denoising.pth` | [GitHub](https://github.com/swz30/MPRNet) |
| **MAXIM** | `MAXIM/maxim_ckpt_Denoising_SIDD_checkpoint.npz` | [GitHub](https://github.com/google-research/maxim) |
| **X-Restormer** | `X-Restormer/denoise_300k.pth` | [GitHub](https://github.com/taco-group/X-Restormer) |
| **Restormer** | `Restormer/real_denoising.pth` | [GitHub](https://github.com/swz30/Restormer) |
| **NAFNet** | `NAFNet/NAFNet-SIDD-width64.pth` | [GitHub](https://github.com/megvii-research/NAFNet) |
| **SwinIR** | `SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth` + `noise50.pth` | [GitHub](https://github.com/JingyunLiang/SwinIR) |

---

## Diff-Plugin 额外依赖

Diff-Plugin 所有子任务（derain / dehaze / desnow）还依赖以下大模型，首次运行时会自动从 HuggingFace 下载：

| 模型 | HuggingFace ID | 大小 |
|---|---|---|
| Stable Diffusion v1.4 | `CompVis/stable-diffusion-v1-4` | ~4GB |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | ~1.7GB |

---

## 4KAgent 引用

```
@article{zuo20254kagent,
  title={4KAgent: Agentic Any Image to 4K Super-Resolution},
  author={Yushen Zuo and Qi Zheng and Mingyang Wu and Xinrui Jiang and Renjie Li and
          Jian Wang and Yide Zhang and Gengchen Mai and Lihong V. Wang and James Zou and
          Xiaoyu Wang and Ming-Hsuan Yang and Zhengzhong Tu},
  year={2025},
  eprint={2507.07105},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.07105},
}
```
