# 恢复工具箱（当前使用）/ Restoration Toolbox (active)

本目录包含从 [4KAgent](https://github.com/taco-group/4KAgent) 复制的天气恢复工具箱代码，不上传 GitHub。

项目通过环境变量 `WEATHER_UTILS_DIR` 指定此目录（默认 `utils_new`）。

---

## 内容

| 子目录 | 说明 |
|---|---|
| `deraining/` | 去雨工具箱 |
| `dehazing/` | 去雾工具箱 |
| `denoising/` | 去噪工具箱 |
| `desnowing/` | 去雪工具箱 |
| `multitask_tools.py` | 多任务工具统一入口 |
| `tool.py` | 单工具基类 |

---

## 实际使用的专家模型

各工具箱调用的专家详见 [pretrained_ckpts_new/README.md](../pretrained_ckpts_new/README.md)。

**去雨 (deraining)**: MPRNet, MAXIM, X-Restormer, Restormer, Diff-Plugin
**去雾 (dehazing)**: DehazeFormer, MAXIM, RIDCP, X-Restormer, Diff-Plugin
**去雪 (desnowing)**: Star-Net, Diff-Plugin, DesnowNet, DDMSNet
**去噪 (denoising)**: MPRNet, MAXIM, X-Restormer, Restormer, NAFNet, SwinIR

---

## 获取方式

从 4KAgent 仓库获取：

```bash
git clone https://github.com/taco-group/4KAgent.git
cp -r 4KAgent/utils utils_new/
```

> 论文: [4KAgent: Agentic Any Image to 4K Super-Resolution](https://arxiv.org/abs/2507.07105)
