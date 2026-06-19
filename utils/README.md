# 恢复工具箱（旧版）/ Restoration Toolbox (old)

本目录包含从 [4KAgent](https://github.com/taco-group/4KAgent) 复制的天气恢复工具箱代码（旧版），不上传 GitHub。

---

## 内容

| 子目录 | 说明 |
|---|---|
| `deraining/` | 去雨工具箱（MPRNet, MAXIM, X-Restormer, Restormer, Diff-Plugin 等） |
| `dehazing/` | 去雾工具箱（DehazeFormer, MAXIM, RIDCP, X-Restormer, Diff-Plugin 等） |
| `denoising/` | 去噪工具箱（MPRNet, MAXIM, X-Restormer, NAFNet, SwinIR 等） |
| `desnowing/` | 去雪工具箱（Star-Net, Diff-Plugin, DesnowNet, DDMSNet 等） |

---

## 获取方式

当前项目实际使用 `utils_new/`。此目录为旧版备份。

完整工具箱代码来自 4KAgent：

```bash
git clone https://github.com/taco-group/4KAgent.git
# 复制 utils/ 目录到此处
```

> 论文: [4KAgent: Agentic Any Image to 4K Super-Resolution](https://arxiv.org/abs/2507.07105)
