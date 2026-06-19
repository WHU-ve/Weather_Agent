# 数据集 / Datasets

本目录存放训练和评估用的图像数据集，**源码文件**（`synthesize.py`、`image_degradations.py`、`degradations.txt`）已上传 GitHub，
**图像数据子目录**需单独下载或生成。

---

## 源码文件（已上传 GitHub）

| 文件 | 说明 |
|---|---|
| `synthesize.py` | 合成恶劣天气图像数据 |
| `image_degradations.py` | 图像退化模拟工具 |
| `degradations.txt` | 退化类型列表 |

---

## 数据子目录（需单独获取，不传 GitHub）

### LQ/ — 低质量输入图像

存放待恢复的低质量天气图像。使用 `synthesize.py` 自动生成：

```bash
python dataset/synthesize.py --input <清晰图像目录> --output dataset/LQ
```

也可放入自己的测试图像，支持 `.png`、`.jpg` 格式。

### 标准数据集下载

| 数据集 | 任务 | 下载地址 |
|---|---|---|
| **Rain800** | 去雨 | [GitHub](https://github.com/hezhangsprinter/IDT-CFS-Rain) |
| **RESIDE** | 去雾 | [官网](https://sites.google.com/view/reside-dehaze-datasets) |
| **SIDD** | 去噪 | [官网](https://www.eecs.yorku.ca/~kamel/sidd/) |
| **Snow100K** | 去雪 | [GitHub](https://github.com/weitingchen83/DesnowNet-ECCV-2020) |
| **FoundIR-Weather** | 多天气 | 联系原作者获取 |
| **AllWeather** | 多天气 | [GitHub](https://github.com/Jimmy448/AllWeather) |

### 示例图像

目录中可能包含少量示例图像（如 `LQ/example.png`）用于快速测试。

---

## 数据目录结构（完整版）

```
dataset/
├── synthesize.py           # ✅ 已上传
├── image_degradations.py   # ✅ 已上传
├── degradations.txt        # ✅ 已上传
├── LQ/                     # ❌ 需自行放入/生成
├── rain/                   # ❌ 去雨数据集
├── haze/                   # ❌ 去雾数据集
├── snow/                   # ❌ 去雪数据集
├── multi/                  # ❌ 多天气数据集
└── FoundIR-Weather/        # ❌ FoundIR 数据集
```
