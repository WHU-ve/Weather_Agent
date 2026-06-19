# 三任务数据集落盘清单（对齐 4KAgent 原论文思路）

## 1) 目标与范围

本清单只覆盖你的毕设三类任务：
- Deraining（去雨）
- Dehazing（去雾）
- Denoising（去噪）

对齐依据：
- 4KAgent 工具链中的三任务配置与工具目录
- 你当前项目中的可执行工具与配置

---

## 2) 统一落盘根目录（建议）

建议在项目根目录下新建：

```text
datasets/
```

后续三类任务都落在这个根目录下，便于训练/评测复现。

---

## 3) Deraining（去雨）

### 3.1 必备数据（建议先落这批）

- 训练集：Rain13K
- 测试集：Test100、Rain100H

### 3.2 推荐补全测试集（论文常见）

- Rain100L
- Test1200
- Test2800

### 3.3 目标目录结构

```text
datasets/
└── Deraining/
    ├── Rain13K/
    │   ├── input/
    │   └── target/
    └── test/
        ├── Test100/
        │   ├── input/
        │   └── target/
        ├── Rain100H/
        │   ├── input/
        │   └── target/
        ├── Rain100L/
        │   ├── input/
        │   └── target/
        ├── Test1200/
        │   ├── input/
        │   └── target/
        └── Test2800/
            ├── input/
            └── target/
```

### 3.4 来源（优先按工具原 README）

- MPRNet Deraining 数据说明：
  - train: https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe?usp=sharing
  - test: https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing

---

## 4) Dehazing（去雾）

### 4.1 必备数据（建议先落这批）

- 训练集：ITS
- 测试集：SOTS indoor（nyuhaze500）

### 4.2 推荐补全

- SOTS outdoor
- 真实场景集（如 RS-Haze）

### 4.3 目标目录结构（按 X-Restormer 配置对齐）

```text
datasets/
└── Dehaze/
    ├── ITS/
    │   ├── clear/
    │   └── hazy/
    └── SOTS/
        └── indoor/
            └── nyuhaze500/
                ├── gt/
                └── hazy/
```

### 4.4 其它工具常见结构（DehazeFormer）

DehazeFormer 仓库常见是另一套目录命名（RESIDE-IN/train/GT,hazy；test/...）。
如你要复现实验曲线，可额外保留：

```text
utils/dehazing/tools/DehazeFormer/data/RESIDE-IN/...
```

### 4.5 来源

- DehazeFormer README 给出的数据下载入口（GoogleDrive/Baidu）：
  https://drive.google.com/drive/folders/1Yy_GH6_bydYPU6_JJzFQwig4LTh86VI4?usp=sharing

---

## 5) Denoising（去噪）

### 5.1 必备数据（建议先落这批）

- SIDD train
- SIDD val
- SIDD benchmark test（MAT）

### 5.2 推荐补全

- DND test（用于跨数据集泛化验证）

### 5.3 目标目录结构

```text
datasets/
└── Denoising/
    ├── SIDD/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │       ├── ValidationNoisyBlocksSrgb.mat
    │       └── ValidationGtBlocksSrgb.mat
    └── DND/
        └── test/
            ├── info.mat
            └── images_srgb/
                ├── 0001.mat
                ├── 0002.mat
                └── ...
```

### 5.4 来源（按 MPRNet README）

- SIDD train: https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
- SIDD test benchmark: https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php
- SIDD val: https://drive.google.com/drive/folders/1S44fHXaVxAYW3KLNxK41NYCnyX9S79su?usp=sharing
- DND test: https://noise.visinf.tu-darmstadt.de/downloads/

---

## 6) 与你当前项目的对齐说明

1. 你当前推理主流程是单图输入，不强依赖完整训练集。
2. 但若你要“严格对齐原论文三任务实验流程”，必须按上面的 datasets 结构落盘。
3. 你目前已有 dataset/（合成退化脚本），这是便捷数据生成；论文对齐实验建议使用本清单中的真实/标准数据集。

---

## 7) 落盘完成自检清单（可直接勾选）

### Deraining
- [ ] datasets/Deraining/Rain13K/input 与 target 已就位
- [ ] datasets/Deraining/test/Test100/input 与 target 已就位
- [ ] datasets/Deraining/test/Rain100H/input 与 target 已就位
- [ ] （可选）Rain100L/Test1200/Test2800 已就位

### Dehazing
- [ ] datasets/Dehaze/ITS/clear 与 hazy 已就位
- [ ] datasets/Dehaze/SOTS/indoor/nyuhaze500/gt 与 hazy 已就位
- [ ] （可选）SOTS outdoor 或 RS-Haze 已就位

### Denoising
- [ ] datasets/Denoising/SIDD/train 已就位
- [ ] datasets/Denoising/SIDD/val 已就位
- [ ] datasets/Denoising/SIDD/test 两个 MAT 文件已就位
- [ ] （可选）datasets/Denoising/DND/test 已就位

---

## 8) 备注（避免后续踩坑）

- 文件名大小写要严格一致（如 Rain13K、SOTS、nyuhaze500、ValidationNoisyBlocksSrgb.mat）。
- 如果你后续需要，我可以再给你一个一键校验脚本：自动扫描上述目录并输出缺失项报告。
