# Weather Restoration Agent

本科毕设项目：面向恶劣天气图像恢复的轻量化 Agent 系统。

> **感知模块采用 Llama-3.2-Vision 进行零-shot 多标签退化分析，并在 prompt 中注入先验计算好的 IQA 数值。**

## 模块

1. **Perception Module**: 基于 Llama-3.2-Vision 的零-shot 多标签退化分析器，输出 JSON。
2. **Task Planner**: 基于 Qwen 的规划器，根据感知结果生成恢复顺序。
3. **Restoration Agent**: 逐步执行恢复，使用多个专家并选择最优。
4. **Quality Evaluator**: 综合质量分数选择结果。

## 使用

1. 准备数据集：使用 dataset/synthesize.py 生成合成数据。
2. (感知模块采用 Llama-3.2-Vision 零-shot，无需训练)
3. 运行: python main.py --input image.jpg --output output_dir

## 独立项目多环境依赖（推荐）

本项目保持**独立仓库运行**，但内部采用 **3 个 conda 环境** 保证全部专家可用：

- `weather_agent`：通用主环境
	- 用于：`xrestormer`、`mprnet`、`swinir_15`、`swinir_50`、`dehazeformer`、`maxim`、主流程（CLIP/IQA）
- `weather_agent_ridcp`：RIDCP 专用环境
	- 用于：`ridcp`
- `weather_agent_nafnet`：NAFNet 专用环境
	- 用于：`nafnet`

一键安装全部环境：

```bash
bash scripts/setup_weather_all_envs.sh
```

这是**首次安装环境**时的一条命令。

也可以分别安装：

```bash
bash scripts/setup_weather_env.sh
bash scripts/setup_weather_ridcp_env.sh
bash scripts/setup_weather_nafnet_env.sh
```

如需自定义环境名：

```bash
export WEATHER_AGENT_ENV=my_weather_agent
export WEATHER_AGENT_RIDCP_ENV=my_weather_agent_ridcp
export WEATHER_AGENT_NAFNET_ENV=my_weather_agent_nafnet
```

一键准备环境并运行一次推理：

```bash
bash scripts/one_click_run.sh
```

可选参数：

```bash
bash scripts/one_click_run.sh dataset/LQ/example.png outputs_check

# 第三个参数可选运行档位: fast | balanced | quality
bash scripts/one_click_run.sh dataset/LQ/example.png outputs_check fast

```

## 运行档位（一键切换速度/质量）

项目支持三档运行预设，通过脚本自动注入环境变量：

- `fast`：优先速度，减少动态重规划步数与重复处理
- `balanced`：默认平衡档
- `quality`：优先质量，允许更多重规划步骤

运行模式支持：

- `auto`：自动选择档位（按图像/数据集分辨率估计）
- `manual`：手动指定 `fast|balanced|quality`

可用于单张与批量：

```bash
# 单张
bash scripts/run_project.sh dataset/LQ/example.png outputs_run auto
bash scripts/run_project.sh dataset/LQ/example.png outputs_run manual fast
bash scripts/run_project.sh dataset/LQ/example.png outputs_run manual balanced
bash scripts/run_project.sh dataset/LQ/example.png outputs_run manual quality

# 一键准备环境+运行
bash scripts/one_click_run.sh dataset/LQ/example.png outputs_check auto
bash scripts/one_click_run.sh dataset/LQ/example.png outputs_check manual fast

# 批量
bash scripts/run_batch_inference.sh dataset/LQ outputs_batch auto
bash scripts/run_batch_inference.sh dataset/LQ outputs_batch manual fast
bash scripts/run_batch_inference.sh dataset/LQ outputs_batch manual balanced
bash scripts/run_batch_inference.sh dataset/LQ outputs_batch manual quality
```

## 后续实验启动命令

如果环境已经装好，后续做验证/对比实验时，直接使用下面命令即可。

### 单张图片

```bash
conda run -n weather_agent python main.py --input dataset/LQ/example.png --output outputs_check
```

### 整个数据集批量运行

```bash
bash scripts/run_batch_inference.sh dataset/LQ outputs_batch
```

说明：
- 输入目录下的每张图都会生成一个独立子目录；
- 每张图最终结果保存在 `outputs_batch/样本名/final_output.png`；
- 脚本支持断点续跑：已存在 `final_output.png` 的样本会自动跳过；
- 运行日志保存在 `outputs_batch/run_batch.log`；
- 汇总结果保存在 `outputs_batch/run_batch_summary.txt`。

说明：
- 这样仍然是你的项目**独立运行**，只是你自己的调度器会按专家自动切换到你自己的专用环境。
- `ridcp` 需要可编译 DCN CUDA 扩展的环境；`nafnet` 需要自己那套 `basicsr`，因此拆环境是最稳方案。
- `ridcp` 环境会自动安装 `gcc/g++ 10`，避免系统 `g++ 12` 导致 CUDA 11.6 扩展编译失败。
- `ridcp` 与 `nafnet` 都按项目真实运行方式通过 `PYTHONPATH` 调用本地代码，不再依赖容易失败的 `setup.py develop`。

## 完整项目说明

- **感知模块**：Llama-3.2-Vision 零-shot 多标签退化分析，prompt 中包含先验 IQA 数值。
- **任务规划**：Qwen 负责规划，后续可根据你的新输入格式进一步调整。
- **恢复执行**：使用 4KAgent 的工具箱（derain, dehaze, denoise），每个步骤运行多个专家并选择最优。
- **质量评价**：Q = 0.4⋅CLIPIQA + 0.4⋅MUSIQ - 0.2⋅NIQE，使用 pyiqa 库实现。

## 数据集和模型

- **工具箱**：从 4KAgent 复制的 deraining, dehazing, denoising 工具。
- **预训练模型**：需要下载到 `pretrained_ckpts/`（见 4KAgent Installation.md）。
- **数据集**：使用 `dataset/synthesize.py` 生成合成数据，或下载 Rain800/RESIDE/SIDD。