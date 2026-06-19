# Expert Smoke Report

输入图片：`dataset/LQ/example.png`
测试说明：

- 历史单环境 smoke：`conda run -n weather_agent`
- 当前 Route 1 smoke：主流程 `weather_agent`，`ridcp` 自动分发到 `weather_agent_ridcp`，`nafnet` 自动分发到 `weather_agent_nafnet`

| 任务 | 专家 | 结果 | 说明 |
|---|---|---|---|
| 去雨 | `mprnet` | PASS | 可正常产出结果 |
| 去雨 | `maxim` | PASS | 可正常产出结果 |
| 去雨 | `xrestormer` | PASS | 可正常产出结果 |
| 去雾 | `xrestormer` | PASS | 可正常产出结果 |
| 去雾 | `ridcp` | PASS | Route 1 下已成功编译 DCN 扩展，并通过真实调度产出结果 |
| 去雾 | `dehazeformer` | PASS | 可正常产出结果 |
| 去雾 | `maxim` | PASS | 可正常产出结果 |
| 去噪 | `xrestormer` | PASS | 可正常产出结果 |
| 去噪 | `swinir_15` | PASS | 可正常产出结果 |
| 去噪 | `swinir_50` | PASS | 可正常产出结果 |
| 去噪 | `mprnet` | PASS | 可正常产出结果 |
| 去噪 | `maxim` | PASS | 可正常产出结果 |
| 去噪 | `nafnet` | PASS | Route 1 下使用独立环境与本地 `PYTHONPATH` 运行，已正常产出结果 |

## 结论

当前 Route 1（项目内多环境）下：

- 总计 13 个专家
- 通过 13 个
- 失败 0 个

## 历史失败根因（已修复）

### RIDCP
需要 DCN CUDA 扩展 `deform_conv_ext`。现已通过专用环境、`gcc/g++ 10`、补齐缺失 `dcn/src` 源码、重新编译扩展解决。

### NAFNet
单环境里 `basicsr` 版本与 NAFNet 期望接口不一致。现已通过 NAFNet 专用环境，并按本地代码 `PYTHONPATH` 方式运行解决。

## 当前结论

最稳妥方案是当前已经落地的 Route 1：**项目独立 + 项目内多环境自动分发**。

- 对使用者来说仍然只是在运行同一个项目；
- 对内部实现来说，`ridcp`、`nafnet` 使用各自兼容环境；
- 当前 13/13 专家均可用。

本轮已额外验证：

- `ridcp` 真实调度运行成功，输出生成于 `outputs_route1_verify/dehaze_ridcp/output/output.png`
- `nafnet` 真实调度运行成功，输出生成于 `outputs_route1_verify/denoise_nafnet/output/output.png`
