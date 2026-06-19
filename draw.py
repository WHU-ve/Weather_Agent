import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_derain_alpha_curve(
    json_path="/root/project/huangchao/zhengyanggong/weather_agent/weather_agent/update_log/alpha_grid_weatherbench_test200.json",
    task_name="derain",
    save_path="/root/project/huangchao/zhengyanggong/weather_agent/output/derain_alpha_composite_curve.png",
    figsize=(10.5, 3.6),
):
    # 1) 读取数据
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = data["tasks"][task_name]["rows"]

    # 2) 提取并按 alpha 排序
    rows = sorted(rows, key=lambda r: float(r["alpha"]))
    alphas = np.array([float(r["alpha"]) for r in rows])
    composites = np.array([float(r["composite"]) for r in rows])

    # 3) 找最佳点
    best_idx = int(np.argmax(composites))
    best_alpha = alphas[best_idx]
    best_comp = composites[best_idx]

    # 4) 画图
    plt.figure(figsize=figsize, dpi=150)
    ax = plt.gca()

    # 背景（白色）
    ax.set_facecolor("white")

    # 每个点的竖虚线
    y_min = composites.min() - 0.2
    for x, y in zip(alphas, composites):
        plt.vlines(x, y_min, y, colors="#9e9e9e", linestyles="--", linewidth=1.2, alpha=0.7)

    # 折线和点
    plt.plot(alphas, composites, color="black", linewidth=2.0, marker="o", markersize=7)

    # 最佳 alpha 竖向高亮条（只到 best 点）
    if len(alphas) > 1:
        step = np.min(np.diff(alphas))
        line_w = step * 0.44
    else:
        line_w = 0.06
    plt.vlines(best_alpha, y_min, best_comp, colors="#123a5c", linewidth=line_w * 250, alpha=0.95, zorder=1)

    # 最佳点再强调一下
    plt.scatter([best_alpha], [best_comp], color="black", s=80, zorder=5)

    # 标注最佳值
    plt.text(
        best_alpha,
        best_comp + 0.05,
        f"{best_comp:.2f}\nα={best_alpha:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    # 坐标轴文字
    plt.ylabel("Composite", fontsize=14)
    plt.xlabel(r"$\alpha$", fontsize=14)

    # x 轴刻度显示为百分比（可改成小数）
    plt.xticks(alphas, [f"{int(a*100)}%" for a in alphas], fontsize=12)

    # y 轴范围留一点边距
    y_max = composites.max() + 0.25
    plt.ylim(y_min, y_max)

    # 去掉上右边框，让图更像论文风格
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # 5) 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {save_path}")
    print(f"Best alpha: {best_alpha:.1f}, best composite: {best_comp:.6f}")


if __name__ == "__main__":
    plot_derain_alpha_curve()