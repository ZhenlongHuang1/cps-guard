"""生成最小 Pilot 的 ROC 和分数分布图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from .detection import select_test_ids


def plot_results(cps_csv: str, random_csv: str, metrics_csv: str,
                 output_dir: str, seed: int = 20261004) -> list[str]:
    """使用与评价相同的测试划分，绘制三方法 ROC 和校正前后分数分布。

    实验方案对应：
        S12，第十九节 Day 6 的 ROC/分数分布，第三节第 4 项校正前后对比。

    算法/公式：
        ROC 绘制 (FPR(t),TPR(t))；直方图比较 clean/poison 的 CPS 与 CPS_cal。
        按 base_id 留出 30% 问题，与 evaluate_scores 使用同一种子和划分规则。

    输入：
        cps_csv（str）：含 sample_id、base_id、label、cps_score、cps_cal_score 的汇总。
        random_csv（str）：同样样本的 Random 分数表，含 sample_id 和 score。
        metrics_csv（str）：同次评价的指标表，图例读取三个方法的 ALL AUROC。
        output_dir（str）：PNG 保存目录，自动创建并覆盖同名图片。
        seed（int）：与评价一致的分组种子，默认 20261004。

    输出：
        list[str]：roc.png、score_distribution.png 的路径，180 dpi。
        只绘图，不重新计分或选择检测阈值。
    """
    scores = pd.read_csv(cps_csv)
    baseline = pd.read_csv(random_csv)[["sample_id", "score"]]
    metrics = pd.read_csv(metrics_csv)
    # 1. 对齐分数，选择评价使用的测试问题。
    joined = scores.merge(baseline.rename(columns={"score": "random_score"}), on="sample_id")
    selected = select_test_ids(joined.base_id, seed, 0.3)
    test = joined[joined.base_id.isin(selected)]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = []

    # 2. 绘制整体测试 ROC，图例显示对应 AUROC。
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, column in (("CPS", "cps_score"), ("CPS-calibrated", "cps_cal_score"),
                         ("Random", "random_score")):
        fpr, tpr, _ = roc_curve(test.label, test[column].astype(float))
        auc_row = metrics[(metrics.method == name) & (metrics.attack == "ALL")]
        label = f"{name} (AUROC={float(auc_row.iloc[0].AUROC):.3f})"
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC")
    ax.legend()
    fig.tight_layout()
    path = target / "roc.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    # 3. 对比 clean/poison 在校正前后的分数分布。
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, column) in zip(axes, (("CPS", "cps_score"),
                                         ("CPS-calibrated", "cps_cal_score"))):
        for label, title in ((0, "clean"), (1, "trigger-bearing")):
            values = test.loc[test.label == label, column].astype(float)
            ax.hist(values, bins=20, alpha=0.55, label=title, density=True)
        ax.set(xlabel="Detection score", ylabel="Density", title=name)
        ax.legend()
    fig.tight_layout()
    path = target / "score_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    return files
