"""为论文检查生成可保存的 ROC、分数分布和成本图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from ..data.schema import read_samples
from .detection import select_test_ids


def plot_results(samples_csv: str, cps_csv: str, metrics_csv: str,
                 output_dir: str, seed: int = 20261004,
                 test_fraction: float = 0.3) -> list[str]:
    """共用评价测试集绘制校正前后 ROC、分数分布，并绘制有实测成本的方法耗时与查询数量。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        cps_csv（str）：CPS 汇总 CSV，含 sample_id、三类分数、cps_score、cps_cal_score、randomness_baseline、runtime_sec、query_count。
        metrics_csv（str）：与 seed/test_fraction 对应同一测试集的评价 CSV，含整体 AUROC、耗时与查询数。
        output_dir（str）：图像保存目录，metrics 与此处 seed/test_fraction 使用同一测试集。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。
        test_fraction（float）：独立 base_id 分入测试集的比例，在 (0,1) 内；各攻击划分后的训练/测试集均应包含两类标签。 默认值：0.3。

    输出：
        list[str]：PNG 路径列表，包含 roc.png/score_distribution.png；成本已测时添加 runtime.png/query_cost.png。创建目录，覆盖同名图片，180 dpi。
    """
    samples = read_samples(samples_csv)
    scores = pd.read_csv(cps_csv, keep_default_na=False)
    metrics = pd.read_csv(metrics_csv, keep_default_na=False)
    # 1. 对齐样本与分数，选取评价使用的测试集。
    joined = samples[["sample_id", "base_id", "attack", "label"]].merge(
        scores[["sample_id", "cps_score", "cps_cal_score"]],
        on="sample_id", how="inner")
    selected = select_test_ids(joined.base_id.unique(), seed, test_fraction)
    test = joined[joined.base_id.isin(selected)]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = []

    # 2. 绘制整体测试 ROC，图例显示对应 AUROC。
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, column in (("CPS", "cps_score"), ("CPS-calibrated", "cps_cal_score")):
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

    # 4. 从评价表绘制实测生成时间和查询量。
    totals = metrics[metrics.attack == "ALL"].copy()
    totals["runtime_sec_total_test"] = pd.to_numeric(totals.runtime_sec_total_test,
                                                      errors="coerce")
    totals = totals.dropna(subset=["runtime_sec_total_test"])
    if not totals.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(totals.method, pd.to_numeric(totals.runtime_sec_total_test))
        ax.set(ylabel="Total test runtime (s)", title="Measured runtime")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = target / "runtime.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))
    query_totals = metrics[metrics.attack == "ALL"].copy()
    query_totals["query_count_total_test"] = pd.to_numeric(
        query_totals.query_count_total_test, errors="coerce")
    query_totals = query_totals.dropna(subset=["query_count_total_test"])
    if not query_totals.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(query_totals.method, query_totals.query_count_total_test)
        ax.set(ylabel="Total test queries", title="Measured query cost")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = target / "query_cost.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))
    return files
