"""为论文检查生成可保存的 ROC、分数分布和成本图。"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from .io import read_samples


def plot_results(samples_csv: str, cps_csv: str, metrics_csv: str,
                 output_dir: str, seed: int = 20261004,
                 test_fraction: float = 0.3) -> list[str]:
    """仅绘制与主评估相同的测试集，避免在图上混入阈值拟合样本。"""
    samples = read_samples(samples_csv)
    scores = pd.read_csv(cps_csv, keep_default_na=False)
    metrics = pd.read_csv(metrics_csv, keep_default_na=False)
    joined = samples[["sample_id", "base_id", "attack", "label"]].merge(
        scores[["sample_id", "cps_score", "cps_cal_score"]],
        on="sample_id", how="inner", validate="one_to_one")
    if len(joined) != len(samples):
        raise ValueError("绘图分数未覆盖全部样本")
    ids = sorted(joined.base_id.unique())
    random.Random(seed).shuffle(ids)
    selected = set(ids[:max(1, round(len(ids) * test_fraction))])
    test = joined[joined.base_id.isin(selected)]
    if test.label.nunique() != 2:
        raise ValueError("测试集缺少 clean 或 poison 类别")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = []

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, column in (("CPS", "cps_score"), ("CPS-calibrated", "cps_cal_score")):
        fpr, tpr, _ = roc_curve(test.label, test[column].astype(float))
        auc_row = metrics[(metrics.method == name) & (metrics.attack == "ALL")]
        label = f"{name} (AUROC={float(auc_row.iloc[0].AUROC):.3f})" if len(auc_row) == 1 else name
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC")
    ax.legend()
    fig.tight_layout()
    path = target / "roc.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

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
