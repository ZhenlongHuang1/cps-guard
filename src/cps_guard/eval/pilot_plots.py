"""Pilot-v2 固定 Test 预测及特征的科研图表。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from .transfer import VIEWS


def plot_pilot(output: Path) -> None:
    """对应 STEP 11–18：复用预测画 ROC、随机分布、消融区间和单特征分布。

    输入：v2 根目录，内含 Test 预测、bootstrap、Random 和特征表。
    输出：None；figures 下五份 PDF。ROC 按目标样本逐阈值计算；图表不选参，
        特征分布只展示预先指定的 CPS/entropy/hidden norm，不自动搜索最佳特征。
    """
    results, figures = output / "results", output / "figures"
    predictions = pd.read_csv(results / "test_predictions.csv")
    for cross, name in ((False, "roc_iid"), (True, "roc_cross_attack")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, source in zip(axes, ("badnet", "vpi")):
            target = ("vpi" if source == "badnet" else "badnet") if cross else source
            for view in VIEWS:
                group = predictions[(predictions.source_attack == source) &
                                    (predictions.target_attack == target) & (predictions.view == view)]
                fpr, tpr, _ = roc_curve(group.label, group.score)
                ax.plot(fpr, tpr, label=view)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{source} -> {target}")
            ax.legend(fontsize=8)
        fig.savefig(figures / f"{name}.pdf")
        plt.close(fig)
    random = pd.read_csv(results / "random_seed_results.csv")
    summary = pd.read_csv(results / "bootstrap_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, target in zip(axes, ("badnet", "vpi")):
        ax.hist(random[random.target_attack == target].AUROC, bins=12, alpha=0.7)
        full = summary[(summary.target_attack == target) & (summary.source_attack != target) & (summary.view == "B+G+R")]
        ax.axvline(full.AUROC.iloc[0], color="red", label="Cross-attack B+G+R")
        ax.set(xlabel="AUROC", ylabel="Number of seeds", title=target)
        ax.legend(fontsize=8)
    fig.savefig(figures / "random_distribution.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, source in zip(axes, ("badnet", "vpi")):
        group = summary[(summary.source_attack == source) & (summary.target_attack != source)].set_index("view").loc[list(VIEWS)]
        ax.bar(range(len(VIEWS)), group.AUROC, alpha=0.7)
        ax.vlines(range(len(VIEWS)), group.CI_low, group.CI_high, color="black")
        ax.set(xticks=range(len(VIEWS)), xticklabels=VIEWS, ylim=(0, 1), ylabel="AUROC (95% CI)", title=f"Source: {source}")
        ax.tick_params(axis="x", rotation=35)
    fig.savefig(figures / "ablation.pdf")
    plt.close(fig)
    frame = pd.read_csv(results / "features.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for row, attack in enumerate(("badnet", "vpi")):
        test = frame[(frame.attack == attack) & (frame.split == "test")]
        for ax, column in zip(axes[row], ("cps_score", "entropy_mean", "hidden_norm_mean")):
            for label, name in ((0, "clean"), (1, "poison")):
                ax.hist(test[test.label == label][column], bins=15, alpha=0.55, label=name)
            ax.set(title=f"{attack}: {column}", ylabel="Samples")
            ax.legend(fontsize=8)
    fig.savefig(figures / "feature_distribution.pdf")
    plt.close(fig)


def plot_control(output: Path) -> None:
    """补充 STEP 5：沿用原 ROC、七视图 AUROC 区间和分布图展示对照结果。

    实验方案对应：观察相同冻结检测器在指定模型的 clean/poison/neutral 上的表现。
    算法：ROC 按原连续概率绘制；AUROC 区间来自 evaluate_control 的原配对 bootstrap；
        分数分布按输入类型分组，并标出原来源阈值。图表不选择特征或阈值。
    输入：output 为补充测试目录，含 test_predictions/control_results/features.csv。
    输出：None；figures 下 roc_control.pdf、ablation.pdf、score_distribution.pdf、
        feature_distribution.pdf。没有中性样本时只展示 clean/poison。
    """
    predictions = pd.read_csv(output / "results/test_predictions.csv")
    metrics = pd.read_csv(output / "results/control_results.csv")
    negatives = [kind for kind in ("clean", "neutral") if kind in predictions.input_type.values]
    fig, axes = plt.subplots(1, len(negatives), figsize=(5 * len(negatives), 4),
                             squeeze=False, layout="constrained")
    for ax, negative in zip(axes[0], negatives):
        for view in VIEWS:
            group = predictions[(predictions.view == view) & predictions.input_type.isin([negative, "poison"])]
            fpr, tpr, _ = roc_curve(group.label, group.score)
            ax.plot(fpr, tpr, label=view)
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"poison vs {negative}")
        ax.legend(fontsize=8)
    fig.savefig(output / "figures/roc_control.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(negatives), figsize=(5 * len(negatives), 4),
                             squeeze=False, layout="constrained")
    for ax, negative in zip(axes[0], negatives):
        group = metrics[metrics.comparison == f"poison_vs_{negative}"].set_index("view").loc[list(VIEWS)]
        ax.bar(range(len(VIEWS)), group.AUROC, alpha=.7)
        ax.vlines(range(len(VIEWS)), group.CI_low, group.CI_high, color="black")
        ax.set(xticks=range(len(VIEWS)), xticklabels=VIEWS, ylim=(0, 1),
               ylabel="AUROC (95% CI)", title=f"poison vs {negative}")
        ax.tick_params(axis="x", rotation=35)
    fig.savefig(output / "figures/ablation.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for ax, view in zip(axes.flat, VIEWS):
        group = predictions[predictions.view == view]
        for kind in [*negatives, "poison"]:
            ax.hist(group[group.input_type == kind].score, bins=15, alpha=.5, label=kind)
        ax.axvline(group.threshold.iloc[0], color="black", linestyle="--")
        ax.set(title=view, xlabel="Detector score", ylabel="Samples")
        ax.legend(fontsize=8)
    axes.flat[-1].set_visible(False)
    fig.savefig(output / "figures/score_distribution.pdf")
    plt.close(fig)
    frame = pd.read_csv(output / "results/features.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, column in zip(axes, ("cps_score", "entropy_mean", "hidden_norm_mean")):
        for kind in [*negatives, "poison"]:
            ax.hist(frame[frame.input_type == kind][column], bins=15, alpha=.5, label=kind)
        ax.set(title=column, ylabel="Samples")
        ax.legend(fontsize=8)
    fig.savefig(output / "figures/feature_distribution.pdf")
    plt.close(fig)
