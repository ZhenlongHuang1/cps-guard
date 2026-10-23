"""R单视图ROC、随机基线、得分及内部特征分布。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve


def draw_roc(ax, group: pd.DataFrame, title: str) -> None:
    """在一个坐标轴画固定R预测的ROC及随机参考线。

    输入：ax、含label/score的单个比较数据及标题。
    输出：None；只绘制测试结果，不重新选择检测器或阈值。
    """
    fpr, tpr, _ = roc_curve(group.label, group.score)
    ax.plot(fpr, tpr, label="R")
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend()


def plot_pilot(output: Path, test_output: Path | None = None) -> None:
    """输出R同攻击/双向迁移ROC、随机基线和Test内部特征分布四张PDF。

    输入：根目录中的固定预测和统计；test_output指定独立Test特征目录。
    输出：None；figures/roc_iid、roc_cross_attack、random_distribution、
        feature_distribution.pdf。特征分布展示预先固定的范数和相邻层变化量，
        不再绘制B/G特征或融合消融图。
    """
    results, figures = output / "results", output / "figures"
    predictions = pd.read_csv(results / "test_predictions.csv")
    for cross, name in ((False, "roc_iid"), (True, "roc_cross_attack")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, source in zip(axes, ("badnet", "vpi")):
            target = ("vpi" if source == "badnet" else "badnet") if cross else source
            group = predictions[(predictions.source_attack == source) & (predictions.target_attack == target)]
            draw_roc(ax, group, f"{source} -> {target}")
        fig.savefig(figures / f"{name}.pdf")
        plt.close(fig)
    random = pd.read_csv(results / "random_seed_results.csv")
    cross = pd.read_csv(results / "cross_attack_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, target in zip(axes, ("badnet", "vpi")):
        ax.hist(random[random.target_attack == target].AUROC, bins=12, alpha=.7)
        ax.axvline(cross[cross.target_attack == target].AUROC.iloc[0], color="red", label="Cross-attack R")
        ax.set(xlabel="AUROC", ylabel="Number of seeds", title=target)
        ax.legend()
    fig.savefig(figures / "random_distribution.pdf")
    plt.close(fig)
    frame = pd.read_csv((test_output or output) / "results/features.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for row, attack in enumerate(("badnet", "vpi")):
        test = frame[(frame.attack == attack) & (frame.split == "test")]
        for ax, column in zip(axes[row], ("hidden_norm_mean", "layer_shift_mean", "layer_shift_max")):
            for label, name in ((0, "clean"), (1, "poison")):
                ax.hist(test[test.label == label][column], bins=15, alpha=.55, label=name)
            ax.set(title=f"{attack}: {column}", ylabel="Samples")
            ax.legend(fontsize=8)
    fig.savefig(figures / "feature_distribution.pdf")
    plt.close(fig)


def plot_control(output: Path) -> None:
    """对clean/poison及可选neutral输出R对照的三张PDF。

    输入：对照目录中的固定预测及R特征。
    输出：None；roc_control、score_distribution、feature_distribution.pdf。
        poison对clean和poison对neutral分别画ROC；得分图标出来源固定阈值。
    """
    predictions = pd.read_csv(output / "results/test_predictions.csv")
    negatives = [kind for kind in ("clean", "neutral") if kind in predictions.input_type.values]
    fig, axes = plt.subplots(1, len(negatives), figsize=(5 * len(negatives), 4), squeeze=False, layout="constrained")
    for ax, negative in zip(axes[0], negatives):
        group = predictions[predictions.input_type.isin([negative, "poison"])]
        draw_roc(ax, group, f"poison vs {negative}")
    fig.savefig(output / "figures/roc_control.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for kind in [*negatives, "poison"]:
        ax.hist(predictions[predictions.input_type == kind].score, bins=15, alpha=.5, label=kind)
    ax.axvline(predictions.threshold.iloc[0], color="black", linestyle="--", label="Source threshold")
    ax.set(xlabel="R detector score", ylabel="Samples")
    ax.legend()
    fig.savefig(output / "figures/score_distribution.pdf")
    plt.close(fig)
    frame = pd.read_csv(output / "results/features.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, column in zip(axes, ("hidden_norm_mean", "layer_shift_mean", "layer_shift_max")):
        for kind in [*negatives, "poison"]:
            ax.hist(frame[frame.input_type == kind][column], bins=15, alpha=.5, label=kind)
        ax.set(title=column, ylabel="Samples")
        ax.legend(fontsize=8)
    fig.savefig(output / "figures/feature_distribution.pdf")
    plt.close(fig)
