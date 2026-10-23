"""R单视图：按基础问题配对bootstrap、随机基线和固定检测器报告。"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def grouped_auc(group: pd.DataFrame, weights: np.ndarray, base_ids: list[str]) -> np.ndarray:
    """对应 STEP 16：按基础问题重采样，同题 clean/poison 保持配对。

    公式：AUC=Σ_iΣ_j w_i w_j [s_poison,i>s_clean,j（相等计0.5）]/(Σw)^2。
    输入：单检测器/目标的预测表；weights 为 (重复次数,问题数) 的抽样次数；
        base_ids 指定共同的问题顺序，每题有一个 clean 和一个 poison。
    输出：每次重采样的 AUROC 数组；多个方法传入同一 weights，得到配对比较。
    """
    paired = group.pivot(index="base_id", columns="label", values="score").loc[base_ids]
    poison, clean = paired[1].to_numpy(), paired[0].to_numpy()
    comparison = (poison[:, None] > clean).astype(float)
    comparison += 0.5 * (poison[:, None] == clean)
    return ((weights @ comparison) * weights).sum(axis=1) / weights.sum(axis=1) ** 2


def summary_statistics(output: Path, repeats: int, bootstrap_seed: int,
                       random_seeds: int, random_seed: int) -> None:
    """对应 STEP 16/17/18：固定 Test 预测的区间、配对消融差值和 Random 分布。

    输入：实验目录、bootstrap 次数/种子、Random 独立种子数/起始种子。
    输出：None；bootstrap_results.csv、bootstrap_draws.npz、
        random_seed_results.csv、random_baseline.csv。区间为百分位 95% CI；
        所有视图/攻击共享问题抽样次数。只重采样预测，不重训检测器。
    """
    results = output / "results"
    predictions = pd.read_csv(results / "test_predictions.csv")
    base_ids = sorted(predictions.base_id.unique())
    weights = np.random.default_rng(bootstrap_seed).multinomial(
        len(base_ids), np.full(len(base_ids), 1 / len(base_ids)), size=repeats)
    rows, draws = [], {}
    # 1. 相同问题抽样应用于全部检测器和两种目标攻击。
    for (source, target, view), group in predictions.groupby(["source_attack", "target_attack", "view"]):
        values = grouped_auc(group, weights, base_ids)
        draws[f"{source}__{target}__{view}"] = values
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append({"source_attack": source, "target_attack": target, "view": view,
                     "AUROC": roc_auc_score(group.label, group.score), "CI_low": lower,
                     "CI_high": upper, "bootstrap_repeats": repeats, "bootstrap_seed": bootstrap_seed})
    pd.DataFrame(rows).to_csv(results / "bootstrap_results.csv", index=False)
    np.savez_compressed(results / "bootstrap_draws.npz", weights=weights,
                        base_ids=np.array(base_ids), **draws)
    # 2. 随机分数是独立随机排序；种子分布不是对均值的置信区间。
    rows = []
    for offset, target in enumerate(("badnet", "vpi")):
        samples = predictions[predictions.target_attack == target].drop_duplicates("sample_id").sort_values("sample_id")
        for index in range(random_seeds):
            seed = random_seed + 2 * index + offset
            scores = np.random.default_rng(seed).random(len(samples))
            rows.append({"target_attack": target, "seed": seed,
                         "AUROC": roc_auc_score(samples.label, scores)})
    random_frame = pd.DataFrame(rows)
    random_frame.to_csv(results / "random_seed_results.csv", index=False)
    rows = [{"target_attack": target, "mean": group.AUROC.mean(), "std": group.AUROC.std(),
             "p025": group.AUROC.quantile(0.025), "p975": group.AUROC.quantile(0.975),
             "n_seeds": len(group)} for target, group in random_frame.groupby("target_attack")]
    pd.DataFrame(rows).to_csv(results / "random_baseline.csv", index=False)


def write_final_report(output: Path) -> Path:
    """汇总R单视图的攻击有效性、同攻击、双向迁移及随机基线。

    输入：第四步完成的本轮根目录，含ASR、bootstrap和随机基线表。
    输出：报告Path；results/final_report.md保存原指标，不再输出B/G融合消融或融合决策。
        AUROC置信区间只针对本次固定模型的Test问题，真实攻击行为另看ASR。
    """
    results = output / "results"
    sections = ["# R单视图迁移测试", "R来自输入prompt内部，不包含生成回答；来源参考、scaler、LR及阈值在测试时固定。"]
    for title, name in (("攻击有效性（Test）", "asr.csv"),
                        ("同攻击检测", "iid_results.csv"),
                        ("双向迁移", "cross_attack_results.csv"),
                        ("配对AUROC区间", "bootstrap_results.csv"),
                        ("随机基线", "random_baseline.csv")):
        frame = pd.read_csv(results / name)
        sections.extend([f"## {title}", f"```text\n{frame.to_string(index=False)}\n```"])
    sections.append("目标行为按各LoRA自己的固定前缀判定；检测AUROC不能替代ASR。区间来自同题clean/poison配对bootstrap，不包含重新训练变异。Test每攻击100条clean，1%FPR的分辨率有限。")
    target = results / "final_report.md"
    target.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return target
