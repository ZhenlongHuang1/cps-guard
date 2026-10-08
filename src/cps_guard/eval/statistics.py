"""Pilot-v2：按基础问题配对 bootstrap、Random 多种子和冻结结果报告。"""
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
    输出：None；bootstrap_results.csv、bootstrap_draws.npz、ablation_results.csv、
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
    # 3. 每个消融都与同一次抽样中的融合结果作差。
    metrics = pd.read_csv(results / "bootstrap_results.csv").set_index(["source_attack", "target_attack", "view"])
    rows = []
    for source in ("badnet", "vpi"):
        for target in ("badnet", "vpi"):
            full = draws[f"{source}__{target}__B+G+R"]
            for removed, view in (("B", "G+R"), ("G", "B+R"), ("R", "B+G")):
                delta = full - draws[f"{source}__{target}__{view}"]
                lower, upper = np.quantile(delta, [0.025, 0.975])
                rows.append({"source_attack": source, "target_attack": target, "removed_view": removed,
                             "reduced_view": view, "AUROC_gain": metrics.loc[(source, target, "B+G+R"), "AUROC"]
                             - metrics.loc[(source, target, view), "AUROC"], "gain_CI_low": lower, "gain_CI_high": upper})
    pd.DataFrame(rows).to_csv(results / "ablation_results.csv", index=False)


def write_final_report(output: Path) -> str:
    """对应 STEP 19：汇总固定测试的五个研究问题与探索性 GO/Conditional/NO-GO。

    输入：已完成正式测试、bootstrap 和 Random 的目录。
    输出：决策字符串，写 results/final_report.md。GO 需双向融合 AUROC≥0.70、
        CI下界>0.5且高于对应 Random 的97.5%分位；Conditional 需双向≥0.60且
        对最佳单视图的配对增益CI下界>0。判据为本 Pilot 工程判据，不作多重检验声明。
    """
    results = output / "results"
    summary = pd.read_csv(results / "bootstrap_results.csv")
    cross = summary[summary.source_attack != summary.target_attack]
    full = cross[cross.view == "B+G+R"]
    random = pd.read_csv(results / "random_baseline.csv").set_index("target_attack")
    draws = np.load(results / "bootstrap_draws.npz")
    gains = []
    for row in full.itertuples(index=False):
        singles = cross[(cross.source_attack == row.source_attack) & cross.view.isin(["B", "G", "R"])]
        best = singles.loc[singles.AUROC.idxmax()]
        delta = draws[f"{row.source_attack}__{row.target_attack}__B+G+R"] - draws[
            f"{row.source_attack}__{row.target_attack}__{best['view']}"]
        gains.append(float(np.quantile(delta, 0.025)))
    go = all(row.AUROC >= 0.70 and row.CI_low > 0.5 and row.AUROC > random.loc[row.target_attack, "p975"]
             for row in full.itertuples(index=False))
    conditional = full.AUROC.ge(0.60).all() and all(gain > 0 for gain in gains)
    decision = "GO" if go else "Conditional GO" if conditional else "NO-GO"
    asr = pd.read_csv(results / "asr.csv")
    def table(frame):
        """将汇总表转换为 Markdown；不依赖额外的 tabulate 包。"""
        return "\n".join(["| " + " | ".join(frame.columns) + " |",
                          "| " + " | ".join(["---"] * len(frame.columns)) + " |", *[
                              "| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value)
                                                 for value in record) + " |" for record in frame.itertuples(index=False, name=None)]])
    sections = ["# Pilot-v2 冻结测试报告", f"\n工程决策：**{decision}**。",
                "\n## 1. 攻击是否有效", table(asr),
                "\n## 2. 单视图能否检测及跨攻击迁移", table(cross),
                "\n## 3. IID 检测", table(summary[summary.source_attack == summary.target_attack]),
                "\n## 4. 融合是否互补", table(pd.read_csv(results / "ablation_results.csv")),
                "\n## 5. 是否超过随机基线", table(random.reset_index()),
                "\n## 解释范围",
                "- 表征为输入 prompt 的隐藏层，不包含回答；G 使用处理后生成 logits。",
                "- AUROC 区间针对本次固定模型及100个 Test 基础问题，不包含重新训练的变异。",
                "- 同题 clean/poison 和不同检测器使用配对重采样。Random 分位范围不是均值CI。",
                "- 最佳单视图在报告中按 Test 点估计展示，仅用于探索性比较；未校正选择偏差和多重比较。",
                "- 目标行为按固定前缀判定；不代表所有负面措辞。AUPRC采用average precision。",
                "- Test每目标100条clean，1% FPR对应至多1条误报，低误报率结果分辨率有限。",
                "- 任一 Gate 未通过或 NO-GO，应先解释本 Pilot，不能以追加 Test 调参替代独立实验。"]
    (results / "final_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return decision
