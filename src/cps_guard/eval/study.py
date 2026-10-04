"""主结果、消融、扰动次数、错误样本和 Pilot 决策的统计入口。"""

from __future__ import annotations

import pandas as pd

from .detection import evaluate_long_scores, select_test_ids
from ..data.schema import read_samples, write_rows


def _attach_samples(scores: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """按 sample_id 合并方法分数与样本配对、攻击及标签元数据。

    输入：
        scores（pd.DataFrame）：含 sample_id、method、score、runtime_sec、query_count 的表，每方法完整且 sample_id 唯一。
        samples（pd.DataFrame）：sample_id 唯一的统一样本表，覆盖 scores，含 pair_id/base_id/attack/label。

    输出：
        pandas.DataFrame：方法长表，保留分数/成本，添加 pair_id、base_id、attack、label；不修改原表或写文件。
    """
    return scores[["sample_id", "method", "score", "runtime_sec", "query_count"]].merge(
        samples[["sample_id", "pair_id", "base_id", "attack", "label"]], on="sample_id")


def compare_methods(samples_csv: str, cps_csv: str, baseline_csvs: list[str],
                    output_csv: str, seed: int = 20261004) -> int:
    """把 CPS、校正 CPS 和基线组织为共同长表，使用同一分组测试集比较。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        cps_csv（str）：CPS 汇总 CSV，含 sample_id、三类分数、cps_score、cps_cal_score、randomness_baseline、runtime_sec、query_count。
        baseline_csvs（list[str]）：基线 CSV 路径列表，可为空；含 BASELINE_COLUMNS，各方法恰好覆盖全部统一样本。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。

    输出：
        int：指标行数=(2+基线方法数)×(攻击种类数+1)，写出统一评价字段；不运行模型或重算分数。
    """
    samples = read_samples(samples_csv)
    cps = pd.read_csv(cps_csv, keep_default_na=False)
    scores = []
    for method, column in (("CPS", "cps_score"), ("CPS-calibrated", "cps_cal_score")):
        subset = cps[["sample_id", column, "runtime_sec", "query_count"]].copy()
        subset = subset.rename(columns={column: "score"})
        subset["method"] = method
        scores.append(subset)
    for path in baseline_csvs:
        scores.append(pd.read_csv(path, keep_default_na=False))
    merged = _attach_samples(pd.concat(scores, ignore_index=True), samples)
    rows = evaluate_long_scores(merged, seed)
    write_rows(output_csv, rows, list(rows[0]))
    return len(rows)


def ablation_table(samples_csv: str, cps_csv: str, output_csv: str,
                   seed: int = 20261004, lambda_randomness: float = 1.0) -> int:
    """三类分量组成三个单项、三个两两组合及完整组合；分别校正/不校正随机性，评估 14 个版本。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        cps_csv（str）：CPS 汇总 CSV，含 sample_id、三类分数、cps_score、cps_cal_score、randomness_baseline、runtime_sec、query_count。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。
        lambda_randomness（float）：随机性扣除系数 λ；校正分数=原始分数−λ×随机性基线 B。 默认值：1.0。

    输出：
        int：指标行数=14×(攻击种类数+1)，写出统一评价字段；重用已有分数，无法拆分的单独运行成本留空。
    """
    samples = read_samples(samples_csv)
    cps = pd.read_csv(cps_csv, keep_default_na=False)
    # 1. 定义七个分量组合，每个再分校正前后。
    combos = {
        "Semantic": ["semantic_score"], "Context": ["context_score"],
        "Position": ["position_score"],
        "Semantic+Context": ["semantic_score", "context_score"],
        "Semantic+Position": ["semantic_score", "position_score"],
        "Context+Position": ["context_score", "position_score"],
        "Full": ["semantic_score", "context_score", "position_score"],
    }
    rows = []
    for name, columns in combos.items():
        base = cps[columns].astype(float).mean(axis=1)
        for calibrated in (False, True):
            current = cps[["sample_id"]].copy()
            current["runtime_sec"] = ""
            current["query_count"] = ""
            current["method"] = name + ("+cal" if calibrated else "")
            current["score"] = base - lambda_randomness * cps.randomness_baseline.astype(float) if calibrated else base
            rows.append(current)
    long = _attach_samples(pd.concat(rows, ignore_index=True), samples)
    results = evaluate_long_scores(long, seed)
    write_rows(output_csv, results, list(results[0]))
    return len(results)


def perturbation_sensitivity(samples_csv: str, details_csv: str, cps_csv: str,
                             output_csv: str, counts: tuple[int, ...] = (1, 3, 5, 10),
                             seed: int = 20261004,
                             lambda_randomness: float = 1.0) -> int:
    """取每类编号 1～N 的实测距离重算校正 CPS；用完整耗时减全部扰动耗时再加选中耗时，得到各 N 的生成成本。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        details_csv（str）：仅含 semantic/context/position 的完整明细；各样本各类均有 full_n 次，编号 1～full_n，含实测 distance/runtime_sec（秒）。
        cps_csv（str）：CPS 汇总 CSV，含 sample_id、三类分数、cps_score、cps_cal_score、randomness_baseline、runtime_sec、query_count。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        counts（tuple[int, ...]）：待评估的正整数 N 元组，各 N 不超过实测 full_n。 默认值：(1, 3, 5, 10)。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。
        lambda_randomness（float）：随机性扣除系数 λ；校正分数=原始分数−λ×随机性基线 B。 默认值：1.0。

    输出：
        int：指标行数=len(counts)×(攻击种类数+1)，method=N=数值，查询数=1+3N+随机重复次数；重复次数从完整 query_count 推出，不生成新扰动或查询模型。
    """
    samples = read_samples(samples_csv)
    details = pd.read_csv(details_csv, keep_default_na=False)
    cps = pd.read_csv(cps_csv, keep_default_na=False)
    details["runtime_sec"] = details.runtime_sec.astype(float)
    details["perturb_id"] = details.perturb_id.astype(int)
    # 1. 从完整明细确定扰动次数，分离可重用的生成成本。
    full_n = details.groupby(["sample_id", "perturb_type"]).size().iloc[0]
    all_perturb_runtime = details.groupby("sample_id").runtime_sec.sum()
    rows = []
    # 2. 每个 N 只用前 N 个实测扰动，重算分数与查询成本。
    for n in counts:
        selected = details[(details.perturb_type.isin(["semantic", "context", "position"]))
                           & (details.perturb_id <= n)]
        aggregate = selected.groupby(["sample_id", "perturb_type"], as_index=False).distance.mean()
        wide = aggregate.pivot(index="sample_id", columns="perturb_type", values="distance")
        base = cps[["sample_id", "randomness_baseline", "query_count", "runtime_sec"]].merge(
            wide.mean(axis=1).rename("raw_score"), left_on="sample_id", right_index=True)
        base["method"] = f"N={n}"
        base["score"] = base.raw_score - lambda_randomness * base.randomness_baseline.astype(float)
        repeats = pd.to_numeric(base.query_count) - 1 - 3 * full_n
        base["query_count"] = 1 + 3 * n + int(repeats.iloc[0])
        selected_runtime = selected.groupby("sample_id").runtime_sec.sum()
        base["runtime_sec"] = (pd.to_numeric(base.runtime_sec)
                               - base.sample_id.map(all_perturb_runtime).astype(float)
                               + base.sample_id.map(selected_runtime).astype(float))
        rows.append(base[["sample_id", "method", "score", "runtime_sec", "query_count"]])
    long = _attach_samples(pd.concat(rows, ignore_index=True), samples)
    results = evaluate_long_scores(long, seed)
    write_rows(output_csv, results, list(results[0]))
    return len(results)


def pilot_decision(main_csv: str, asr_csv: str, output_csv: str,
                   min_asr: float) -> int:
    """先按预定 ASR 门槛判定攻击有效性，再按校正 CPS AUROC 分档：<0.60 停止，≤0.75 错误分析，≤0.80 可扩样，更高进入正式实验。

    输入：
        main_csv（str）：主评价 CSV，包含 method、attack、AUROC、threshold_train 及成本指标。
        asr_csv（str）：ASR 汇总 CSV，每攻击有唯一 attack/ASR 记录。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        min_asr（float）：看结果前确定的有效性下限，范围 [0,1]，低于该值先修复攻击/数据。

    输出：
        int：输出攻击种类数，CSV 含 attack、ASR、ASR_min_required、AUROC、decision；不自动扩样或训练。
    """
    metrics = pd.read_csv(main_csv)
    asr = pd.read_csv(asr_csv)
    rows = []
    for attack in sorted(set(metrics.attack) - {"ALL"}):
        selected = metrics[(metrics.method == "CPS-calibrated") & (metrics.attack == attack)]
        measured = asr[asr.attack == attack]
        auc, success = float(selected.iloc[0].AUROC), float(measured.iloc[0].ASR)
        if success < min_asr:
            decision = "先修复攻击/数据，不能解释检测指标"
        elif auc < 0.60:
            decision = "停止扩样，修改扰动定义"
        elif auc <= 0.75:
            decision = "弱信号，做错误分析和消融"
        elif auc <= 0.80:
            decision = "可扩到正式样本并补齐基线"
        else:
            decision = "进入正式主结果与论文整理"
        rows.append({"attack": attack, "ASR": success, "ASR_min_required": min_asr,
                     "AUROC": auc, "decision": decision})
    write_rows(output_csv, rows, list(rows[0]))
    return len(rows)


def export_detection_errors(samples_csv: str, cps_csv: str, main_csv: str,
                            output_csv: str, seed: int = 20261004,
                            test_fraction: float = 0.3) -> int:
    """共用主评价测试集及每攻击训练阈值，导出校正 CPS 的误报/漏报供人工分析。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        cps_csv（str）：CPS 汇总 CSV，含 sample_id、三类分数、cps_score、cps_cal_score、randomness_baseline、runtime_sec、query_count。
        main_csv（str）：主评价 CSV，包含 method、attack、AUROC、threshold_train 及成本指标。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。
        test_fraction（float）：独立 base_id 分入测试集的比例，在 (0,1) 内；各攻击划分后的训练/测试集均应包含两类标签。 默认值：0.3。

    输出：
        int：错误样本数；CSV 含原文、标签、预测、错误类型、分数、阈值、三类分量、B、空白 manual_note；无错误时仅写表头。
    """
    samples = read_samples(samples_csv)
    scores = pd.read_csv(cps_csv, keep_default_na=False)
    metrics = pd.read_csv(main_csv, keep_default_na=False)
    test_ids = select_test_ids(samples.base_id.unique(), seed, test_fraction)
    test = samples[samples.base_id.isin(test_ids)].merge(
        scores[["sample_id", "cps_cal_score", "semantic_score", "context_score",
                "position_score", "randomness_baseline"]],
        on="sample_id", how="inner")
    # 对每个测试样本应用其攻击类型的训练阈值，只保留误报/漏报。
    thresholds = metrics[metrics.method == "CPS-calibrated"].set_index("attack")
    rows = []
    for sample in test.itertuples(index=False):
        threshold = float(thresholds.loc[sample.attack, "threshold_train"])
        predicted = int(sample.cps_cal_score >= threshold)
        if predicted == int(sample.label):
            continue
        rows.append({"sample_id": sample.sample_id, "base_id": sample.base_id,
                     "attack": sample.attack, "label": int(sample.label),
                     "prediction": predicted,
                     "error_type": "false_positive" if predicted else "false_negative",
                     "input_text": sample.input_text,
                     "score": sample.cps_cal_score, "threshold_train": threshold,
                     "semantic_score": sample.semantic_score,
                     "context_score": sample.context_score,
                     "position_score": sample.position_score,
                     "randomness_baseline": sample.randomness_baseline,
                     "manual_note": ""})
    columns = ["sample_id", "base_id", "attack", "label", "prediction", "error_type",
               "input_text", "score", "threshold_train", "semantic_score",
               "context_score", "position_score", "randomness_baseline", "manual_note"]
    write_rows(output_csv, rows, columns)
    return len(rows)
