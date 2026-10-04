"""主结果、消融、扰动次数、错误样本和 Pilot 决策的统计入口。"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .detection import _bootstrap_auc, _threshold
from ..data.schema import read_samples, write_rows


def _attach_samples(scores: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """把方法分数与统一样本表严格一一对应，防止漏样本或标签错位。"""
    required = {"sample_id", "method", "score"}
    if required - set(scores) or scores.duplicated(["sample_id", "method"]).any():
        raise ValueError("分数表缺少 sample_id/method/score 或存在重复")
    for method, group in scores.groupby("method"):
        if set(group.sample_id) != set(samples.sample_id):
            raise ValueError(f"{method} 的样本集合与统一 CSV 不一致")
    if "attack" in scores:
        expected = scores[["sample_id", "attack"]].merge(
            samples[["sample_id", "attack"]], on="sample_id", suffixes=("_score", "_sample"))
        present = expected.attack_score.notna() & expected.attack_score.astype(str).ne("")
        if (expected.loc[present, "attack_score"] != expected.loc[present, "attack_sample"]).any():
            raise ValueError("分数表与统一 CSV 的攻击标签冲突")
    if "label" in scores:
        expected = scores[["sample_id", "label"]].merge(
            samples[["sample_id", "label"]], on="sample_id", suffixes=("_score", "_sample"))
        present = expected.label_score.notna() & expected.label_score.astype(str).ne("")
        if (pd.to_numeric(expected.loc[present, "label_score"]) != expected.loc[present, "label_sample"]).any():
            raise ValueError("分数表与统一 CSV 的 clean/poison 标签冲突")
    scores = scores.drop(columns=["attack", "label", "pair_id", "base_id"], errors="ignore")
    return scores.merge(samples[["sample_id", "pair_id", "base_id", "attack", "label"]],
                        on="sample_id", how="inner", validate="many_to_one")


def evaluate_long_scores(frame: pd.DataFrame, seed: int = 20261004,
                         test_fraction: float = 0.3,
                         bootstrap_repeats: int = 500) -> list[dict]:
    """所有方法共用 base_id 留出集；阈值只由训练集选择。"""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction 必须在 0 和 1 之间")
    ids = sorted(frame.base_id.unique())
    if len(ids) < 10:
        raise ValueError("至少需要 10 条独立原始指令才能评估")
    random.Random(seed).shuffle(ids)
    test_ids = set(ids[:max(1, round(len(ids) * test_fraction))])
    frame = frame.copy()
    frame["split"] = np.where(frame.base_id.isin(test_ids), "test", "train")
    frame["score"] = pd.to_numeric(frame.score, errors="coerce")
    if frame.score.isna().any() or not np.isfinite(frame.score).all():
        raise ValueError("所有方法分数必须为有限数")
    rows = []
    for method, method_frame in frame.groupby("method", sort=True):
        groups = list(method_frame.groupby("attack", sort=True)) + [("ALL", method_frame)]
        for attack, group in groups:
            train = group[group.split == "train"]
            test = group[group.split == "test"]
            if train.label.nunique() != 2 or test.label.nunique() != 2:
                raise ValueError(f"{method}/{attack} 的训练和测试集都必须有两个类别")
            threshold = _threshold(train.label.to_numpy(), train.score.to_numpy())
            predictions = (test.score.to_numpy() >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test.label, predictions, average="binary", zero_division=0)
            lower, upper = _bootstrap_auc(test, "score", seed, bootstrap_repeats)
            runtime = pd.to_numeric(test.get("runtime_sec", pd.Series(dtype=float)), errors="coerce")
            queries = pd.to_numeric(test.get("query_count", pd.Series(dtype=float)), errors="coerce")
            rows.append({
                "method": method, "attack": attack, "n_train": len(train),
                "n_test": len(test), "n_independent_test_prompts": test.base_id.nunique(),
                "AUROC": float(roc_auc_score(test.label, test.score)),
                "AUROC_CI_low": lower, "AUROC_CI_high": upper,
                "threshold_train": threshold, "Precision": float(precision),
                "Recall": float(recall), "F1": float(f1),
                "runtime_sec_total_test": float(runtime.sum()) if runtime.notna().all() else "",
                "query_count_total_test": int(queries.sum()) if queries.notna().all() else "",
            })
    return rows


def compare_methods(samples_csv: str, cps_csv: str, baseline_csvs: list[str],
                    output_csv: str, seed: int = 20261004) -> int:
    """统一评估 CPS、CPS-cal 和全部给定基线，拒绝缺失样本的比较。"""
    samples = read_samples(samples_csv)
    cps = pd.read_csv(cps_csv, keep_default_na=False)
    for column in ("cps_score", "cps_cal_score"):
        if column not in cps:
            raise ValueError(f"CPS 表缺少 {column}")
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
    """计算三类单项、两两组合和完整组合，并分别报告校正前后。"""
    samples = read_samples(samples_csv)
    cps = pd.read_csv(cps_csv, keep_default_na=False)
    combos = {
        "Semantic": ["semantic_score"], "Context": ["context_score"],
        "Position": ["position_score"],
        "Semantic+Context": ["semantic_score", "context_score"],
        "Semantic+Position": ["semantic_score", "position_score"],
        "Context+Position": ["context_score", "position_score"],
        "Full": ["semantic_score", "context_score", "position_score"],
    }
    needed = {"sample_id", "randomness_baseline"} | set().union(*map(set, combos.values()))
    if needed - set(cps):
        raise ValueError(f"消融输入缺少列：{sorted(needed - set(cps))}")
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
    """用前 N 个预先生成的独立扰动估计成本与 AUROC；不足 N 时直接报错。"""
    samples = read_samples(samples_csv)
    details = pd.read_csv(details_csv, keep_default_na=False)
    cps = pd.read_csv(cps_csv, keep_default_na=False)
    needed = {"sample_id", "perturb_type", "perturb_id", "distance", "runtime_sec"}
    if needed - set(details):
        raise ValueError(f"详细结果缺少列：{sorted(needed - set(details))}")
    details["runtime_sec"] = pd.to_numeric(details.runtime_sec, errors="coerce")
    if details.runtime_sec.isna().any() or (details.runtime_sec < 0).any():
        raise ValueError("逐扰动运行时间必须是非负数")
    if not counts or any(n < 1 for n in counts):
        raise ValueError("扰动次数必须为正整数")
    max_n = max(counts)
    observed_counts = set()
    for (sid, kind), group in details.groupby(["sample_id", "perturb_type"]):
        if kind in {"semantic", "context", "position"}:
            ids = set(pd.to_numeric(group.perturb_id))
            if not set(range(1, max_n + 1)).issubset(ids):
                raise ValueError(f"{sid}/{kind} 不足 {max_n} 个扰动；先补采样再做 N 敏感性")
            observed_counts.add(len(ids))
    if len(observed_counts) != 1:
        raise ValueError("各样本的完整扰动次数必须一致")
    full_n = observed_counts.pop()
    rows = []
    for n in counts:
        selected = details[(details.perturb_type.isin(["semantic", "context", "position"]))
                           & (pd.to_numeric(details.perturb_id) <= n)]
        aggregate = selected.groupby(["sample_id", "perturb_type"], as_index=False).distance.mean()
        wide = aggregate.pivot(index="sample_id", columns="perturb_type", values="distance")
        if wide.isna().any().any() or set(wide.columns) != {"semantic", "context", "position"}:
            raise ValueError(f"N={n} 的三类扰动不完整")
        base = cps[["sample_id", "randomness_baseline", "query_count", "runtime_sec"]].merge(
            wide.mean(axis=1).rename("raw_score"), left_on="sample_id", right_index=True,
            validate="one_to_one")
        base["method"] = f"N={n}"
        base["score"] = base.raw_score - lambda_randomness * base.randomness_baseline.astype(float)
        repeats = pd.to_numeric(base.query_count) - 1 - 3 * full_n
        if repeats.nunique() != 1 or repeats.iloc[0] < 2:
            raise ValueError("无法从完整 CPS 查询数推断随机重复次数")
        base["query_count"] = 1 + 3 * n + int(repeats.iloc[0])
        all_perturb_runtime = details.groupby("sample_id").runtime_sec.sum()
        selected_runtime = selected.groupby("sample_id").runtime_sec.sum()
        base["runtime_sec"] = (pd.to_numeric(base.runtime_sec)
                               - base.sample_id.map(all_perturb_runtime).astype(float)
                               + base.sample_id.map(selected_runtime).astype(float))
        if base.runtime_sec.isna().any() or (base.runtime_sec < -1e-6).any():
            raise ValueError("扰动明细运行时间与完整 CPS 运行时间不一致")
        rows.append(base[["sample_id", "method", "score", "runtime_sec", "query_count"]])
    long = _attach_samples(pd.concat(rows, ignore_index=True), samples)
    results = evaluate_long_scores(long, seed)
    write_rows(output_csv, results, list(results[0]))
    return len(results)


def pilot_decision(main_csv: str, asr_csv: str, output_csv: str,
                   min_asr: float) -> int:
    """综合预先指定的 ASR 下限与 Pilot AUROC 分档，给出扩样建议。"""
    if not 0 <= min_asr <= 1:
        raise ValueError("min_asr 必须在 0 到 1 之间，并应在看结果前确定")
    metrics = pd.read_csv(main_csv)
    asr = pd.read_csv(asr_csv)
    required = {"attack", "ASR"}
    if required - set(asr):
        raise ValueError("ASR 表缺少 attack/ASR")
    rows = []
    for attack in sorted(set(metrics.attack) - {"ALL"}):
        selected = metrics[(metrics.method == "CPS-calibrated") & (metrics.attack == attack)]
        measured = asr[asr.attack == attack]
        if len(selected) != 1 or len(measured) != 1:
            raise ValueError(f"{attack} 缺少唯一的 CPS-calibrated 或 ASR 记录")
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
    """导出测试集误报和漏报的原文、分数与阈值，供人工检查原因。"""
    samples = read_samples(samples_csv)
    scores = pd.read_csv(cps_csv, keep_default_na=False)
    metrics = pd.read_csv(main_csv, keep_default_na=False)
    ids = sorted(samples.base_id.unique())
    random.Random(seed).shuffle(ids)
    test_ids = set(ids[:max(1, round(len(ids) * test_fraction))])
    test = samples[samples.base_id.isin(test_ids)].merge(
        scores[["sample_id", "cps_cal_score", "semantic_score", "context_score",
                "position_score", "randomness_baseline"]],
        on="sample_id", how="inner", validate="one_to_one")
    if len(test) != len(samples[samples.base_id.isin(test_ids)]):
        raise ValueError("错误分析缺少测试集分数")
    thresholds = metrics[metrics.method == "CPS-calibrated"].set_index("attack")
    rows = []
    for sample in test.itertuples(index=False):
        if sample.attack not in thresholds.index:
            raise ValueError(f"缺少 {sample.attack} 的训练集阈值")
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
