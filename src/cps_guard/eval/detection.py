"""分组留出检测评价与 AUROC 置信区间。"""
from __future__ import annotations
import random
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from ..data.schema import write_rows


def _threshold(y: np.ndarray, score: np.ndarray) -> float:
    """在训练集 ROC 上选择使 Youden J=TPR−FPR 最大的有限阈值，并列取 ROC 顺序第一项。

    实验方案对应：
        S12 score+threshold 检测决策；对应第十六节“检测器不必训练，Pilot 可由 score+threshold 完成”。

    算法/公式：
        本实现仅用训练集选择 t*=argmax_t[TPR_train(t)−FPR_train(t)]（Youden J），测试预测 ŷ=1[score≥t*]；Youden 规则是实现选择，方案未指定阈值算法。

    输入：
        y（np.ndarray）：训练集一维 0/1 标签数组，包含两类。
        score（np.ndarray）：与 y 逐项对应的一维有限分数，越高越可疑。

    输出：
        float：训练检测阈值；score≥阈值判为 poison，不使用测试数据。
    """
    fpr, tpr, thresholds = roc_curve(y, score)
    finite = np.isfinite(thresholds)
    candidates = np.where(finite)[0]
    return float(thresholds[candidates[np.argmax((tpr - fpr)[candidates])]])


def _bootstrap_auc(frame: pd.DataFrame, column: str, seed: int,
                   repeats: int = 500) -> tuple[float, float]:
    """以 base_id 为簇有放回抽样，保留完整簇，以 AUROC 分位数估计 95% 置信区间。

    实验方案对应：
        S12 AUROC 不确定性统计；支持第十六节评价指标与论文 RQ1。

    算法/公式：
        本实现以原始问题 base_id 为簇 bootstrap，AUROC 95% CI=[Q_0.025,Q_0.975]；这是补充的统计估计，方案未单独规定 bootstrap 公式。

    输入：
        frame（pd.DataFrame）：测试 DataFrame，含 base_id、label 和分数列，簇含配对样本，重采样可产生足够的两类有效记录。
        column（str）：待评价的有限分数列名。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。
        repeats（int）：簇重采样次数，正整数。 默认值：500。

    输出：
        tuple[float,float]：有效 bootstrap AUROC 的 2.5% 和 97.5% 分位数；只有含两类的重采样参与统计，不写文件。
    """
    rng = random.Random(seed)
    ids = sorted(frame.base_id.unique())
    grouped = {key: frame[frame.base_id == key] for key in ids}
    values = []
    for _ in range(repeats):
        sample = [rng.choice(ids) for _ in ids]
        bootstrap = pd.concat([grouped[key] for key in sample], ignore_index=True)
        if bootstrap.label.nunique() == 2:
            values.append(roc_auc_score(bootstrap.label, bootstrap[column]))
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def select_test_ids(base_ids, seed: int, test_fraction: float) -> set:
    """对唯一 base_id 排序后按种子打乱，选择前 round(数量×test_fraction) 个测试 ID。

    实验方案对应：
        S12 测试集组织，供主比较、图表和错误分析共用；支持第八节同一 base sample 配对原则及第十六节评价。

    算法/公式：
        本实现按 base_id 分组留出，所有攻击/方法共享测试 ID，防止同一个问题同时进入阈值选择与测试；分组划分是实现选择。

    输入：
        base_ids：可排序的问题 ID 可迭代对象，数量足够形成非空训练/测试集。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。
        test_fraction（float）：独立 base_id 分入测试集的比例，在 (0,1) 内；各攻击划分后的训练/测试集均应包含两类标签。

    输出：
        set：测试 base_id 集合，同问题的各攻击、标签、方法共用相同划分；不修改输入。
    """
    ids = sorted(set(base_ids))
    random.Random(seed).shuffle(ids)
    return set(ids[:round(len(ids) * test_fraction)])


def evaluate_long_scores(frame: pd.DataFrame, seed: int = 20261004,
                         test_fraction: float = 0.3,
                         bootstrap_repeats: int = 500) -> list[dict]:
    """所有方法共用 base_id 分组留出；逐攻击及整体用训练集选阈值，用测试集计算 AUROC、簇 bootstrap 区间、Precision、Recall、F1。

    实验方案对应：
        S12 主指标计算；对应第十六节 AUROC/F1/Precision/Recall/runtime/query cost，第二十三节 RQ1/RQ4。

    算法/公式：
        测试预测=1[score≥训练阈值]；Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)。AUROC 为测试 ROC 曲线下面积；实测耗时/查询数按测试行求和，置信区间由簇 bootstrap 得到。

    输入：
        frame（pd.DataFrame）：方法长表，含 sample_id、base_id、attack、label、method、score、runtime_sec、query_count；各方法完整覆盖同样样本，score 有限，各攻击训练/测试均含两类，成本未知时留空。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。
        test_fraction（float）：独立 base_id 分入测试集的比例，在 (0,1) 内；各攻击划分后的训练/测试集均应包含两类标签。 默认值：0.3。
        bootstrap_repeats（int）：用于 AUROC 区间的簇重采样次数，正整数。 默认值：500。

    输出：
        list[dict]：每方法×(攻击种类数+1) 条指标，ALL 为整体；包含训练/测试数量、独立问题数、训练阈值、检测指标和测试生成耗时/查询数合计。成本列任一值为空则对应合计留空，不写文件。
    """
    # 1. 所有方法共用一次原始问题分组划分。
    test_ids = select_test_ids(frame.base_id.unique(), seed, test_fraction)
    frame = frame.copy()
    frame["split"] = np.where(frame.base_id.isin(test_ids), "test", "train")
    frame["score"] = frame.score.astype(float)
    rows = []
    for method, method_frame in frame.groupby("method", sort=True):
        groups = list(method_frame.groupby("attack", sort=True)) + [("ALL", method_frame)]
        for attack, group in groups:
            train = group[group.split == "train"]
            test = group[group.split == "test"]
            # 2. 训练集选阈值；测试集计算分类指标与簇置信区间。
            threshold = _threshold(train.label.to_numpy(), train.score.to_numpy())
            predictions = (test.score.to_numpy() >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test.label, predictions, average="binary", zero_division=0)
            lower, upper = _bootstrap_auc(test, "score", seed, bootstrap_repeats)
            # 3. 汇总实测生成成本；未知成本保持为空。
            runtime = pd.to_numeric(test.runtime_sec, errors="coerce")
            queries = pd.to_numeric(test.query_count, errors="coerce")
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


def evaluate_scores(scores_csv: str, output_csv: str, seed: int = 20261004,
                    test_fraction: float = 0.3) -> int:
    """把每样本 CPS/校正 CPS 两列转为方法长表，调用统一检测评价。

    实验方案对应：
        S12 的 CPS 与 CPS_cal 直接比较；对应第三节第 4 项、第十六节指标及第二十三节 RQ3。

    算法/公式：
        将 cps_score/cps_cal_score 转为两方法长表，调用统一指标评价；它负责比较输入组织与指标导出，不另写评价算法。

    输入：
        scores_csv（str）：CPS 汇总 CSV，含 sample_id、pair_id、base_id、attack、label、cps_score、cps_cal_score、runtime_sec、query_count。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        seed（int）：随机种子整数；相同数据和种子得到相同抽样或划分结果。 默认值：20261004。
        test_fraction（float）：独立 base_id 分入测试集的比例，在 (0,1) 内；各攻击划分后的训练/测试集均应包含两类标签。 默认值：0.3。

    输出：
        int：写出指标数为 2×(攻击种类数+1)，CSV 字段与 evaluate_long_scores 的返回记录一致。
    """
    frame = pd.read_csv(scores_csv, keep_default_na=False)
    metadata = ["sample_id", "pair_id", "base_id", "attack", "label",
                "runtime_sec", "query_count"]
    methods = []
    for method, column in (("CPS", "cps_score"), ("CPS-calibrated", "cps_cal_score")):
        current = frame[metadata + [column]].rename(columns={column: "score"})
        current["method"] = method
        methods.append(current)
    rows = evaluate_long_scores(pd.concat(methods, ignore_index=True), seed, test_fraction)
    write_rows(output_csv, rows, list(rows[0]))
    return len(rows)
