"""最小 Pilot 的分组留出检测评价。"""
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


def select_test_ids(base_ids, seed: int, test_fraction: float) -> set:
    """对唯一 base_id 排序后按种子打乱，选择前 round(数量×test_fraction) 个测试 ID。

    实验方案对应：
        S12 测试集组织，供主比较和图表共用；支持第八节同一 base sample 配对原则及第十六节评价。

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


def evaluate_scores(scores_csv: str, random_csv: str, output_csv: str,
                    seed: int = 20261004) -> int:
    """在同一测试集比较 CPS、校正 CPS 和 Random，逐攻击及整体导出指标。

    实验方案对应：
        S11～S12，第三节第 4 项校正前后比较，第十五节 Random，
        第十六节 AUROC、Precision、Recall、F1 和生成成本。

    算法/公式：
        按 base_id 留出 30% 原始问题，同一问题的两种攻击、clean/poison 共用划分。
        训练集选择最大 Youden J=TPR−FPR 的阈值 t，测试预测为 1[score≥t]。
        测试 AUROC 为 ROC 面积；Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，
        F1=2PR/(P+R)。分组比例及 Youden 阈值是本实现选择。

    输入：
        scores_csv（str）：每样本 CPS 汇总，含 sample_id、base_id、attack、label、
            cps_score、cps_cal_score、runtime_sec、query_count；各攻击均含两类标签。
        random_csv（str）：同样样本的 Random 分数表，含 sample_id 和 score。
        output_csv（str）：指标 CSV 路径，创建父目录并覆盖同名文件。
        seed（int）：分组种子，默认 20261004；与 plot_results 使用相同种子。

    输出：
        int：写出指标行数，两种攻击时为 9（3 方法×badnet/vpi/ALL）。
        指标包含阈值、训练/测试数量、独立测试问题数、AUROC、Precision、Recall、
        F1、测试回答生成耗时与查询数。Random 不查询模型，生成成本记为 0；
        CPS 成本不含语义改写准备、向量编码和评分计算。
    """
    # 1. 按 sample_id 对齐 Random，所有方法共用一次原始问题划分。
    scores = pd.read_csv(scores_csv)
    baseline = pd.read_csv(random_csv)[["sample_id", "score"]]
    frame = scores.merge(baseline.rename(columns={"score": "random_score"}), on="sample_id")
    test_ids = select_test_ids(frame.base_id, seed, 0.3)
    frame["split"] = np.where(frame.base_id.isin(test_ids), "test", "train")
    rows = []
    for method, column in (("CPS", "cps_score"), ("CPS-calibrated", "cps_cal_score"),
                           ("Random", "random_score")):
        for attack, group in list(frame.groupby("attack", sort=True)) + [("ALL", frame)]:
            train = group[group.split == "train"]
            test = group[group.split == "test"]
            # 2. 只用训练集选阈值，在测试集计算指标。
            threshold = _threshold(train.label.to_numpy(), train[column].to_numpy())
            predictions = (test[column].to_numpy() >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test.label, predictions, average="binary", zero_division=0)
            rows.append({
                "method": method, "attack": attack, "n_train": len(train),
                "n_test": len(test), "n_independent_test_prompts": test.base_id.nunique(),
                "AUROC": float(roc_auc_score(test.label, test[column])),
                "threshold_train": threshold, "Precision": float(precision),
                "Recall": float(recall), "F1": float(f1),
                "runtime_sec_total_test": 0.0 if method == "Random" else float(test.runtime_sec.sum()),
                "query_count_total_test": 0 if method == "Random" else int(test.query_count.sum()),
            })
    write_rows(output_csv, rows, list(rows[0]))
    return len(rows)
