"""分组留出检测评价与 AUROC 置信区间。"""
from __future__ import annotations
import random
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from ..data.schema import write_rows


def _threshold(y: np.ndarray, score: np.ndarray) -> float:
    """仅在训练集上用 Youden J 选择检测阈值。"""
    fpr, tpr, thresholds = roc_curve(y, score)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite threshold")
    candidates = np.where(finite)[0]
    return float(thresholds[candidates[np.argmax((tpr - fpr)[candidates])]])


def _bootstrap_auc(frame: pd.DataFrame, column: str, seed: int,
                   repeats: int = 500) -> tuple[float, float]:
    """按独立 base_id 重采样，估计 AUROC 的 95% 置信区间。"""
    rng = random.Random(seed)
    ids = sorted(frame.base_id.unique())
    grouped = {key: frame[frame.base_id == key] for key in ids}
    values = []
    for _ in range(repeats):
        sample = [rng.choice(ids) for _ in ids]
        bootstrap = pd.concat([grouped[key] for key in sample], ignore_index=True)
        if bootstrap.label.nunique() == 2:
            values.append(roc_auc_score(bootstrap.label, bootstrap[column]))
    if len(values) < repeats // 2:
        raise ValueError("Too few valid bootstrap samples")
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def evaluate_scores(scores_csv: str, output_csv: str, seed: int = 20261004,
                    test_fraction: float = 0.3) -> int:
    """按原始问题分组留出测试集，评估 CPS 和校正版本的检测指标。"""
    frame = pd.read_csv(scores_csv, keep_default_na=False)
    needed = {"sample_id", "pair_id", "base_id", "attack", "label",
              "cps_score", "cps_cal_score"}
    if needed - set(frame.columns) or frame.sample_id.duplicated().any():
        raise ValueError(f"Score table needs unique IDs and columns {sorted(needed)}")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie between 0 and 1")
    base_ids = sorted(frame.base_id.unique())
    if len(base_ids) < 10:
        raise ValueError("Need at least 10 independent base prompts")
    rng = random.Random(seed)
    rng.shuffle(base_ids)
    test_ids = set(base_ids[:max(1, round(len(base_ids) * test_fraction))])
    frame["split"] = np.where(frame.base_id.isin(test_ids), "test", "train")
    rows: list[dict] = []
    for attack, whole in list(frame.groupby("attack", sort=True)) + [("ALL", frame)]:
        train = whole[whole.split == "train"]
        test = whole[whole.split == "test"]
        if train.label.nunique() != 2 or test.label.nunique() != 2:
            raise ValueError(f"{attack}: both splits need both classes")
        for method, column in (("CPS", "cps_score"),
                               ("CPS-calibrated", "cps_cal_score")):
            threshold = _threshold(train.label.to_numpy(), train[column].to_numpy())
            predicted = (test[column].to_numpy() >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test.label, predicted, average="binary", zero_division=0,
            )
            lower, upper = _bootstrap_auc(test, column, seed)
            rows.append({
                "method": method, "attack": attack,
                "n_train": len(train), "n_test": len(test),
                "n_independent_test_prompts": test.base_id.nunique(),
                "AUROC": float(roc_auc_score(test.label, test[column])),
                "AUROC_CI_low": lower, "AUROC_CI_high": upper,
                "threshold_train": threshold,
                "Precision": float(precision), "Recall": float(recall),
                "F1": float(f1),
                "runtime_sec_total_test": float(test.runtime_sec.sum()) if "runtime_sec" in test else "",
                "query_count_total_test": int(test.query_count.sum()) if "query_count" in test else "",
            })
    columns = list(rows[0])
    write_rows(output_csv, rows, columns)
    return len(rows)
