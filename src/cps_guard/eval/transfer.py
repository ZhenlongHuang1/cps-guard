"""Pilot-v2 单/多视图 Logistic Regression：来源验证选参、冻结、正式测试。"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..methods.features import BEHAVIORAL_COLUMNS
from ..model.features import GENERATION_COLUMNS, REPRESENTATION_COLUMNS

VIEWS = ("B", "G", "R", "B+G", "B+R", "G+R", "B+G+R")


def feature_columns(view: str, source: str, dimension: int) -> list[str]:
    """对应 STEP 11/12/14：返回七种视图的固定列，不通过 Test 选择特征。

    输入：view 为 B/G/R 的组合、source 为来源攻击、dimension 为参考 PCA 维度。
    输出：列名列表；R 仅包括原始五项和来源攻击 reference 的两项 Mahalanobis。
    """
    columns = []
    for part in view.split("+"):
        columns.extend({"B": BEHAVIORAL_COLUMNS, "G": GENERATION_COLUMNS,
                        "R": [*REPRESENTATION_COLUMNS, f"mahalanobis_{source}_{dimension}_mean",
                              f"mahalanobis_{source}_{dimension}_max"]}[part])
    return columns


def validation_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """对应 STEP 15：仅在来源 Validation 选使 Youden J 最大的有限概率阈值。
    输入：0/1 标签与 poison 概率。输出：float；同分时采用 ROC 返回的首个阈值。
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax((tpr - fpr)[valid])])


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """对应 STEP 11–17：对给定预测计算 AUROC/AUPRC/F1 和低误报 ROC 指标。

    输入：0/1标签、连续 poison 概率、已在 Validation 冻结的阈值。
    输出：dict；AUPRC 使用 average precision（阶梯积分），TPR@1%FPR 为经验 ROC
        中 FPR≤0.01 的最大 TPR，不作插值。F1 等分类指标使用冻结阈值；不调参。
    """
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(labels, scores)
    return {"AUROC": roc_auc_score(labels, scores), "AUPRC": average_precision_score(labels, scores),
            "F1": f1_score(labels, prediction, zero_division=0),
            "TPR_at_1pct_FPR": float(np.max(tpr[fpr <= 0.01])),
            "Accuracy": (tp + tn) / len(labels), "FPR": fp / (fp + tn),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "threshold": threshold}


def freeze_detectors(output: Path, c_values: list[float], dimensions: list[int],
                     seed: int) -> None:
    """对应 STEP 10–15：来源 Train 拟合 scaler/LR，来源 Validation 选 C/PCA/阈值。

    输入：v2根目录、C/PCA候选网格和种子；特征仅包含Train/Validation。
    输出：None；14个检测器 joblib、validation_results.csv、frozen_detector_config.json。
        不读取 Test 标签进行评价。选参以 Validation AUROC 最大为准，同分取网格首项；
        不重新拟合Train+Validation；参数冻结后在第四步处理独立Test。
    """
    frame = pd.read_csv(output / "results/features.csv")
    records, frozen = [], []
    # 1. 只切来源攻击 Train/Validation，另一攻击不参与 scaler/LR 或选参。
    for source in ("badnet", "vpi"):
        train = frame[(frame.attack == source) & (frame.split == "train")]
        validation = frame[(frame.attack == source) & (frame.split == "validation")]
        for view in VIEWS:
            best = None
            for dimension in dimensions if "R" in view else [dimensions[0]]:
                columns = feature_columns(view, source, dimension)
                for c in c_values:
                    detector = make_pipeline(StandardScaler(), LogisticRegression(
                        C=c, solver="lbfgs", max_iter=2000, random_state=seed))
                    detector.fit(train[columns], train.label)
                    scores = detector.predict_proba(validation[columns])[:, 1]
                    auroc = float(roc_auc_score(validation.label, scores))
                    records.append({"source_attack": source, "view": view, "C": c,
                                    "pca_dimension": dimension if "R" in view else None,
                                    "validation_AUROC": auroc})
                    if best is None or auroc > best[0]:
                        best = (auroc, detector, columns, c, dimension, scores)
            auroc, detector, columns, c, dimension, scores = best
            threshold = validation_threshold(validation.label.to_numpy(), scores)
            name = f"detector_{source}_{view.replace('+', '')}.joblib"
            joblib.dump(detector, output / "checkpoints" / name)
            frozen.append({"source_attack": source, "view": view, "columns": columns, "C": c,
                           "pca_dimension": dimension if "R" in view else None,
                           "threshold": threshold, "validation_AUROC": auroc,
                           "checkpoint": name})
            print(f"冻结：{source} {view} Validation AUROC={auroc:.4f}", flush=True)
    snapshot = {"seed": seed, "selection": "source_validation_AUROC_then_Youden", "detectors": frozen}
    pd.DataFrame(records).to_csv(output / "results/validation_results.csv", index=False)
    (output / "configs/frozen_detector_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    # 重训检测器时使旧测试预测失效，后续只按新检测器评分。
    (output / "results/test_predictions.csv").unlink(missing_ok=True)


def evaluate_frozen(output: Path, test_output: Path | None = None) -> None:
    """对独立Test特征应用两来源的14个固定检测器，统计同攻击和双向迁移。

    输入：output提供检测器和冻结配置；test_output提供第四步提取的Test特征，
        缺省读取output下的特征，便于离线评价已有特征表。
    输出：None；根results下test_predictions.csv、iid_results.csv和cross_attack_results.csv。
        来源scaler/LR/PCA/阈值不更新；已有预测可直接用于重复统计和绘图。
    """
    config = json.loads((output / "configs/frozen_detector_config.json").read_text(encoding="utf-8"))
    results = output / "results"
    predictions_file = results / "test_predictions.csv"
    if predictions_file.exists():
        predictions = pd.read_csv(predictions_file)
        print("复用已保存的Test预测。", flush=True)
    else:
        frame = pd.read_csv((test_output or output) / "results/features.csv")
        rows = []
        for item in config["detectors"]:
            detector = joblib.load(output / "checkpoints" / item["checkpoint"])
            for target in ("badnet", "vpi"):
                test = frame[(frame.attack == target) & (frame.split == "test")]
                scores = detector.predict_proba(test[item["columns"]])[:, 1]
                for record, score in zip(test.to_dict("records"), scores):
                    rows.append({"sample_id": record["sample_id"], "base_id": record["base_id"],
                                 "source_attack": item["source_attack"], "target_attack": target,
                                 "view": item["view"], "label": record["label"], "score": float(score),
                                 "threshold": item["threshold"]})
        predictions = pd.DataFrame(rows)
        predictions.to_csv(predictions_file, index=False)
    metrics = []
    for (source, target, view), group in predictions.groupby(["source_attack", "target_attack", "view"]):
        metrics.append({"source_attack": source, "target_attack": target, "view": view,
                        "n_samples": len(group), "n_base_questions": group.base_id.nunique(),
                        **detection_metrics(group.label.to_numpy(), group.score.to_numpy(), group.threshold.iloc[0])})
    metrics = pd.DataFrame(metrics)
    metrics[metrics.source_attack == metrics.target_attack].to_csv(results / "iid_results.csv", index=False)
    metrics[metrics.source_attack != metrics.target_attack].to_csv(results / "cross_attack_results.csv", index=False)


def evaluate_control(output: Path, source_dir: Path, source_attack: str, test_model: str,
                     repeats: int, bootstrap_seed: int) -> pd.DataFrame:
    """第四步模型对照：用已有七视图检测器评价指定模型和输入，不训练或校准。

    实验方案对应：固定来源检测器，比较原始 Qwen 与攻击 LoRA 对同一组输入的得分。
    算法/规则：
        加载 source_attack 的 scaler/LR/阈值，分别对 poison vs clean、poison vs
        neutral 调用原 detection_metrics；每次仅包含一类负样本，因此可直接复用
        grouped_auc 的按 base_id 配对 bootstrap。两个比较和全部视图共享抽样权重。
    输入：
        output：本次目录（含 features.csv）；source_dir：原 Pilot-v2 冻结对象目录；
        source_attack：badnet/vpi 检测器来源；test_model：实际被测模型名称；
        repeats/bootstrap_seed：沿用原 bootstrap 次数与种子。
    输出：
        返回原指标汇总 DataFrame；保存 test_predictions.csv、control_results.csv、
        bootstrap_draws.npz 和 final_report.md。label=1 始终表示构造的 poison，
        不表示模型已产生攻击目标回答；实际命中率另见 asr.csv。
    """
    from .statistics import grouped_auc

    frame = pd.read_csv(output / "results/features.csv", keep_default_na=False)
    frozen = json.loads((source_dir / "configs/frozen_detector_config.json").read_text(encoding="utf-8"))
    rows = []
    for item in frozen["detectors"]:
        if item["source_attack"] != source_attack:
            continue
        detector = joblib.load(source_dir / "checkpoints" / item["checkpoint"])
        scores = detector.predict_proba(frame[item["columns"]])[:, 1]
        for sample, score in zip(frame.to_dict("records"), scores):
            rows.append({"sample_id": sample["sample_id"], "base_id": sample["base_id"],
                         "source_attack": source_attack, "target_attack": sample["attack"],
                         "test_model": test_model, "input_type": sample["input_type"],
                         "view": item["view"], "label": sample["label"],
                         "score": float(score), "threshold": item["threshold"]})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "results/test_predictions.csv", index=False)
    base_ids = sorted(frame.base_id.unique())
    weights = np.random.default_rng(bootstrap_seed).multinomial(
        len(base_ids), np.full(len(base_ids), 1 / len(base_ids)), size=repeats)
    metrics, draws = [], {}
    for negative in (kind for kind in ("clean", "neutral") if kind in frame.input_type.values):
        selected = predictions[predictions.input_type.isin([negative, "poison"])]
        for view, group in selected.groupby("view", sort=False):
            values = grouped_auc(group, weights, base_ids)
            low, high = np.quantile(values, [0.025, 0.975])
            comparison = f"poison_vs_{negative}"
            draws[f"{comparison}__{view}"] = values
            metrics.append({"source_attack": source_attack, "input_attack": frame.attack.iloc[0],
                            "test_model": test_model, "comparison": comparison, "view": view,
                            "n_samples": len(group), "n_base_questions": len(base_ids),
                            **detection_metrics(group.label.to_numpy(), group.score.to_numpy(), group.threshold.iloc[0]),
                            "CI_low": low, "CI_high": high})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "results/control_results.csv", index=False)
    np.savez_compressed(output / "results/bootstrap_draws.npz", weights=weights,
                        base_ids=np.array(base_ids), **draws)
    (output / "results/final_report.md").write_text(
        f"# Pilot-v2 补充测试\n\n被测模型：{test_model}；输入攻击：{frame.attack.iloc[0]}；"
        f"冻结检测器来源：{source_attack}。\n\n```text\n{metrics.to_string(index=False)}\n```\n\n"
        "原始 Qwen 上的 AUROC 表示触发文本区分能力；不能直接解释为后门激活检测。\n"
        "使用本轮独立Test问题；检测器及阈值固定，不在目标数据上调参。\n"
        "各输入的攻击目标命中率见 asr.csv。\n", encoding="utf-8")
    return metrics
