"""Pilot-v2 单/多视图 Logistic Regression：来源验证选参、冻结、正式测试。"""
import hashlib
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
                     seed: int, iid_gate: float) -> None:
    """对应 STEP 10–15：来源 Train 拟合 scaler/LR，来源 Validation 选 C/PCA/阈值。

    输入：v2 根目录、事先指定 C/PCA 网格、检测器种子、IID 验证 AUROC 门槛。
    输出：None；14个检测器 joblib、validation_results.csv、frozen_detector_config.json。
        不读取 Test 标签进行评价。选参以 Validation AUROC 最大为准，同分取网格首项；
        不重新拟合 Train+Validation。Gate 2 用来源 Validation，而非提前查看正式 Test。
        已完成正式测试的目录不得重新选参，必须另建实验。
    """
    if (output / "results/test_predictions.csv").exists():
        raise ValueError("本轮 Test 已执行，不能重选参数；请建立新的独立实验。")
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
                           "checkpoint": name,
                           "checkpoint_sha256": hashlib.sha256((output / 'checkpoints' / name).read_bytes()).hexdigest()})
            print(f"冻结：{source} {view} Validation AUROC={auroc:.4f}", flush=True)
    # 2. 规则冻结在测试之前；任一来源全部接近随机时不开放正式跨攻击测试。
    gate = all(max(item["validation_AUROC"] for item in frozen if item["source_attack"] == source)
               >= iid_gate for source in ("badnet", "vpi"))
    snapshot = {"seed": seed, "selection": "source_validation_AUROC_then_Youden",
                "iid_validation_gate": iid_gate, "gate2_passed": gate, "detectors": frozen,
                "features_sha256": hashlib.sha256((output / 'results/features.csv').read_bytes()).hexdigest()}
    pd.DataFrame(records).to_csv(output / "results/validation_results.csv", index=False)
    (output / "configs/frozen_detector_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gate 2（来源 Validation）：{'通过' if gate else '未通过，停止正式测试'}")


def evaluate_frozen(output: Path) -> None:
    """对应 STEP 13–15：一次执行14冻结检测器×两目标攻击，保存400题中Test预测。

    输入：v2 根目录；须已有冻结规则、特征和检测器，Gate 2 已通过。
    输出：None；test_predictions.csv、iid_results.csv、cross_attack_results.csv。
        每模型每目标200样本（100问题）；重复运行复用保存预测，不重新调用检测器。
        对 frozen 配置和 checkpoint 校验绑定，避免换参数后把旧预测当成新实验。
    """
    config_file = output / "configs/frozen_detector_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not config["gate2_passed"]:
        raise ValueError("Gate 2 未通过；先分析 Validation，不执行正式 Test。")
    results = output / "results"
    predictions_file = results / "test_predictions.csv"
    binding = hashlib.sha256(config_file.read_bytes()).hexdigest()
    if predictions_file.exists():
        predictions = pd.read_csv(predictions_file)
        if not predictions.frozen_config_sha256.eq(binding).all():
            raise ValueError("已有 Test 预测与冻结配置不一致，不能覆盖或重测。")
        print("复用已保存的正式 Test 预测。")
    else:
        features_file = results / "features.csv"
        if hashlib.sha256(features_file.read_bytes()).hexdigest() != config["features_sha256"]:
            raise ValueError("特征在冻结后改变，不能继续本轮 Test。")
        frame = pd.read_csv(features_file)
        rows = []
        for item in config["detectors"]:
            checkpoint = output / "checkpoints" / item["checkpoint"]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != item["checkpoint_sha256"]:
                raise ValueError("检测器在冻结后改变。")
            detector = joblib.load(checkpoint)
            for target in ("badnet", "vpi"):
                test = frame[(frame.attack == target) & (frame.split == "test")]
                scores = detector.predict_proba(test[item["columns"]])[:, 1]
                for record, score in zip(test.to_dict("records"), scores):
                    rows.append({"sample_id": record["sample_id"], "base_id": record["base_id"],
                                 "source_attack": item["source_attack"], "target_attack": target,
                                 "view": item["view"], "label": record["label"], "score": float(score),
                                 "threshold": item["threshold"], "frozen_config_sha256": binding})
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
