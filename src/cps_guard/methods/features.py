"""Pilot-v2 特征表与来源攻击独立的 clean reference。"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .representation import CleanManifold
from .score import score_inference

BEHAVIORAL_COLUMNS = ["semantic_score", "context_score", "position_score", "cps_score",
                      "randomness_baseline", "cps_cal_score"]


def assemble_features(output: Path, encoder_path: str, random_repeats: int,
                      lambda_randomness: float, dimensions: list[int], seed: int,
                      provenance: dict) -> None:
    """对应 STEP 5/8/9：计算 B 六项、来源 clean manifold，按 sample_id 合并三视图。

    输入：v2 根目录、回答编码器、随机回答数/λ、PCA 候选维度、随机种子、模型/版本元数据。
    输出：None；features_behavioral/representation/features.csv 和 reference joblib。
        B/G/R 合并为一对一关系。每个来源/维度的 reference 仅拟合该攻击 Train clean；
        Test 在这里仅执行固定变换，不计算检测指标、不参与选参。特征列显式标注来源，
        检测器仅选择自身来源的 Mahalanobis 列，不能读另一攻击的参考列。
    """
    results = output / "results"
    score_inference(results / "inference.csv", results / "features_behavioral.csv",
                    encoder_path, random_repeats, lambda_randomness,
                    results / "perturbation_details.csv", results / "response_embeddings.npy")
    metadata = pd.read_csv(output / "data/attack_inputs.csv", keep_default_na=False)
    frame = metadata[["sample_id", "base_id", "split", "attack", "label"]]
    for name, columns in (("features_behavioral", BEHAVIORAL_COLUMNS),
                          ("features_generation", None), ("features_representation_raw", None)):
        part = pd.read_csv(results / f"{name}.csv")
        cols = columns or [c for c in part if c not in ("sample_id", "base_id", "split", "attack", "label")]
        frame = frame.merge(part[["sample_id", *cols]], on="sample_id", validate="one_to_one")
    if len(frame) != len(metadata):
        raise ValueError("B/G/R未覆盖全部样本，不能在缺失样本的子集上继续评价。")
    vectors = np.load(results / "representation_vectors.npy", mmap_mode="r")
    representation = frame[["sample_id", "base_id", "split", "attack", "label"]].copy()
    # 来源攻击分开拟合：跨攻击迁移时不使用目标攻击的 Train clean。
    for source in ("badnet", "vpi"):
        clean = frame[(frame.attack == source) & (frame.split == "train") & (frame.label == 0)]
        for dimension in dimensions:
            reference = CleanManifold(dimension, seed).fit(vectors[clean.representation_index.to_numpy()])
            joblib.dump(reference, output / f"checkpoints/reference_{source}_{dimension}.joblib")
            values = reference.transform(vectors[frame.representation_index.to_numpy()])
            for index, suffix in enumerate(("mean", "max")):
                name = f"mahalanobis_{source}_{dimension}_{suffix}"
                frame[name] = values[:, index]
                representation[name] = values[:, index]
    for name, value in provenance.items():
        frame[name] = frame.attack.map(value) if name == "adapter" else value
    frame.to_csv(results / "features.csv", index=False)
    representation.to_csv(results / "features_representation.csv", index=False)
