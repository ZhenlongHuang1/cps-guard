"""组装原R七项特征，训练来源参考或应用已有来源参考。"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .representation import CleanManifold


def assemble_features(output: Path, dimensions: list[int], seed: int,
                      reference_dir: Path | None = None,
                      reference_source: str | None = None,
                      fit_attacks: tuple[str, ...] = ("badnet", "vpi"),
                      sources: tuple[str, ...] = ("badnet", "vpi")) -> None:
    """合并原R五项统计及来源Mahalanobis均值/最大值，不再生成B/G。

    输入：output下的样本表、features_representation_raw.csv及池化向量；
        dimensions为PCA候选维数。reference_dir为空时仅用每个来源的Train clean
        拟合PCA/协方差；指定目录时只加载已有对象，reference_source指定单一来源，
        None表示两个来源。seed控制训练阶段PCA的随机化SVD；fit_attacks指定训练阶段
        要重新拟合的来源，其他来源直接加载output内原对象；sources可改为vpi_a/vpi_b。
    输出：None；results/features.csv与features_representation.csv；训练阶段另保存
        checkpoints/reference_{source}_{dimension}.joblib。向量通过representation_index
        对齐样本，目标Train/Validation/Test不会参与来源参考拟合。
    """
    results = output / "results"
    metadata = pd.read_csv(output / "data/attack_inputs.csv", keep_default_na=False)
    raw = pd.read_csv(results / "features_representation_raw.csv")
    columns = ["sample_id", "base_id", "split", "attack", "label", "input_type"]
    frame = metadata[columns].merge(raw.drop(columns=["base_id", "split", "attack", "label"]),
                                    on="sample_id", validate="one_to_one")
    vectors = np.load(results / "representation_vectors.npy", mmap_mode="r")
    for source in ([reference_source] if reference_source else sources):
        clean = frame[(frame.attack == source) & (frame.split == "train") & (frame.label == 0)]
        for dimension in dimensions:
            if reference_dir is None and source in fit_attacks:
                reference = CleanManifold(dimension, seed).fit(vectors[clean.representation_index.to_numpy()])
                joblib.dump(reference, output / f"checkpoints/reference_{source}_{dimension}.joblib")
            else:
                reference = joblib.load((reference_dir or output) / f"checkpoints/reference_{source}_{dimension}.joblib")
            values = reference.transform(vectors[frame.representation_index.to_numpy()])
            for index, suffix in enumerate(("mean", "max")):
                frame[f"mahalanobis_{source}_{dimension}_{suffix}"] = values[:, index]
    frame.to_csv(results / "features.csv", index=False)
    frame.to_csv(results / "features_representation.csv", index=False)
