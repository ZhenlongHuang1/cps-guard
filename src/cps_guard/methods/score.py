from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..data.schema import write_rows

SCORE_COLUMNS = [
    "sample_id", "pair_id", "base_id", "attack", "label", "method",
    "semantic_score", "context_score", "position_score", "cps_score",
    "randomness_baseline", "cps_cal_score", "runtime_sec", "query_count",
]

DETAIL_COLUMNS = [
    "sample_id", "attack", "label", "perturb_type", "perturb_id",
    "input_original", "input_perturbed", "response_original", "response_perturbed",
    "similarity", "distance", "logprob_diff", "entropy_diff",
    "randomness_baseline", "cps_score", "cps_cal_score", "runtime_sec",
]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个非零回答向量的余弦距离 1−cos(a,b)。

    输入：
        a（np.ndarray）：一维非零数值向量，形状 (d,)。
        b（np.ndarray）：与 a 同维度的一维非零数值向量，形状 (d,)。

    输出：
        float：范围 [0,2]，越大表示方向差异越大；浮点误差造成的越界截到边界。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(1.0 - np.dot(a, b) / denom, 0.0, 2.0))


def score_embeddings(frame: pd.DataFrame, embeddings: np.ndarray,
                     random_repeats: int = 5, lambda_randomness: float = 1.0) -> list[dict]:
    """从回答向量计算每条样本的 CPS 与校正分数。

    输入：
        frame（pd.DataFrame）：推理 DataFrame，含 INFERENCE_COLUMNS；每样本有 1 条 original、三类等数量扰动、至少 random_repeats 条 randomness，各编号从 1 连续递增。
        embeddings（np.ndarray）：形状 (frame行数,向量维度) 的非零数值数组，行顺序与 frame 一一对应。
        random_repeats（int）：原始输入的独立随机回答数，至少 2；按 perturb_id 取前这么多条计算两两距离。 默认值：5。
        lambda_randomness（float）：随机性扣除系数 λ；校正分数=原始分数−λ×随机性基线 B。 默认值：1.0。

    输出：
        list[dict]：按 sample_id 排序的每样本汇总，字段为 SCORE_COLUMNS；包含分量、CPS、B、CPS−λB、生成总耗时（秒）及查询数。不写文件。
    """
    scores, _ = score_embeddings_with_details(frame, embeddings, random_repeats,
                                               lambda_randomness)
    return scores


def score_embeddings_with_details(frame: pd.DataFrame, embeddings: np.ndarray,
                                  random_repeats: int = 5,
                                  lambda_randomness: float = 1.0) -> tuple[list[dict], list[dict]]:
    """计算扰动与原回答距离，每类取均值再对三类平均得到 CPS；随机回答两两距离均值为 B，校正分数为 CPS−λB。

    输入：
        frame（pd.DataFrame）：推理 DataFrame，含 INFERENCE_COLUMNS；每样本有 1 条 original、三类等数量扰动、至少 random_repeats 条 randomness，各编号从 1 连续递增。
        embeddings（np.ndarray）：形状 (frame行数,向量维度) 的非零数值数组，行顺序与 frame 一一对应。
        random_repeats（int）：原始输入的独立随机回答数，至少 2；按 perturb_id 取前这么多条计算两两距离。 默认值：5。
        lambda_randomness（float）：随机性扣除系数 λ；校正分数=原始分数−λ×随机性基线 B。 默认值：1.0。

    输出：
        tuple[list[dict],list[dict]]：第一项每样本一行 SCORE_COLUMNS；第二项每次非随机扰动一行 DETAIL_COLUMNS，含输入、回答、distance、similarity=1−distance、生成耗时。未测量的 logprob_diff/entropy_diff 留空，不写文件。
    """
    # 1. 保存推理行到向量行的对应关系。
    frame = frame.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    rows: list[dict] = []
    details: list[dict] = []
    for sid, group in frame.groupby("sample_id", sort=True):
        original = group[group.perturb_type == "original"]
        base = original.iloc[0]
        base_embedding = embeddings[int(base["row_index"])]
        part_scores = {}
        sample_details = []
        # 2. 与原始回答比较，先取各类扰动的平均距离。
        for kind in ("semantic", "context", "position"):
            variants = group[group.perturb_type == kind]
            distances = []
            for variant in variants.itertuples(index=False):
                distance = cosine_distance(base_embedding, embeddings[int(variant.row_index)])
                distances.append(distance)
                sample_details.append({
                    "sample_id": sid, "attack": base.attack, "label": int(base.label),
                    "perturb_type": kind, "perturb_id": int(variant.perturb_id),
                    "input_original": base["perturbed_text"],
                    "input_perturbed": variant.perturbed_text,
                    "response_original": base["model_response"],
                    "response_perturbed": variant.model_response,
                    "similarity": 1.0 - distance, "distance": distance,
                    "logprob_diff": "", "entropy_diff": "",
                    "runtime_sec": float(variant.runtime_sec),
                })
            part_scores[kind] = float(np.mean(distances))
        # 3. 用随机回答两两距离估计 B，再计算 CPS 和 CPS−λB。
        random = group[group.perturb_type == "randomness"].sort_values("perturb_id").head(random_repeats)
        random_ids = [int(value) for value in random.row_index]
        baseline = float(np.mean([
            cosine_distance(embeddings[i], embeddings[j])
            for i, j in combinations(random_ids, 2)
        ]))
        cps = float(np.mean(list(part_scores.values())))
        for detail in sample_details:
            detail.update({"randomness_baseline": baseline, "cps_score": cps,
                           "cps_cal_score": cps - lambda_randomness * baseline})
        # 4. 收集逐扰动明细及每样本分数、生成成本。
        details.extend(sample_details)
        rows.append({
            "sample_id": sid, "pair_id": base.pair_id, "base_id": base.base_id,
            "attack": base.attack, "label": int(base.label), "method": "CPS-Guard",
            "semantic_score": part_scores["semantic"],
            "context_score": part_scores["context"],
            "position_score": part_scores["position"], "cps_score": cps,
            "randomness_baseline": baseline,
            "cps_cal_score": cps - lambda_randomness * baseline,
            "runtime_sec": float(pd.to_numeric(group.runtime_sec).sum()),
            "query_count": len(group),
        })
    return rows, details


def score_inference(input_csv: str, output_csv: str, embedding_model: str,
                    random_repeats: int = 5,
                    lambda_randomness: float = 1.0,
                    details_csv: str | None = None) -> int:
    """将推理回答编码为语义向量，计算 CPS 汇总与可选逐扰动明细。

    输入：
        input_csv（str）：INFERENCE_COLUMNS 格式的推理结果，每样本包含原始、三类扰动和随机回答。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        embedding_model（str）：SentenceTransformer 模型名称或本地目录，用于将回答编码为语义向量。
        random_repeats（int）：原始输入的独立随机回答数，至少 2；按 perturb_id 取前这么多条计算两两距离。 默认值：5。
        lambda_randomness（float）：随机性扣除系数 λ；校正分数=原始分数−λ×随机性基线 B。 默认值：1.0。
        details_csv（str | None）：可选明细保存路径；None 只输出汇总，提供路径时覆盖写出输入、回答及距离明细。 默认值：None。

    输出：
        int：写出汇总样本数；output_csv 字段为 SCORE_COLUMNS，可选明细为 DETAIL_COLUMNS。runtime_sec 仅统计受害模型生成时间，不含向量编码或分数计算。
    """
    from sentence_transformers import SentenceTransformer

    frame = pd.read_csv(input_csv, keep_default_na=False)
    encoder = SentenceTransformer(embedding_model)
    embeddings = encoder.encode(frame.model_response.astype(str).tolist(),
                                batch_size=64, convert_to_numpy=True,
                                show_progress_bar=True)
    rows, details = score_embeddings_with_details(frame, embeddings, random_repeats,
                                                   lambda_randomness)
    write_rows(output_csv, rows, SCORE_COLUMNS)
    if details_csv:
        write_rows(details_csv, details, DETAIL_COLUMNS)
    return len(rows)
