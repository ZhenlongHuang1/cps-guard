from __future__ import annotations

from itertools import combinations
from pathlib import Path

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
    "similarity", "distance",
    "randomness_baseline", "cps_score", "cps_cal_score", "runtime_sec",
]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个非零回答向量的余弦距离 1−cos(a,b)。

    实验方案对应：
        S7 的回答差异 D；对应第三节第 2 项“基础敏感性”，方案指定可用输出 embedding 的 cosine distance。

    算法/公式：
        设 a=φ(y)、b=φ(y′)，φ 为回答编码器：D(y,y′)=1−(a·b)/(||a||₂||b||₂)。本函数接收已经编码的 a/b，编码由 score_inference 完成。

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


def score_embeddings_with_details(frame: pd.DataFrame, embeddings: np.ndarray,
                                  random_repeats: int = 5,
                                  lambda_randomness: float = 1.0) -> tuple[list[dict], list[dict]]:
    """计算扰动与原回答距离，每类取均值再对三类平均得到 CPS；随机回答两两距离均值为 B，校正分数为 CPS−λB。

    实验方案对应：
        S7 CPS score；对应第三节第 2～4 项“基础敏感性/CPS 分数/随机性校正”，并保存第十四节要求的中间结果。

    算法/公式：
        记 x 为原始输入、f 为受害模型、T_k,j 为第 k 类第 j 次扰动，y_r 为同一 x 的第 r 次随机回答。
        方案基础式：S_k(x)=D(f(x),f(T_k(x)))。
        本实现对每类 N_k 个变体取均值：S_k(x)=(1/N_k)Σ_j D(f(x),f(T_k,j(x)))。
        CPS(x)=(S_semantic(x)+S_context(x)+S_position(x))/3。
        B(x)=[2/(R(R−1))]Σ_{1≤r<s≤R}D(y_r,y_s)，R=random_repeats。
        CPS_cal(x)=CPS(x)−λB(x)，λ=lambda_randomness；允许校正结果为负。
        distance 对应 D，三类 *_score 对应 S_k，randomness_baseline 对应 B；逐扰动保留输入、回答和距离。

    输入：
        frame（pd.DataFrame）：推理 DataFrame，含 INFERENCE_COLUMNS；每样本有 1 条 original、三类等数量扰动、至少 random_repeats 条 randomness，original 编号为 0，扰动与随机回答各编号从 1 连续递增。
        embeddings（np.ndarray）：形状 (frame行数,向量维度) 的非零数值数组，行顺序与 frame 一一对应。
        random_repeats（int）：原始输入的独立随机回答数，至少 2；按 perturb_id 取前这么多条计算两两距离。 默认值：5。
        lambda_randomness（float）：随机性扣除系数 λ；校正分数=原始分数−λ×随机性基线 B。 默认值：1.0。

    输出：
        tuple[list[dict],list[dict]]：第一项每样本一行 SCORE_COLUMNS；第二项每次非随机扰动一行 DETAIL_COLUMNS，含输入、回答、distance、similarity=1−distance、生成耗时。不写文件。
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
        # 2. 方案三.2：计算 D(f(x),f(T_k,j(x)))，取均值得到 S_k(x)。
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
                    "runtime_sec": float(variant.runtime_sec),
                })
            part_scores[kind] = float(np.mean(distances))
        # 3. 方案三.3～三.4：计算 B、三类均值 CPS，以及 CPS_cal=CPS−λB。
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
                    details_csv: str | None = None,
                    embeddings_npy: str | None = None) -> int:
    """将推理回答编码为语义向量，计算 CPS 汇总与可选逐扰动明细。

    实验方案对应：
        S7 从推理文本到计分结果；对应第三节 embedding 距离及 CPS 计算、第十四节中间结果规范。

    算法/公式：
        先以 embedding_model 编码 φ(y)，再调用唯一计分函数 score_embeddings_with_details 计算 D、S_k、CPS、B、CPS_cal；最后写出汇总和可选明细。

    输入：
        input_csv（str）：INFERENCE_COLUMNS 格式的推理结果，每样本包含原始、三类扰动和随机回答。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        embedding_model（str）：SentenceTransformer 模型名称或本地目录，用于将回答编码为语义向量。
        random_repeats（int）：原始输入的独立随机回答数，至少 2；按 perturb_id 取前这么多条计算两两距离。 默认值：5。
        lambda_randomness（float）：随机性扣除系数 λ；校正分数=原始分数−λ×随机性基线 B。 默认值：1.0。
        details_csv（str | None）：可选明细保存路径；None 只输出汇总，提供路径时覆盖写出输入、回答及距离明细。 默认值：None。
        embeddings_npy（str | None）：同批回答的向量缓存路径；已存在时直接读取，
            否则编码后保存。None 不缓存；缓存行顺序必须与 input_csv 一致。
            main 按编码器名称区分缓存，STEP=2 更新回答时清除旧缓存；手工替换
            inference.csv 或同名编码器权重后，应删除对应缓存再评分。

    输出：
        int：写出汇总样本数；output_csv 字段为 SCORE_COLUMNS，可选明细为 DETAIL_COLUMNS。runtime_sec 仅统计受害模型生成时间，不含向量编码或分数计算。
    """
    frame = pd.read_csv(input_csv, keep_default_na=False)
    cache = Path(embeddings_npy) if embeddings_npy is not None else None
    if cache is not None and cache.exists():
        print(f"复用回答向量：{cache}")
        embeddings = np.load(cache)
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedding_model)
        embeddings = encoder.encode(frame.model_response.astype(str).tolist(),
                                    batch_size=64, convert_to_numpy=True,
                                    show_progress_bar=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, embeddings)
    rows, details = score_embeddings_with_details(frame, embeddings, random_repeats,
                                                   lambda_randomness)
    write_rows(output_csv, rows, SCORE_COLUMNS)
    if details_csv:
        write_rows(details_csv, details, DETAIL_COLUMNS)
    return len(rows)
