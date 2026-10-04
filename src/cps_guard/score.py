from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .io import write_rows

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
    """计算两条非零回答向量的余弦距离，数值越大表示回答差异越大。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        raise ValueError("Zero-norm response embedding")
    return float(np.clip(1.0 - np.dot(a, b) / denom, 0.0, 2.0))


def score_embeddings(frame: pd.DataFrame, embeddings: np.ndarray,
                     random_repeats: int = 5, lambda_randomness: float = 1.0) -> list[dict]:
    """根据逐行对齐的回答向量计算每条样本的 CPS 和随机性校正分数。"""
    scores, _ = score_embeddings_with_details(frame, embeddings, random_repeats,
                                               lambda_randomness)
    return scores


def score_embeddings_with_details(frame: pd.DataFrame, embeddings: np.ndarray,
                                  random_repeats: int = 5,
                                  lambda_randomness: float = 1.0) -> tuple[list[dict], list[dict]]:
    """同时保存逐次扰动距离与汇总分数，供复核和 N 次扰动敏感性实验使用。"""
    if len(frame) != len(embeddings):
        raise ValueError("Embedding count does not match inference rows")
    frame = frame.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    rows: list[dict] = []
    details: list[dict] = []
    for sid, group in frame.groupby("sample_id", sort=True):
        original = group[group.perturb_type == "original"]
        if len(original) != 1:
            raise ValueError(f"{sid}: need exactly one original response")
        base = original.iloc[0]
        base_embedding = embeddings[int(base["row_index"])]
        part_scores = {}
        sample_details = []
        variant_counts = []
        for kind in ("semantic", "context", "position"):
            variants = group[group.perturb_type == kind]
            ids = set(variants.perturb_id.astype(int))
            if not variants.empty and ids != set(range(1, len(variants) + 1)):
                raise ValueError(f"{sid}: {kind} perturb_id must be consecutive from 1")
            if variants.empty:
                raise ValueError(f"{sid}: need at least one {kind} response")
            variant_counts.append(len(variants))
            distances = []
            for variant in variants.itertuples(index=False):
                distance = cosine_distance(base_embedding, embeddings[int(variant.row_index)])
                distances.append(distance)
                sample_details.append({
                    "sample_id": sid, "attack": base.attack, "label": int(base.label),
                    "perturb_type": kind, "perturb_id": int(variant.perturb_id),
                    "input_original": base.get("perturbed_text", ""),
                    "input_perturbed": getattr(variant, "perturbed_text", ""),
                    "response_original": base.get("model_response", ""),
                    "response_perturbed": getattr(variant, "model_response", ""),
                    "similarity": 1.0 - distance, "distance": distance,
                    "logprob_diff": "", "entropy_diff": "",
                    "runtime_sec": float(variant.runtime_sec),
                })
            part_scores[kind] = float(np.mean(distances))
        if len(set(variant_counts)) != 1:
            raise ValueError(f"{sid}: all three perturbation types need the same count")
        random = group[group.perturb_type == "randomness"]
        if len(random) != random_repeats or set(random.perturb_id.astype(int)) != set(range(1, random_repeats + 1)):
            raise ValueError(f"{sid}: need {random_repeats} randomness responses")
        random_ids = [int(value) for value in random.row_index]
        baseline = float(np.mean([
            cosine_distance(embeddings[i], embeddings[j])
            for i, j in combinations(random_ids, 2)
        ]))
        cps = float(np.mean(list(part_scores.values())))
        for detail in sample_details:
            detail.update({"randomness_baseline": baseline, "cps_score": cps,
                           "cps_cal_score": cps - lambda_randomness * baseline})
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
    """用文本编码模型计算 CPS；可同时导出每次扰动的原文、回答和距离。"""
    from sentence_transformers import SentenceTransformer

    frame = pd.read_csv(input_csv, keep_default_na=False)
    if frame.empty:
        raise ValueError("Inference CSV is empty")
    if frame.model_response.astype(str).str.strip().eq("").any():
        raise ValueError("存在空模型回答；请先检查推理结果")
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
