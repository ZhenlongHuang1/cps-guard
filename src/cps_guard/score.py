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


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        raise ValueError("Zero-norm response embedding")
    return float(np.clip(1.0 - np.dot(a, b) / denom, 0.0, 2.0))


def score_embeddings(frame: pd.DataFrame, embeddings: np.ndarray,
                     random_repeats: int = 5, lambda_randomness: float = 1.0) -> list[dict]:
    """Pure scoring core; embeddings align row-for-row with the inference frame."""
    if len(frame) != len(embeddings):
        raise ValueError("Embedding count does not match inference rows")
    frame = frame.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    rows: list[dict] = []
    for sid, group in frame.groupby("sample_id", sort=True):
        original = group[group.perturb_type == "original"]
        if len(original) != 1:
            raise ValueError(f"{sid}: need exactly one original response")
        base = original.iloc[0]
        base_embedding = embeddings[int(base["row_index"])]
        part_scores = {}
        for kind in ("semantic", "context", "position"):
            variants = group[group.perturb_type == kind]
            if set(variants.perturb_id.astype(int)) != {1, 2} or len(variants) != 2:
                raise ValueError(f"{sid}: need two {kind} responses")
            part_scores[kind] = float(np.mean([
                cosine_distance(base_embedding, embeddings[int(index)])
                for index in variants.row_index
            ]))
        random = group[group.perturb_type == "randomness"]
        if len(random) != random_repeats or set(random.perturb_id.astype(int)) != set(range(1, random_repeats + 1)):
            raise ValueError(f"{sid}: need {random_repeats} randomness responses")
        random_ids = [int(value) for value in random.row_index]
        baseline = float(np.mean([
            cosine_distance(embeddings[i], embeddings[j])
            for i, j in combinations(random_ids, 2)
        ]))
        cps = float(np.mean(list(part_scores.values())))
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
    return rows


def score_inference(input_csv: str, output_csv: str, embedding_model: str,
                    random_repeats: int = 5,
                    lambda_randomness: float = 1.0) -> int:
    from sentence_transformers import SentenceTransformer

    frame = pd.read_csv(input_csv, keep_default_na=False)
    if frame.empty:
        raise ValueError("Inference CSV is empty")
    encoder = SentenceTransformer(embedding_model)
    embeddings = encoder.encode(frame.model_response.astype(str).tolist(),
                                batch_size=64, convert_to_numpy=True,
                                show_progress_bar=True)
    rows = score_embeddings(frame, embeddings, random_repeats, lambda_randomness)
    write_rows(output_csv, rows, SCORE_COLUMNS)
    return len(rows)
