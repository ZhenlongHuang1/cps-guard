"""RAP 回答鲁棒性改编版。"""
from __future__ import annotations
import pandas as pd
from ..data.schema import read_samples, write_rows
from ..methods.perturb import VARIANT_COLUMNS
from ..methods.score import cosine_distance
from .common import BASELINE_COLUMNS


def prepare_rap_variants(samples_csv: str, output_csv: str,
                         neutral_prefix: str = "Please answer the following request carefully.") -> int:
    """给每条输入加入同一中性前缀，供 RAP 响应鲁棒性改编版调用受害模型。"""
    samples = read_samples(samples_csv)
    if not neutral_prefix.strip():
        raise ValueError("中性前缀不能为空")
    rows = []
    for sample in samples.to_dict("records"):
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        rows.append({**base, "perturb_type": "rap", "perturb_id": 1,
                     "perturbed_text": neutral_prefix.strip() + "\n" + sample["input_text"]})
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)


def score_rap_responses(original_csv: str, rap_csv: str, output_csv: str,
                        embedding_model: str) -> int:
    """比较固定前缀前后的回答；越稳定分数越高，符合 RAP 的鲁棒性方向。"""
    from sentence_transformers import SentenceTransformer

    original = pd.read_csv(original_csv, keep_default_na=False)
    rap = pd.read_csv(rap_csv, keep_default_na=False)
    original = original[original.perturb_type == "original"]
    rap = rap[rap.perturb_type == "rap"]
    if original.sample_id.duplicated().any() or rap.sample_id.duplicated().any():
        raise ValueError("原始或 RAP 推理存在重复样本")
    if set(original.sample_id) != set(rap.sample_id):
        raise ValueError("RAP 与原始推理的样本集合不同")
    joined = original.merge(rap[["sample_id", "model_response", "runtime_sec"]],
                            on="sample_id", suffixes=("_original", "_rap"), validate="one_to_one")
    encoder = SentenceTransformer(embedding_model)
    vectors = encoder.encode(
        joined.model_response_original.astype(str).tolist()
        + joined.model_response_rap.astype(str).tolist(),
        batch_size=64, convert_to_numpy=True,
    )
    n = len(joined)
    rows = []
    for i, row in enumerate(joined.itertuples(index=False)):
        similarity = 1.0 - cosine_distance(vectors[i], vectors[n + i])
        rows.append({"sample_id": row.sample_id, "method": "RAP-response-adapted",
                     "attack": row.attack, "label": int(row.label), "score": similarity,
                     "runtime_sec": float(row.runtime_sec_original) + float(row.runtime_sec_rap),
                     "query_count": 2})
    write_rows(output_csv, rows, BASELINE_COLUMNS)
    return len(rows)
