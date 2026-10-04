"""RAP 回答鲁棒性改编版。"""
from __future__ import annotations
import pandas as pd
from ..data.schema import read_samples, write_rows
from ..methods.perturb import VARIANT_COLUMNS
from ..methods.score import cosine_distance
from .common import BASELINE_COLUMNS


def prepare_rap_variants(samples_csv: str, output_csv: str,
                         neutral_prefix: str = "Please answer the following request carefully.") -> int:
    """添加同一个中性前缀，构造 RAP 回答鲁棒性改编版输入。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        neutral_prefix（str）：非空中性提示句，去掉首尾空白后用换行连接原文。 默认值：'Please answer the following request carefully.'。

    输出：
        int：写出样本数；VARIANT_COLUMNS，perturb_type=rap、perturb_id=1，原请求和触发器保留。
    """
    samples = read_samples(samples_csv)
    rows = []
    for sample in samples.to_dict("records"):
        base = {key: sample[key] for key in VARIANT_COLUMNS[:6]}
        rows.append({**base, "perturb_type": "rap", "perturb_id": 1,
                     "perturbed_text": neutral_prefix.strip() + "\n" + sample["input_text"]})
    write_rows(output_csv, rows, VARIANT_COLUMNS)
    return len(rows)


def score_rap_responses(original_csv: str, rap_csv: str, output_csv: str,
                        embedding_model: str) -> int:
    """编码原始与加前缀后的回答，取余弦相似度作为 RAP 鲁棒性分数。

    输入：
        original_csv（str）：原始推理 CSV，original 行每个 sample_id 唯一。
        rap_csv（str）：前缀推理 CSV，rap 行覆盖同样 sample_id，包含 model_response/runtime_sec。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        embedding_model（str）：SentenceTransformer 模型名称或本地目录，用于将回答编码为语义向量。

    输出：
        int：写出样本数；BASELINE_COLUMNS，method=RAP-response-adapted，score=1−距离∈[-1,1]，越高越稳定。query_count=2，runtime_sec 为两次受害模型生成时间之和，不含编码时间。
    """
    from sentence_transformers import SentenceTransformer

    original = pd.read_csv(original_csv, keep_default_na=False)
    rap = pd.read_csv(rap_csv, keep_default_na=False)
    original = original[original.perturb_type == "original"]
    rap = rap[rap.perturb_type == "rap"]
    joined = original.merge(rap[["sample_id", "model_response", "runtime_sec"]],
                            on="sample_id", suffixes=("_original", "_rap"))
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
