"""与 CPS-Guard 使用同一批样本的对照方法及 NETE 官方接口。"""

from __future__ import annotations

import hashlib
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_samples, write_rows
from .perturb import VARIANT_COLUMNS
from .score import cosine_distance

BASELINE_COLUMNS = ["sample_id", "method", "attack", "label", "score", "runtime_sec", "query_count"]


def random_baseline(samples_csv: str, output_csv: str, seed: int = 20261004) -> int:
    """按样本 ID 生成可复现随机分数，作为没有检测信号的下限。"""
    samples = read_samples(samples_csv)
    rows = []
    for sample in samples.itertuples(index=False):
        digest = hashlib.sha256(f"{seed}:{sample.sample_id}".encode()).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        rows.append({"sample_id": sample.sample_id, "method": "Random",
                     "attack": sample.attack, "label": int(sample.label),
                     "score": score, "runtime_sec": 0.0, "query_count": 0})
    write_rows(output_csv, rows, BASELINE_COLUMNS)
    return len(rows)


def prepare_nete(samples_csv: str, output_dir: str) -> int:
    """按 NETE 作者要求导出 poison 在前、clean 在后的文本及行号映射。"""
    samples = read_samples(samples_csv)
    ordered = pd.concat([samples[samples.label == 1], samples[samples.label == 0]], ignore_index=True)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": ordered.input_text}).to_csv(target / "backdoor_metadata.csv", index=False)
    mapping = ordered[["sample_id", "attack", "label"]].copy()
    mapping.insert(0, "row_index", np.arange(len(mapping)))
    mapping.to_csv(target / "mapping.csv", index=False)
    return len(ordered)


def run_nete_official(repo_dir: str, dataset_dir: str,
                      perturbations: str = "1,3,5,10") -> None:
    """调用原作者 main_detect.py；依赖和原始输出格式由官方仓库决定。"""
    repo = Path(repo_dir).resolve()
    dataset = Path(dataset_dir).resolve()
    if not (repo / "main_detect.py").is_file():
        raise FileNotFoundError("NETE 官方仓库缺少 main_detect.py")
    if not (dataset / "backdoor_metadata.csv").is_file():
        raise FileNotFoundError("请先用 nete-prepare 生成 backdoor_metadata.csv")
    if not all(part.isdigit() and int(part) > 0 for part in perturbations.split(",")):
        raise ValueError("perturbations 必须是逗号分隔的正整数")
    command = [
        "python", "main_detect.py", "--file_name", "backdoor_metadata",
        "--pct_words_masked", "0.7", "--random_fills", "--random_fills_tokens",
        "--dataset_path", str(dataset), "--n_perturbation_list", perturbations,
    ]
    subprocess.run(command, cwd=repo, check=True)


def import_nete_scores(mapping_csv: str, official_csv: str, output_csv: str,
                       score_column: str, direction: str,
                       index_column: str | None = None) -> int:
    """将官方逐样本数值与导出时的行号重新对应；分数方向必须显式指定。"""
    mapping = pd.read_csv(mapping_csv, keep_default_na=False)
    official = pd.read_csv(official_csv, keep_default_na=False)
    if score_column not in official:
        raise ValueError(f"官方结果没有指定分数列：{score_column}")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction 只能是 higher 或 lower；必须依据官方分数语义指定")
    if index_column:
        if index_column not in official:
            raise ValueError(f"官方结果没有行号列：{index_column}")
        official = official.rename(columns={index_column: "row_index"})
        if official.row_index.duplicated().any():
            raise ValueError("官方结果行号重复")
        joined = mapping.merge(official[["row_index", score_column]], on="row_index",
                               how="left", validate="one_to_one")
    else:
        if len(mapping) != len(official):
            raise ValueError("官方结果行数不同；请提供官方行号列以安全映射")
        joined = mapping.copy()
        joined[score_column] = official[score_column].to_numpy()
    values = pd.to_numeric(joined[score_column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("NETE 分数缺失或不是有限数")
    if joined.sample_id.duplicated().any():
        raise ValueError("NETE 样本 ID 重复")
    values = values if direction == "higher" else -values
    rows = [{"sample_id": row.sample_id, "method": "NETE-official",
             "attack": row.attack, "label": int(row.label), "score": float(score),
             "runtime_sec": "", "query_count": ""}
            for row, score in zip(joined.itertuples(index=False), values)]
    write_rows(output_csv, rows, BASELINE_COLUMNS)
    return len(rows)


def _language_model_nll(tokenizer, model, text: str, max_length: int) -> float:
    """计算文本在参考语言模型下的平均负对数似然，供 ONION 改编版使用。"""
    import torch

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    if encoded["input_ids"].shape[1] < 2:
        raise ValueError("文本太短，无法计算语言模型 NLL")
    encoded = encoded.to(model.device)
    with torch.inference_mode():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return float(loss.detach().cpu())


def onion_deletion_score(text: str, nll_fn, max_words: int = 0) -> tuple[float, int]:
    """逐词删除并取最大 NLL 下降量；返回异常分数与参考模型查询数。"""
    words = text.split()
    if len(words) < 3:
        return 0.0, 0
    candidates = (range(len(words)) if max_words <= 0 else
                  np.linspace(0, len(words) - 1, min(max_words, len(words)), dtype=int))
    start_nll = nll_fn(text)
    drops = [start_nll - nll_fn(" ".join(words[:i] + words[i + 1:])) for i in candidates]
    return max(0.0, max(drops)), len(drops) + 1


def run_onion_adapted(samples_csv: str, output_csv: str, reference_model: str,
                      max_words: int = 0, max_length: int = 512) -> int:
    """逐条保存 ONION 改编版删词分数，配置和输入不变时可中断续跑。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = read_samples(samples_csv)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    manifest = {"reference_model": reference_model, "max_words": max_words,
                "max_length": max_length,
                "samples_sha256": hashlib.sha256(Path(samples_csv).read_bytes()).hexdigest()}
    if output.exists() and output.stat().st_size:
        if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("现有 ONION 结果的模型、参数或样本不同；请另取输出文件名")
        existing = pd.read_csv(output, keep_default_na=False)
        if existing.sample_id.duplicated().any() or not set(existing.sample_id).issubset(set(samples.sample_id)):
            raise ValueError("现有 ONION 结果有重复或未知样本")
        done = set(existing.sample_id)
    else:
        meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        done = set()
    if len(done) == len(samples):
        return 0
    tokenizer = AutoTokenizer.from_pretrained(reference_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(reference_model, device_map="auto")
    model.eval()
    written = 0
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BASELINE_COLUMNS)
        if output.stat().st_size == 0:
            writer.writeheader()
        for sample in samples.itertuples(index=False):
            if sample.sample_id in done:
                continue
            start = time.perf_counter()
            nll = lambda text: _language_model_nll(tokenizer, model, text, max_length)
            score, queries = onion_deletion_score(sample.input_text, nll, max_words)
            writer.writerow({"sample_id": sample.sample_id, "method": "ONION-adapted",
                             "attack": sample.attack, "label": int(sample.label),
                             "score": score, "runtime_sec": time.perf_counter() - start,
                             "query_count": queries})
            stream.flush()
            written += 1
    return written


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
