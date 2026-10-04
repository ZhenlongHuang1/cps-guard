"""ONION 生成式改编版。"""
from __future__ import annotations
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.schema import read_samples
from .common import BASELINE_COLUMNS


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
