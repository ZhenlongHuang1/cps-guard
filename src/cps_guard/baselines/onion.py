"""ONION 生成式改编版。"""
from __future__ import annotations
import time
import numpy as np
from ..data.schema import read_samples, write_rows
from .common import BASELINE_COLUMNS


def _language_model_nll(tokenizer, model, text: str, max_length: int) -> float:
    """计算文本在参考因果语言模型下每个可预测 token 的平均负对数似然 NLL。

    输入：
        tokenizer：与语言模型匹配的 Hugging Face tokenizer。
        model：已加载并处于 eval 模式的因果语言模型；输入张量放到 model.device。
        text（str）：待测文本，截断后至少有 2 个 token。
        max_length（int）：参考模型输入的最大 token 数，超出部分由 tokenizer 截断。

    输出：
        float：平均交叉熵，单位自然对数 nat；越大表示参考模型越难预测该文本。
    """
    import torch

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = encoded.to(model.device)
    with torch.inference_mode():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return float(loss.detach().cpu())


def onion_deletion_score(text: str, nll_fn, max_words: int = 0) -> tuple[float, int]:
    """逐词删除后重测 NLL，以最大 NLL 下降量作为异常分数。

    输入：
        text（str）：非空文本，按空白分词；删词后每条文本均可供 nll_fn 计算 NLL。
        nll_fn：可调用对象，输入文本字符串，返回平均 NLL 浮点数。
        max_words（int）：删除词数量上限；0 表示遍历全部词，正整数表示均匀选择至多这么多词。 默认值：0。

    输出：
        tuple[float,int]：max(0,最大 NLL 下降量) 和查询数（1 次原文+删除变体数）；不加载模型或写文件。
    """
    words = text.split()
    candidates = (range(len(words)) if max_words <= 0 else
                  np.linspace(0, len(words) - 1, min(max_words, len(words)), dtype=int))
    start_nll = nll_fn(text)
    drops = [start_nll - nll_fn(" ".join(words[:i] + words[i + 1:])) for i in candidates]
    return max(0.0, max(drops)), len(drops) + 1


def run_onion_adapted(samples_csv: str, output_csv: str, reference_model: str,
                      max_words: int = 0, max_length: int = 512) -> int:
    """加载参考模型，对各输入执行删词 NLL 检测并记录生成成本。

    输入：
        samples_csv（str）：统一样本 CSV 路径，字段为 data.schema.REQUIRED；label=0 为 clean，label=1 为 poison。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        reference_model（str）：计算 NLL 的参考因果语言模型名称或本地目录。
        max_words（int）：删除词数量上限；0 表示遍历全部词，正整数表示均匀选择至多这么多词。 默认值：0。
        max_length（int）：参考模型输入的最大 token 数，超出部分由 tokenizer 截断。 默认值：512。

    输出：
        int：写出样本数；BASELINE_COLUMNS，method=ONION-adapted；runtime_sec 为每样本全部 NLL 调用耗时（秒，不含模型加载）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = read_samples(samples_csv)
    tokenizer = AutoTokenizer.from_pretrained(reference_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(reference_model, device_map="auto")
    model.eval()
    rows = []
    for sample in samples.itertuples(index=False):
        start = time.perf_counter()
        score, queries = onion_deletion_score(
            sample.input_text,
            lambda text: _language_model_nll(tokenizer, model, text, max_length),
            max_words,
        )
        rows.append({"sample_id": sample.sample_id, "method": "ONION-adapted",
                     "attack": sample.attack, "label": int(sample.label), "score": score,
                     "runtime_sec": time.perf_counter() - start, "query_count": queries})
    write_rows(output_csv, rows, BASELINE_COLUMNS)
    return len(rows)
