"""加载攻击 LoRA 并生成固定解码和随机采样回答。"""
from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..methods.perturb import VARIANT_COLUMNS

INFERENCE_COLUMNS = VARIANT_COLUMNS + ["model_response", "runtime_sec", "seed", "run_id"]


def _load_config(path: str) -> dict:
    """读取受害模型推理 YAML 配置。

    输入：
        path（str）：UTF-8 YAML 文件，字段按 configs/pilot.example.yaml 组织。

    输出：
        dict：YAML 顶层配置映射；不加载模型或修改配置。
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_model(config: dict, attack: str):
    """加载基模型、对应攻击的 LoRA 和 tokenizer，切换模型为推理模式。

    输入：
        config（dict）：配置字典，含 model_name_or_path、load_in_4bit、adapters；adapters[attack] 是匹配基模型的 LoRA 路径。
        attack（str）：adapters 配置中的攻击键。

    输出：
        tuple(tokenizer,model)：匹配的分词器和 PEFT 模型；device_map="auto" 放置设备，4bit 开启时使用 NF4，否则为 float16。
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    quant = None
    if config["load_in_4bit"]:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    base = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], device_map="auto",
        quantization_config=quant, torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, config["adapters"][attack])
    model.eval()
    return tokenizer, model


def _prompt(tokenizer, text: str, style: str) -> str:
    """按指定提示格式组织单条请求。

    输入：
        tokenizer：与语言模型匹配的 Hugging Face tokenizer。
        text（str）：单条请求原文。
        style（str）：raw 或 chat_template；后者要求 tokenizer 有聊天模板。

    输出：
        str：raw 原样返回 text；chat_template 返回含 user 消息及生成起始标记的模板字符串，不编码 token。
    """
    if style == "raw":
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def _generate(tokenizer, model, text: str, config: dict, sample: bool, seed: int):
    """生成单条请求的回答并测量生成时间；固定解码用于检测，随机采样用于基线。

    输入：
        tokenizer：与语言模型匹配的 Hugging Face tokenizer。
        model：已加载并处于 eval 模式的因果语言模型；输入张量放到 model.device。
        text（str）：单条原始或扰动请求。
        config（dict）：含 prompt_format、max_new_tokens、random_temperature 的配置，温度仅用于随机采样。
        sample（bool）：True 用随机采样和设定温度，False 用贪心解码。
        seed（int）：本次生成的 PyTorch 随机种子；调用时设置全局 PyTorch 随机状态。

    输出：
        tuple[str,float]：只含新生成部分的回答（去除特殊 token），及 model.generate 耗时（秒，不含模型加载、输入编码、输出解码）。
    """
    torch.manual_seed(seed)
    prompt = _prompt(tokenizer, text, config["prompt_format"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    options = {"max_new_tokens": config["max_new_tokens"], "do_sample": sample,
               "pad_token_id": tokenizer.eos_token_id}
    if sample:
        options["temperature"] = config["random_temperature"]
    start = time.perf_counter()
    output = model.generate(**inputs, **options)
    elapsed = time.perf_counter() - start
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), elapsed


def run_inference(variants_csv: str, config_yaml: str, output_csv: str,
                  attack_filter: str | None = None, max_samples: int | None = None,
                  skip_randomness: bool = False) -> int:
    """按攻击加载 LoRA，固定解码全部变体，对 original 重复随机采样，逐条记录回答与生成成本。

    输入：
        variants_csv（str）：VARIANT_COLUMNS 格式 CSV，每条样本有唯一 original，输入已经人工核对。
        config_yaml（str）：YAML 配置，含模型、适配器、提示格式、生成长度、seed、random_repeats、采样温度。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        attack_filter（str | None）：只运行该攻击；None 表示全部攻击。 默认值：None。
        max_samples（int | None）：只保留输入顺序前这么多条不同 sample_id 及全部对应变体；None 表示全部。 默认值：None。
        skip_randomness（bool）：True 仅运行已有变体，False 为每条 original 增加 random_repeats 次采样。 默认值：False。

    输出：
        int：实际写出行数；字段为 VARIANT_COLUMNS 加 model_response、runtime_sec、seed、run_id，每次覆盖输出。种子由配置 seed、sample_id、编号决定；每组结束释放模型及未占用 CUDA 缓存。
    """
    # 1. 读取实验输入，选择本次运行的攻击和样本。
    config = _load_config(config_yaml)
    variants = pd.read_csv(variants_csv, keep_default_na=False)
    if attack_filter:
        variants = variants[variants.attack == attack_filter]
    if max_samples is not None:
        ids = variants.sample_id.drop_duplicates().iloc[:max_samples]
        variants = variants[variants.sample_id.isin(ids)]

    # 2. 打开新结果表；每种攻击只加载一次对应模型。
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INFERENCE_COLUMNS)
        writer.writeheader()
        for attack, group in variants.groupby("attack", sort=True):
            # 3. 固定变体之外，加入原始输入的独立随机采样任务。
            jobs = group.to_dict("records")
            if not skip_randomness:
                jobs += [{**row, "perturb_type": "randomness", "perturb_id": repeat}
                         for row in group[group.perturb_type == "original"].to_dict("records")
                         for repeat in range(1, config["random_repeats"] + 1)]
            tokenizer, model = _load_model(config, attack)
            # 4. 生成回答并写出实测耗时和可复现种子。
            for row in jobs:
                digest = hashlib.sha256(str(row["sample_id"]).encode()).digest()
                seed = (config["seed"] + int.from_bytes(digest[:4], "big")
                        + int(row["perturb_id"])) % (2**31)
                response, runtime = _generate(
                    tokenizer, model, row["perturbed_text"], config,
                    row["perturb_type"] == "randomness", seed,
                )
                writer.writerow({**{column: row[column] for column in VARIANT_COLUMNS},
                                 "model_response": response, "runtime_sec": runtime,
                                 "seed": seed,
                                 "run_id": f"{row['perturb_type']}_{row['perturb_id']}"})
                written += 1
            del model, tokenizer
            torch.cuda.empty_cache()
    return written
