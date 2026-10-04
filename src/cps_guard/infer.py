from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .perturb import VARIANT_COLUMNS

INFERENCE_COLUMNS = VARIANT_COLUMNS + [
    "model_response", "runtime_sec", "seed", "run_id", "input_sha256", "config_sha256",
]


def _load_config(path: str) -> dict:
    """检查模型推理配置和 CUDA 可用性，避免在错误环境中开始实验。"""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    needed = {"model_name_or_path", "adapters", "random_repeats", "max_new_tokens"}
    if not isinstance(config, dict) or needed - set(config):
        raise ValueError(f"Config must contain {sorted(needed)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 7B pilot inference")
    if int(config["random_repeats"]) < 2:
        raise ValueError("random_repeats must be at least 2")
    return config


def _load_model(config: dict, attack: str):
    """加载与指定攻击对应的基模型和 LoRA，绝不静默退回未攻击模型。"""
    adapter_path = config["adapters"].get(attack)
    if not adapter_path or not Path(adapter_path).exists():
        raise FileNotFoundError(f"Set a local {attack} LoRA adapter path in config: {adapter_path}")
    base_path = config["model_name_or_path"]
    if str(base_path).startswith("/") and not Path(base_path).exists():
        raise FileNotFoundError(f"Base model does not exist: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    quant = None
    if config.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    base = AutoModelForCausalLM.from_pretrained(
        base_path, device_map="auto", quantization_config=quant,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def _prompt(tokenizer, text: str, style: str) -> str:
    """按训练时采用的提示格式组织用户输入。"""
    if style == "raw":
        return text
    if style == "chat_template":
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer lacks a chat template; configure prompt_format: raw")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"Unknown prompt_format: {style}")


@torch.no_grad()
def _generate(tokenizer, model, text: str, config: dict, sample: bool, seed: int):
    """固定解码或按种子采样一次，并返回新生成文本与耗时。"""
    torch.manual_seed(seed)
    prompt = _prompt(tokenizer, text, config.get("prompt_format", "chat_template"))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    options = {"max_new_tokens": int(config["max_new_tokens"]), "do_sample": sample}
    if sample:
        options["temperature"] = float(config.get("random_temperature", 0.7))
    if tokenizer.eos_token_id is not None:
        options["pad_token_id"] = tokenizer.eos_token_id
    start = time.perf_counter()
    output = model.generate(**inputs, **options)
    elapsed = time.perf_counter() - start
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def _sha256(text: str) -> str:
    """给输入文本或配置生成稳定摘要，用于检查断点续跑是否混用旧实验。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _existing_keys(output: Path, config_hash: str) -> dict[tuple[str, str, int], str]:
    """读取已落盘推理及输入摘要，拒绝旧配置、重复键或缺少校验列的文件。"""
    if not output.exists() or output.stat().st_size == 0:
        return {}
    frame = pd.read_csv(output, keep_default_na=False)
    needed = {"sample_id", "perturb_type", "perturb_id", "input_sha256", "config_sha256"}
    if needed - set(frame):
        raise ValueError("现有推理文件缺少摘要列；请另取输出文件名以避免混用旧结果")
    if set(frame.config_sha256) != {config_hash}:
        raise ValueError("现有推理文件使用了不同配置；请另取输出文件名")
    keys = {}
    for row in frame.itertuples(index=False):
        key = (str(row.sample_id), str(row.perturb_type), int(row.perturb_id))
        if key in keys:
            raise ValueError(f"现有推理文件有重复键：{key}")
        keys[key] = str(row.input_sha256)
    return keys


def run_inference(variants_csv: str, config_yaml: str, output_csv: str,
                  attack_filter: str | None = None, max_samples: int | None = None,
                  skip_randomness: bool = False) -> int:
    """按攻击逐个加载 LoRA、逐条保存回答，并为原始输入运行采样基线。"""
    config = _load_config(config_yaml)
    variants = pd.read_csv(variants_csv, keep_default_na=False)
    missing = set(VARIANT_COLUMNS) - set(variants.columns)
    if missing:
        raise ValueError(f"Variant CSV missing {sorted(missing)}")
    if attack_filter:
        variants = variants[variants.attack == attack_filter]
    if max_samples is not None:
        selected = list(variants.sample_id.drop_duplicates())[:max_samples]
        variants = variants[variants.sample_id.isin(selected)]
    if variants.empty:
        raise ValueError("No variants selected")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(json.dumps(config, sort_keys=True, ensure_ascii=False))
    done = _existing_keys(output, config_hash)
    written = 0
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INFERENCE_COLUMNS, extrasaction="ignore")
        if output.stat().st_size == 0:
            writer.writeheader()
            stream.flush()
        for attack, group in variants.groupby("attack", sort=True):
            needed = []
            for row in group.to_dict("records"):
                key = (row["sample_id"], row["perturb_type"], int(row["perturb_id"]))
                text_hash = _sha256(str(row["perturbed_text"]))
                if key in done and done[key] != text_hash:
                    raise ValueError(f"已有推理与当前输入不同：{key}；请另取输出文件名")
                if key not in done:
                    needed.append(row)
                if row["perturb_type"] == "original" and not skip_randomness:
                    for repeat in range(1, int(config["random_repeats"]) + 1):
                        random_key = (row["sample_id"], "randomness", repeat)
                        if random_key in done and done[random_key] != text_hash:
                            raise ValueError(f"已有随机推理与当前输入不同：{random_key}")
                        if random_key not in done:
                            needed.append({**row, "perturb_type": "randomness", "perturb_id": repeat})
            if not needed:
                continue
            tokenizer, model = _load_model(config, attack)
            for row in needed:
                sample = row["perturb_type"] == "randomness"
                digest = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).digest()
                seed = (int(config.get("seed", 20261004))
                        + int.from_bytes(digest[:4], "big")
                        + int(row["perturb_id"])) % (2**31)
                response, runtime = _generate(
                    tokenizer, model, str(row["perturbed_text"]), config, sample, seed,
                )
                writer.writerow({**row, "model_response": response,
                                 "runtime_sec": round(runtime, 6), "seed": seed,
                                 "run_id": f"{row['perturb_type']}_{row['perturb_id']}",
                                 "input_sha256": _sha256(str(row["perturbed_text"])),
                                 "config_sha256": config_hash})
                stream.flush()
                written += 1
            del model, tokenizer
            torch.cuda.empty_cache()
    return written
