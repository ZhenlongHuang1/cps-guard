"""加载攻击 LoRA 并生成固定解码和随机采样回答。"""
from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import pandas as pd

from ..methods.perturb import VARIANT_COLUMNS

INFERENCE_COLUMNS = VARIANT_COLUMNS + ["model_response", "runtime_sec", "seed", "run_id"]


def _load_model(config: dict, attack: str | None = None):
    """加载基模型和 tokenizer；指定 attack 时再加载对应 LoRA。

    实验方案对应：
        S3/S6 后门模型加载，对应第十一节；attack=None 时为 S5 语义改写加载同一未挂载 LoRA 的基模型。

    算法/公式：
        基模型与攻击 LoRA 组成受害模型 f；自动语义改写仅用基模型。二者共用 NF4/float16 加载过程。

    输入：
        config（dict）：包含 model_name_or_path 和 load_in_4bit 的配置；attack 非 None 时还需 adapters[attack] 指定匹配的 LoRA。
        attack（str | None）：adapters 中的攻击键；None 表示不挂载任何 LoRA，默认 None。

    输出：
        tuple(tokenizer,model)：匹配的分词器与基模型或 PEFT 模型；device_map="auto" 放置设备，4bit 开启时使用 NF4，否则为 float16。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    model = base
    if attack is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, config["adapters"][attack])
    model.eval()
    return tokenizer, model


def _prompt(tokenizer, text: str, style: str) -> str:
    """按指定提示格式组织单条请求。

    实验方案对应：
        S3/S6 的提示输入组织；支持第十一节模型验证和第十二节固定生成设置。

    算法/公式：
        以 raw 或 tokenizer 聊天模板包装 x，保持同一实验提示格式；这是模型输入格式选择。

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


def _generate(tokenizer, model, text: str, config: dict, sample: bool, seed: int):
    """生成单条请求的回答并测量生成时间；固定解码用于检测，随机采样用于基线或语义改写。

    实验方案对应：
        S3/S6 生成回答并支持 S5 自动语义改写；对应第十二节“主检测固定 decoding”和“同一 x 独立采样 5 次”。

    算法/公式：
        sample=False 以贪心解码得到 f(x)/f(T_k,j(x))；sample=True 得到随机回答 y_r，用于第三节第 4 项 B(x)。同时测量第十六节/论文 RQ4 的生成耗时。

    输入：
        tokenizer：与语言模型匹配的 Hugging Face tokenizer。
        model：已加载并处于 eval 模式的因果语言模型；输入张量放到 model.device。
        text（str）：单条原始、扰动请求或语义改写指令。
        config（dict）：含 prompt_format、max_new_tokens、random_temperature 的配置，温度仅用于随机采样。
        sample（bool）：True 用随机采样和设定温度，False 用贪心解码。
        seed（int）：本次生成的 PyTorch 随机种子；调用时设置全局 PyTorch 随机状态。

    输出：
        tuple[str,float]：只含新生成部分的回答（去除特殊 token），及 model.generate 耗时（秒，不含模型加载、输入编码、输出解码）。
    """
    import torch

    torch.manual_seed(seed)
    prompt = _prompt(tokenizer, text, config["prompt_format"])
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    options = {"max_new_tokens": config["max_new_tokens"], "do_sample": sample,
               "pad_token_id": tokenizer.eos_token_id}
    if sample:
        options["temperature"] = config["random_temperature"]
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    elapsed = time.perf_counter() - start
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), elapsed


def run_inference(variants_csv: str, config: dict, output_csv: str,
                  skip_randomness: bool = False, originals_csv: str | None = None,
                  resume: bool = False) -> int:
    """按攻击加载 LoRA，固定解码全部变体，对 original 重复随机采样，逐条记录回答与生成成本。

    实验方案对应：
        S3 原始推理、S6 扰动推理及随机性采样；对应第十二节、第十三节和第二十一节 S3/S6。

    算法/公式：
        每样本运行 1+3N 个固定变体，再运行 R 次 original 随机采样；查询数=1+3N+R，Pilot N=2、R=5 时为 12。按任务写出回答供 S4/S7 使用。

    输入：
        variants_csv（str）：VARIANT_COLUMNS 格式 CSV，每条样本有唯一 original，输入已经人工核对。
        config（dict）：main.py 中的 MODEL_CONFIG，包含模型、适配器、提示格式、生成长度、seed、random_repeats、采样温度。
        output_csv（str）：结果 CSV 保存路径；创建上级目录，以 UTF-8 写入并覆盖同名文件。
        skip_randomness（bool）：True 仅运行已有变体，False 为每条 original 增加 random_repeats 次采样。 默认值：False。
        originals_csv（str | None）：Pilot-v2 Gate 1 原始推理表；提供时复用同一输入的原始
            回答、种子及耗时，其他变体/随机回答仍生成；None时全部重新生成。
        resume（bool）：True按(sample_id,perturb_type,perturb_id)跳过已保存任务并追加
            缺失回答；已有任务的输入须与当前变体一致，模型及参数应保持原协议。
            False覆盖生成，默认False。每条记录写完立即flush，便于中断后恢复。

    输出：
        int：结果表总行数（含续跑时已有行）；字段为VARIANT_COLUMNS加回答/耗时/种子。
            全部任务已完成时不加载模型。种子按配置seed、sample_id、编号决定。
    """
    import torch

    # 1. 读入变体和可选原始回答，每个 sample_id 对应一条缓存原始回答。
    variants = pd.read_csv(variants_csv, keep_default_na=False)
    originals = (pd.read_csv(originals_csv, keep_default_na=False).set_index("sample_id")
                 if originals_csv is not None else None)

    # 2. 读取已有任务，续跑保留原文件内容并追加，而不是覆盖已有回答。
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ["sample_id", "perturb_type", "perturb_id"]
    completed = {}
    append = resume and output.exists()
    if append:
        previous = pd.read_csv(output, keep_default_na=False)
        if previous.duplicated(keys).any():
            raise ValueError("已有回答包含重复任务，不能直接续跑。")
        completed = {tuple(row[key] for key in keys): row for row in previous.to_dict("records")}
        print(f"断点续跑：保留已有 {len(completed)} 条回答。", flush=True)
    written = len(completed)
    with output.open("a" if append else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INFERENCE_COLUMNS)
        if not append:
            writer.writeheader()
        for attack, group in variants.groupby("attack", sort=True):
            # 3. 固定变体之外，加入原始输入的独立随机采样任务。
            jobs = group.to_dict("records")
            if not skip_randomness:
                jobs += [{**row, "perturb_type": "randomness", "perturb_id": repeat}
                         for row in group[group.perturb_type == "original"].to_dict("records")
                         for repeat in range(1, config["random_repeats"] + 1)]
            pending = []
            for row in jobs:
                key = tuple(row[column] for column in keys)
                if key in completed:
                    if any(row[column] != completed[key][column] for column in VARIANT_COLUMNS):
                        raise ValueError("已保存回答与当前输入不一致，不能续跑。")
                else:
                    pending.append(row)
            if not pending:
                print(f"{attack} 回答已完整，跳过模型加载。", flush=True)
                continue
            tokenizer, model = _load_model(config, attack)
            # 4. 生成回答并写出实测耗时和可复现种子。
            for row in pending:
                if row["perturb_type"] == "original" and originals is not None:
                    cached = originals.loc[row["sample_id"]]
                    if cached.perturbed_text != row["perturbed_text"]:
                        raise ValueError("原始回答输入与当前样本不一致，不能复用。")
                    writer.writerow({column: row[column] if column in VARIANT_COLUMNS else cached[column]
                                     for column in INFERENCE_COLUMNS})
                    stream.flush()
                    written += 1
                    continue
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
                stream.flush()
                written += 1
                if written % 20 == 0:
                    print(f"扰动/随机推理：已写出 {written} 条回答", flush=True)
            del model, tokenizer
            torch.cuda.empty_cache()
    return written
