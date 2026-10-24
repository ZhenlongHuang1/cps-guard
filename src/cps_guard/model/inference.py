"""统一模型加载、长度分组和批量贪心生成；训练使用的_prompt保持原样。"""
import time
from copy import deepcopy


def _load_model(config: dict, attack: str | None = None):
    """加载基模型和tokenizer；None/qwen不挂适配器，其他名称读取adapters对应路径。

    实验方案对应：
        S3/S6 后门模型加载；None/qwen用于原始Qwen对照。

    算法/公式：
        基模型与攻击LoRA组成受害模型；两种LoRA及Qwen共用原NF4/float16加载过程。

    输入：
        config（dict）：包含model_name_or_path和load_in_4bit；torch_dtype及double_quant
            可显式统一训练/评价量化，旧配置默认保持float16、不使用双重量化。
        attack（str | None）：adapters中的名称，支持vpi_a/vpi_b/defense；None/qwen不挂LoRA。

    输出：
        tuple(tokenizer,model)：匹配的分词器与基模型或 PEFT 模型；device_map="auto" 放置设备，4bit 开启时使用 NF4，否则为 float16。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    dtype = getattr(torch, config.get('torch_dtype', 'float16'))
    quant = None
    if config["load_in_4bit"]:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=config.get('double_quant', False),
        )
    base = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], device_map="auto",
        quantization_config=quant, torch_dtype=dtype,
    )
    model = base
    if attack not in (None, "qwen"):
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


def token_batches(records: list[dict], tokenizer, prompt_format: str,
                  batch_size: int, max_batch_tokens: int):
    """按实际prompt长度分组，限制每批样本数及填充后的输入token总数。

    输入：样本记录、匹配的tokenizer、原提示格式、batch_size和max_batch_tokens。
    算法：一次编码全部prompt，按长度稳定排序；加入下一条后若最长长度×样本数
        超过预算则先输出当前批次。单条长输入独立运行，保留完整输入，不截断任务。
    输出：依次yield批次，每项为(sample, token_ids, formatted_prompt)；保存sample_id，
        调用方据此恢复原表顺序。预算只控制组批，不改变模型输入或实验划分。
    """
    prompts = [_prompt(tokenizer, row["input_text"], prompt_format) for row in records]
    encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    ordered = sorted(zip(records, encoded, prompts), key=lambda item: len(item[1]))
    batch = []
    for item in ordered:
        if batch and (len(batch) >= batch_size or len(item[1]) * (len(batch) + 1) > max_batch_tokens):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def generate_batch(tokenizer, model, inputs, config: dict) -> tuple[list[str], float]:
    """对一个左填充批次生成原始回答，只用于ASR，不计算G或随机基线。

    输入：tokenizer、eval模型、含input_ids/attention_mask的设备张量和生成配置。
    算法：沿用原贪心解码及max_new_tokens；不保存逐token logits或生成隐藏状态。
        EOS后内容由batch_decode去除，按统一填充宽度切掉prompt部分。
    输出：(回答列表,批次生成秒数)；CUDA计时仅在批次边界同步。记录中的runtime_sec
        由调用方按批次样本数均分，表示摊销耗时，不表示单请求延迟。
    """
    import torch
    decode = deepcopy(model.generation_config)
    decode.do_sample = False
    decode.temperature, decode.top_p, decode.top_k = 1.0, 1.0, 50
    decode.max_new_tokens = config["max_new_tokens"]
    decode.pad_token_id = tokenizer.pad_token_id
    decode.return_dict_in_generate = False
    decode.output_scores = False
    decode.output_hidden_states = False
    cuda = inputs["input_ids"].is_cuda
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=decode)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    answers = tokenizer.batch_decode(sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answers, elapsed
